#!/usr/bin/env python3
"""
FastAPI backend for the benchmark dashboard.
Serves the UI and exposes endpoints for:
- Launching benchmark runs
- Streaming live progress via SSE (Server-Sent Events)
- Returning past results from results/full_suite.json
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import asyncio
import httpx
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

load_dotenv()

app = FastAPI(title="Apple Silicon LLM Benchmark")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

RESULTS_PATH = Path(__file__).parent.parent / "results" / "full_suite.json"
OLLAMA_URL = "http://localhost:11434"


class RunConfig(BaseModel):
    models: List[str]
    prompt_lengths: List[str]
    concurrency_levels: List[int]
    max_new_tokens: int = 128
    bench_runs: int = 3


PROMPT_MAP = {
    "short":  "Explain what machine learning is in one sentence.",
    "medium": "Explain how transformer attention mechanisms work, covering key-query-value structure.",
    "long":   "Explain the CAP theorem, how Cassandra handles consistency vs availability, and what eventual consistency means in practice.",
}


@app.get("/api/results")
def get_results():
    if not RESULTS_PATH.exists():
        return JSONResponse([])
    with open(RESULTS_PATH) as f:
        return JSONResponse(json.load(f))
    

@app.get("/api/advanced-results")
def get_advanced_results():
    advanced_path = Path(__file__).parent.parent / "results" / "advanced_results.json"
    if not advanced_path.exists():
        return JSONResponse({})
    with open(advanced_path) as f:
        return JSONResponse(json.load(f))


@app.get("/api/models")
def get_models():
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        return JSONResponse({"models": models, "status": "ok"})
    except Exception as e:
        return JSONResponse({"models": [], "status": "error", "error": str(e)})


@app.post("/api/run")
async def run_benchmark(config: RunConfig):
    """
    Stream benchmark progress via Server-Sent Events.
    SSE = the server keeps the HTTP connection open and pushes
    events as they happen. The browser receives them in real time.
    Format: 'data: {json}\n\n' — the double newline is required by SSE spec.
    """
    async def event_stream():
        import time
        import uuid

        results = []
        total = len(config.models) * len(config.prompt_lengths) * len(config.concurrency_levels)
        condition = 0

        for model in config.models:
            for prompt_length in config.prompt_lengths:
                prompt = PROMPT_MAP.get(prompt_length, PROMPT_MAP["short"])
                for concurrency in config.concurrency_levels:
                    condition += 1

                    yield f"data: {json.dumps({'type': 'progress', 'condition': condition, 'total': total, 'model': model, 'prompt_length': prompt_length, 'concurrency': concurrency, 'status': 'running'})}\n\n"

                    run_results = []
                    for run_num in range(config.bench_runs):
                        try:
                            # Run concurrent requests
                            async with httpx.AsyncClient(timeout=300) as client:
                                tasks = [
                                    _async_request(client, model, prompt, config.max_new_tokens, prompt_length, concurrency)
                                    for _ in range(concurrency)
                                ]
                                batch = await asyncio.gather(*tasks, return_exceptions=True)
                                batch = [r for r in batch if isinstance(r, dict)]

                            if batch:
                                avg = {
                                    "framework": "ollama",
                                    "model": model,
                                    "prompt_length": prompt_length,
                                    "concurrency": concurrency,
                                    "throughput_tok_per_sec": sum(r["throughput"] for r in batch) / len(batch),
                                    "ttft_ms": sum(r["ttft_ms"] for r in batch) / len(batch),
                                    "total_latency_ms": sum(r["total_ms"] for r in batch) / len(batch),
                                    "tokens_generated": sum(r["tokens"] for r in batch),
                                    "run_id": str(uuid.uuid4())[:8],
                                    "timestamp": time.time(),
                                    "batch_size": concurrency,
                                    "peak_memory_mb": 0,
                                    "prompt_tokens": batch[0].get("prompt_tokens", 0),
                                    "error": None,
                                }
                                run_results.append(avg)

                                yield f"data: {json.dumps({'type': 'run_complete', 'run': run_num + 1, 'total_runs': config.bench_runs, 'throughput': round(avg['throughput_tok_per_sec'], 1), 'ttft_ms': round(avg['ttft_ms'], 0)})}\n\n"

                        except Exception as e:
                            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

                    if run_results:
                        final = {
                            k: (sum(r[k] for r in run_results) / len(run_results)
                                if isinstance(run_results[0][k], (int, float)) else run_results[0][k])
                            for k in run_results[0]
                        }
                        results.append(final)
                        yield f"data: {json.dumps({'type': 'condition_complete', 'result': final})}\n\n"

        # Save results
        RESULTS_PATH.parent.mkdir(exist_ok=True)
        existing = []
        if RESULTS_PATH.exists():
            with open(RESULTS_PATH) as f:
                existing = json.load(f)
        with open(RESULTS_PATH, "w") as f:
            json.dump(existing + results, f, indent=2)

        yield f"data: {json.dumps({'type': 'done', 'total_results': len(results)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def _async_request(client, model, prompt, max_tokens, prompt_length, concurrency):
    import time
    payload = {
        "model": model, "prompt": prompt, "stream": True,
        "options": {"num_predict": max_tokens, "temperature": 0},
    }
    first_token_time = None
    t_start = time.perf_counter()
    final_chunk = {}

    async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json=payload) as response:
        async for line in response.aiter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if not chunk.get("done", False) and first_token_time is None:
                first_token_time = time.perf_counter()
            if chunk.get("done", False):
                final_chunk = chunk

    t_end = time.perf_counter()
    tokens = final_chunk.get("eval_count", 0)
    eval_ms = final_chunk.get("eval_duration", 0) / 1_000_000
    total_ms = (t_end - t_start) * 1000
    ttft_ms = (first_token_time - t_start) * 1000 if first_token_time else total_ms
    throughput = tokens / (eval_ms / 1000) if eval_ms > 0 else 0

    return {
        "throughput": throughput, "ttft_ms": ttft_ms,
        "total_ms": total_ms, "tokens": tokens,
        "prompt_tokens": final_chunk.get("prompt_eval_count", 0),
    }


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text())