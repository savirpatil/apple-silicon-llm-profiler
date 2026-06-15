#!/usr/bin/env python3
"""
FastAPI backend for the capacity profiler dashboard.
Serves the UI and exposes endpoints for:
- Launching benchmark runs
- Streaming live progress via SSE (Server-Sent Events)
- Listing cached runs from results/runs
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import httpx
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from harness.config import load_benchmark_config
from harness.environment import get_environment_info
from harness.metrics import compute_statistics
from harness.runner import _average_batch, run_concurrent_batch
from harness.summary import CapacitySLO, summarize_capacity

load_dotenv()

app = FastAPI(title="Apple Silicon LLM Benchmark")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

RESULTS_ROOT = Path(__file__).parent.parent / "results"
RUNS_PATH = RESULTS_ROOT / "runs"
OLLAMA_URL = "http://localhost:11434"
BENCHMARK_CONFIG = load_benchmark_config()


class RunConfig(BaseModel):
    frameworks: List[str] = ["ollama"]
    models: List[str]
    prompt_lengths: List[str]
    concurrency_levels: List[int]
    max_new_tokens: int = 128
    bench_runs: int = 3


PROMPT_MAP = {
    name: prompt.text for name, prompt in BENCHMARK_CONFIG.prompt_lengths.items()
}


@app.get("/api/results")
def get_results():
    latest = _latest_run_id()
    if not latest:
        return JSONResponse([])
    return JSONResponse(_load_run(latest).get("results", []))


@app.get("/api/summary")
def get_summary():
    latest = _latest_run_id()
    if not latest:
        return JSONResponse({})
    return JSONResponse(_load_run(latest).get("summary", {}))


@app.get("/api/runs")
def list_runs():
    return JSONResponse(_list_runs())


@app.get("/api/latest-run")
def get_latest_run():
    latest = _latest_run_id()
    if not latest:
        return JSONResponse({})
    return JSONResponse(_load_run(latest))


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run_dir = RUNS_PATH / run_id
    if not run_dir.exists():
        return JSONResponse({"error": "run not found"}, status_code=404)
    return JSONResponse(_load_run(run_id))
    

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
        return JSONResponse({
            "models": models,
            "ollama_models": models,
            "mlx_models": BENCHMARK_CONFIG.mlx_models,
            "status": "ok",
        })
    except Exception as e:
        return JSONResponse({
            "models": [],
            "ollama_models": [],
            "mlx_models": BENCHMARK_CONFIG.mlx_models,
            "status": "error",
            "error": str(e),
        })


@app.post("/api/run")
async def run_benchmark(config: RunConfig):
    """
    Stream benchmark progress via Server-Sent Events.
    SSE = the server keeps the HTTP connection open and pushes
    events as they happen. The browser receives them in real time.
    Format: 'data: {json}\n\n' — the double newline is required by SSE spec.
    """
    async def event_stream():
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RUNS_PATH / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        results = []
        frameworks = config.frameworks or ["ollama"]
        ollama_total = (
            len(config.models) * len(config.prompt_lengths)
            * len(config.concurrency_levels)
            if "ollama" in frameworks else 0
        )
        mlx_total = (
            len(BENCHMARK_CONFIG.mlx_models) * len(config.prompt_lengths)
            * len(config.concurrency_levels)
            if "mlx" in frameworks else 0
        )
        total = ollama_total + mlx_total
        condition = 0
        yield f"data: {json.dumps({'type': 'run_started', 'run_id': run_id})}\n\n"

        if "ollama" in frameworks:
            for model in config.models:
                for prompt_length in config.prompt_lengths:
                    prompt = PROMPT_MAP.get(prompt_length, PROMPT_MAP["short"])
                    for concurrency in config.concurrency_levels:
                        condition += 1

                        yield f"data: {json.dumps({'type': 'progress', 'condition': condition, 'total': total, 'framework': 'ollama', 'model': model, 'prompt_length': prompt_length, 'concurrency': concurrency, 'status': 'running'})}\n\n"

                        # Warmup runs — discarded, prime the model cache
                        for _ in range(2):
                            try:
                                await run_concurrent_batch(
                                    model=model,
                                    prompt=prompt,
                                    prompt_length=prompt_length,
                                    concurrency=1,
                                    max_new_tokens=config.max_new_tokens,
                                )
                            except Exception:
                                pass

                        run_results = []
                        for run_num in range(config.bench_runs):
                            try:
                                batch = await run_concurrent_batch(
                                    model=model,
                                    prompt=prompt,
                                    prompt_length=prompt_length,
                                    concurrency=concurrency,
                                    max_new_tokens=config.max_new_tokens,
                                )

                                if batch:
                                    avg = _average_batch(batch, peak_memory_mb=0.0)
                                    run_results.append(avg)

                                    yield f"data: {json.dumps({'type': 'run_complete', 'run': run_num + 1, 'total_runs': config.bench_runs, 'throughput': round(avg.throughput_tok_per_sec, 1), 'ttft_ms': round(avg.ttft_ms, 0)})}\n\n"

                            except Exception as e:
                                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

                        if run_results:
                            final = compute_statistics(run_results)
                            results.append(final)
                            yield f"data: {json.dumps({'type': 'condition_complete', 'result': final.to_dict()})}\n\n"

        if "mlx" in frameworks:
            from benchmarks.mlx_bench import MLXBenchmark

            for model in BENCHMARK_CONFIG.mlx_models:
                bench = MLXBenchmark()
                try:
                    bench.setup(model)
                    for prompt_length in config.prompt_lengths:
                        prompt = PROMPT_MAP.get(prompt_length, PROMPT_MAP["short"])
                        for concurrency in config.concurrency_levels:
                            condition += 1
                            yield f"data: {json.dumps({'type': 'progress', 'condition': condition, 'total': total, 'framework': 'mlx', 'model': model, 'prompt_length': prompt_length, 'concurrency': concurrency, 'status': 'running'})}\n\n"

                            # Warmup runs — discarded
                            for _ in range(2):
                                bench.run_single(
                                    prompt, config.max_new_tokens, prompt_length, 1
                                )

                            run_results = []
                            for run_num in range(config.bench_runs):
                                batch = [
                                    bench.run_single(
                                        prompt,
                                        config.max_new_tokens,
                                        prompt_length,
                                        concurrency,
                                    )
                                    for _ in range(concurrency)
                                ]
                                avg = compute_statistics(batch)
                                run_results.append(avg)
                                yield f"data: {json.dumps({'type': 'run_complete', 'run': run_num + 1, 'total_runs': config.bench_runs, 'throughput': round(avg.throughput_tok_per_sec, 1), 'ttft_ms': round(avg.ttft_ms, 0)})}\n\n"

                            final = compute_statistics(run_results)
                            results.append(final)
                            yield f"data: {json.dumps({'type': 'condition_complete', 'result': final.to_dict()})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                finally:
                    bench.teardown()

        summary = summarize_capacity(
            results,
            CapacitySLO(
                ttft_ms=BENCHMARK_CONFIG.slo.ttft_ms,
                p95_latency_ms=BENCHMARK_CONFIG.slo.p95_latency_ms,
            ),
        )
        environment = get_environment_info()
        config_data = config.dict()
        _write_json(run_dir / "results.json", [r.to_dict() for r in results])
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "environment.json", environment)
        _write_json(run_dir / "config.json", config_data)

        yield f"data: {json.dumps({'type': 'done', 'run_id': run_id, 'total_results': len(results), 'summary': summary})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text())


def _write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2))


def _list_runs() -> list[dict]:
    if not RUNS_PATH.exists():
        return []
    runs = []
    for run_dir in sorted(RUNS_PATH.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        run = _load_run(run_dir.name)
        summary = run.get("summary", {})
        environment = run.get("environment", {})
        config = run.get("config", {})
        models = config.get("models") or (
            config.get("ollama_models", []) + config.get("mlx_models", [])
        )
        runs.append({
            "run_id": run_dir.name,
            "created_at": _format_run_id(run_dir.name),
            "chip": environment.get("chip", "unknown"),
            "models": models,
            "prompt_lengths": config.get("prompt_lengths", []),
            "concurrency_levels": config.get("concurrency_levels", []),
            "recommended_runtime": summary.get("recommended_runtime"),
            "recommended_model": summary.get("recommended_model"),
            "result_count": len(run.get("results", [])),
        })
    return runs


def _latest_run_id() -> str | None:
    runs = _list_runs()
    return runs[0]["run_id"] if runs else None


def _load_run(run_id: str) -> dict:
    run_dir = RUNS_PATH / run_id
    return {
        "run_id": run_id,
        "created_at": _format_run_id(run_id),
        "results": _read_json(run_dir / "results.json", []),
        "summary": _read_json(run_dir / "summary.json", {}),
        "environment": _read_json(run_dir / "environment.json", {}),
        "config": _read_json(run_dir / "config.json", {}),
    }


def _read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def _format_run_id(run_id: str) -> str:
    try:
        return datetime.strptime(run_id, "%Y%m%d_%H%M%S").isoformat()
    except ValueError:
        return run_id
