"""
Benchmark runner orchestration utilities.

This module coordinates instantiation and execution of benchmark
implementations. It maps model/config entries to concrete benchmark
classes, runs the setup/run_single/teardown lifecycle, and collects
and aggregates results for reporting.

Core responsibilities:
- Load model matrix (optionally from config/models.yaml) and map to backends.
- Provide high-level run loop that handles retries, timeouts, and warmups.
- Aggregate results and emit them via harness.logger and metrics.

Recommended notes:
- Keep backend selection pluggable (factory map).
- Do not embed backend-specific logic here; benchmarks themselves own that.
"""

import asyncio
import time
import uuid
import json
import httpx
from typing import List, Dict
from dataclasses import dataclass
from harness.metrics import BenchmarkResult, MemoryTracker, compute_statistics

OLLAMA_URL = "http://localhost:11434"


@dataclass
class SweepConfig:
    models: List[str]
    prompt_lengths: Dict[str, str]
    concurrency_levels: List[int]
    max_new_tokens: int = 128
    warmup_runs: int = 2
    bench_runs: int = 5


async def _single_async_request(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    max_new_tokens: int,
    prompt_length: str,
    concurrency: int,
) -> BenchmarkResult:
    """
    Single async request to Ollama.
    Using async here means multiple of these can run concurrently
    — the event loop handles them without blocking each other.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": max_new_tokens, "temperature": 0},
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

    tokens_generated = final_chunk.get("eval_count", 0)
    prompt_tokens = final_chunk.get("prompt_eval_count", 0)
    eval_duration_ms = final_chunk.get("eval_duration", 0) / 1_000_000
    total_ms = (t_end - t_start) * 1000
    ttft_ms = (first_token_time - t_start) * 1000 if first_token_time else total_ms
    throughput = tokens_generated / (eval_duration_ms / 1000) if eval_duration_ms > 0 else 0

    return BenchmarkResult(
        framework="ollama",
        model=model,
        concurrency=concurrency,
        prompt_length=prompt_length,
        batch_size=1,
        ttft_ms=ttft_ms,
        total_latency_ms=total_ms,
        tokens_generated=tokens_generated,
        prompt_tokens=prompt_tokens,
        throughput_tok_per_sec=throughput,
        peak_memory_mb=0.0,
        run_id=str(uuid.uuid4())[:8],
    )


async def run_concurrent_batch(
    model: str,
    prompt: str,
    prompt_length: str,
    concurrency: int,
    max_new_tokens: int,
) -> List[BenchmarkResult]:
    """
    Fire `concurrency` requests at the same time and collect all results.

    asyncio.gather() is the key that launches all coroutines simultaneously
    and waits for all of them to finish. The wall-clock time reflects
    real concurrent load on the server.
    """
    async with httpx.AsyncClient(timeout=300) as client:
        tasks = [
            _single_async_request(
                client, model, prompt, max_new_tokens, prompt_length, concurrency
            )
            for _ in range(concurrency)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    return [r for r in results if isinstance(r, BenchmarkResult)]


def run_full_sweep(config: SweepConfig, logger=None) -> List[BenchmarkResult]:
    """
    The main sweep loop. Iterates over every combination of:
    model × prompt_length × concurrency

    For each combo: warmup runs (discarded) then bench runs (recorded).
    Results use compute_statistics() for mean, stddev, and p95.
    """
    all_results = []
    memory_tracker = MemoryTracker()

    total_conditions = (
        len(config.models)
        * len(config.prompt_lengths)
        * len(config.concurrency_levels)
    )
    condition_num = 0

    for model in config.models:
        for prompt_length, prompt in config.prompt_lengths.items():
            for concurrency in config.concurrency_levels:
                condition_num += 1
                print(f"\n[{condition_num}/{total_conditions}] "
                      f"model={model} | prompt={prompt_length} | concurrency={concurrency}")

                # Warmup
                print(f"  Warming up ({config.warmup_runs} runs)...")
                for _ in range(config.warmup_runs):
                    asyncio.run(run_concurrent_batch(
                        model, prompt, prompt_length,
                        concurrency=1,
                        max_new_tokens=config.max_new_tokens,
                    ))

                # Benchmark runs
                run_results = []
                for run_num in range(config.bench_runs):
                    memory_tracker.reset()
                    batch = asyncio.run(run_concurrent_batch(
                        model, prompt, prompt_length,
                        concurrency, config.max_new_tokens,
                    ))

                    if not batch:
                        print(f"  ⚠ Run {run_num+1} failed, skipping")
                        continue

                    avg_result = _average_batch(batch, memory_tracker.peak_mb)
                    run_results.append(avg_result)

                    print(f"  Run {run_num+1}/{config.bench_runs}: "
                          f"{avg_result.throughput_tok_per_sec:.1f} tok/s | "
                          f"TTFT {avg_result.ttft_ms:.0f}ms | "
                          f"latency {avg_result.total_latency_ms:.0f}ms")

                    if logger:
                        logger.log_result(avg_result)

                if not run_results:
                    continue

                final = compute_statistics(run_results)
                all_results.append(final)
                print(f"  ✓ AVG: {final.throughput_tok_per_sec:.1f} tok/s ±{final.throughput_stddev:.1f} | "
                      f"TTFT {final.ttft_ms:.0f}ms | BW util {final.memory_bandwidth_utilization_pct:.1f}%")

    return all_results


def run_mlx_sweep(config: SweepConfig, mlx_models: list, logger=None) -> List[BenchmarkResult]:
    """
    MLX sweep: same conditions as Ollama sweep but runs in-process.

    Key difference from Ollama: MLX is a library call, not an HTTP server.
    This means lower overhead per request but no real concurrency.
    We simulate concurrency by running N sequential requests and
    reporting the per-request average.
    """
    from benchmarks.mlx_bench import MLXBenchmark

    all_results = []
    total_conditions = (
        len(mlx_models)
        * len(config.prompt_lengths)
        * len(config.concurrency_levels)
    )
    condition_num = 0

    for model_name in mlx_models:
        bench = MLXBenchmark()
        bench.setup(model_name)

        for prompt_length, prompt in config.prompt_lengths.items():
            for concurrency in config.concurrency_levels:
                condition_num += 1
                print(f"\n[{condition_num}/{total_conditions}] "
                      f"model={model_name} | prompt={prompt_length} | concurrency={concurrency}")

                # Warmup
                print(f"  Warming up ({config.warmup_runs} runs)...")
                for _ in range(config.warmup_runs):
                    bench.run_single(prompt, config.max_new_tokens, prompt_length, 1)

                # Benchmark runs
                run_results = []
                for run_num in range(config.bench_runs):
                    batch_results = []
                    for _ in range(concurrency):
                        r = bench.run_single(
                            prompt, config.max_new_tokens, prompt_length, concurrency
                        )
                        batch_results.append(r)

                    avg = compute_statistics(batch_results)
                    run_results.append(avg)

                    print(f"  Run {run_num+1}/{config.bench_runs}: "
                          f"{avg.throughput_tok_per_sec:.1f} tok/s | "
                          f"TTFT {avg.ttft_ms:.0f}ms")

                    if logger:
                        logger.log_result(avg)

                # Use compute_statistics across bench runs
                final = compute_statistics(run_results)
                all_results.append(final)
                print(f"  ✓ AVG: {final.throughput_tok_per_sec:.1f} tok/s ±{final.throughput_stddev:.1f} | "
                      f"TTFT {final.ttft_ms:.0f}ms | BW util {final.memory_bandwidth_utilization_pct:.1f}%")

        bench.teardown()

    return all_results


def _average_batch(batch: List[BenchmarkResult], peak_memory_mb: float) -> BenchmarkResult:
    """Average metrics across concurrent requests in one batch."""
    n = len(batch)
    r = batch[0]
    return BenchmarkResult(
        framework=r.framework,
        model=r.model,
        concurrency=r.concurrency,
        prompt_length=r.prompt_length,
        batch_size=n,
        ttft_ms=sum(x.ttft_ms for x in batch) / n,
        total_latency_ms=sum(x.total_latency_ms for x in batch) / n,
        tokens_generated=sum(x.tokens_generated for x in batch),
        prompt_tokens=r.prompt_tokens,
        throughput_tok_per_sec=sum(x.throughput_tok_per_sec for x in batch) / n,
        peak_memory_mb=peak_memory_mb,
        run_id=str(uuid.uuid4())[:8],
    )