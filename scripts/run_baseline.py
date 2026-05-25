#!/usr/bin/env python3
"""
Quick baseline runner for fast comparisons.

This script runs short, repeatable baseline benchmarks across a small
set of models and backends to produce quick comparative numbers
(useful for development feedback and CI smoke tests).

Core responsibilities:
- Run compact workloads (short prompts, low max_new_tokens) for each backend.
- Produce concise metrics (TTFT, latency, tokens/sec) for fast comparison.
- Exit quickly on failures with informative messages.

Recommended notes:
- Designed for iteration: keep runtime short and deterministic.
- Useful as a pre-commit or CI smoke check.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import time
import uuid
import httpx
from rich.console import Console
from rich.table import Table
from harness.metrics import BenchmarkResult, MemoryTracker
from harness.logger import BenchmarkLogger

console = Console()

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen3:7b"
MODEL_NAME = "qwen3-7b"
MAX_NEW_TOKENS = 64
WARMUP_RUNS = 1
BENCH_RUNS = 3

PROMPTS = {
    "short": "Explain what machine learning is in one sentence.",
    "medium": "Explain how transformer attention mechanisms work, covering key-query-value structure.",
}


def check_ollama():
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        console.print(f"  Ollama running. Models available: {models}")
        return True
    except Exception as e:
        console.print(f"[red]Ollama not reachable: {e}[/red]")
        console.print("  Start it with: ollama serve")
        return False


def run_single(prompt: str, max_new_tokens: int, prompt_length: str) -> BenchmarkResult:
    memory_tracker = MemoryTracker()
    memory_tracker.reset()

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": 0,
        }
    }

    tokens_generated = 0
    first_token_time = None
    t_start = time.perf_counter()
    memory_tracker.sample()

    with httpx.stream("POST", f"{OLLAMA_URL}/api/generate", json=payload, timeout=300) as response:
        for line in response.iter_lines():
            if not line:
                continue
            import json
            chunk = json.loads(line)
            if not chunk.get("done", False):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                tokens_generated += 1
            memory_tracker.sample()

    t_end = time.perf_counter()

    total_ms = (t_end - t_start) * 1000
    ttft_ms = (first_token_time - t_start) * 1000 if first_token_time else total_ms

    return BenchmarkResult(
        framework="ollama",
        model=MODEL_NAME,
        concurrency=1,
        prompt_length=prompt_length,
        batch_size=1,
        ttft_ms=ttft_ms,
        total_latency_ms=total_ms,
        tokens_generated=tokens_generated,
        throughput_tok_per_sec=tokens_generated / (total_ms / 1000),
        peak_memory_mb=memory_tracker.peak_mb,
        run_id=str(uuid.uuid4())[:8],
    )


def run_ollama_baseline():
    console.rule("[bold blue]Ollama (Metal) Baseline")

    if not check_ollama():
        sys.exit(1)

    results = []

    for length_name, prompt in PROMPTS.items():
        console.print(f"\n  Warming up ({length_name})...")
        for _ in range(WARMUP_RUNS):
            run_single(prompt, MAX_NEW_TOKENS, length_name)

        run_results = []
        for i in range(BENCH_RUNS):
            console.print(f"  Run {i+1}/{BENCH_RUNS} [{length_name}]...")
            r = run_single(prompt, MAX_NEW_TOKENS, length_name)
            run_results.append(r)
            console.print(f"    → {r.throughput_tok_per_sec:.1f} tok/s, TTFT {r.ttft_ms:.0f}ms")

        avg = BenchmarkResult(
            framework="ollama",
            model=MODEL_NAME,
            concurrency=1,
            prompt_length=length_name,
            batch_size=1,
            ttft_ms=sum(r.ttft_ms for r in run_results) / len(run_results),
            total_latency_ms=sum(r.total_latency_ms for r in run_results) / len(run_results),
            tokens_generated=int(sum(r.tokens_generated for r in run_results) / len(run_results)),
            throughput_tok_per_sec=sum(r.throughput_tok_per_sec for r in run_results) / len(run_results),
            peak_memory_mb=max(r.peak_memory_mb for r in run_results),
        )
        results.append(avg)
        console.print(f"  [green]✓[/green] {length_name} avg: {avg.throughput_tok_per_sec:.1f} tok/s, {avg.total_latency_ms:.0f}ms latency, TTFT {avg.ttft_ms:.0f}ms")

    return results


def display_results(results):
    table = Table(title="A2 Baseline Results — Ollama/Metal on M4", show_header=True, header_style="bold magenta")
    table.add_column("Framework")
    table.add_column("Model")
    table.add_column("Prompt")
    table.add_column("Throughput (tok/s)", justify="right")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Tokens", justify="right")

    for r in results:
        table.add_row(
            r.framework,
            r.model,
            r.prompt_length,
            f"{r.throughput_tok_per_sec:.1f}",
            f"{r.ttft_ms:.0f}",
            f"{r.total_latency_ms:.0f}",
            str(r.tokens_generated),
        )

    console.print(table)


def log_to_wandb(results):
    logger = BenchmarkLogger(run_name="a2-baseline-ollama-metal")
    logger.init({
        "checkpoint": "A2",
        "model": MODEL_NAME,
        "framework": "ollama",
        "mode": "metal",
        "chip": "M4",
    })
    for r in results:
        logger.log_result(r)
    logger.log_results_table(results)
    logger.finish()
    console.print("[green]✓ Results logged to W&B[/green]")


if __name__ == "__main__":
    console.print("[bold]Apple Silicon LLM Benchmark — Checkpoint A2[/bold]\n")
    results = run_ollama_baseline()
    display_results(results)

    if os.getenv("WANDB_API_KEY"):
        log_to_wandb(results)
    else:
        console.print("[yellow]⚠ WANDB_API_KEY not set — skipping W&B logging[/yellow]")

    console.print("\n[bold green]✅ Checkpoint A2 complete.[/bold green]")