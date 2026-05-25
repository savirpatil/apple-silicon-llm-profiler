#!/usr/bin/env python3
"""
Orchestrator for the full benchmark suite.

This script runs the complete set of benchmarks across configured
models and backends, coordinating long and short tests, aggregating
results, and producing final artifacts for reporting.

Core responsibilities:
- Sequence baseline, advanced, and sweep runs per config.
- Handle job scheduling, optional parallelism, and result consolidation.
- Emit well-structured outputs for downstream analysis and CI artifacts.

Recommended notes:
- Expect longer execution times; provide resumption/checkpointing.
- Use models.yaml (if present) to parameterize runs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import json
import httpx
from rich.console import Console
from rich.table import Table
from harness.runner import SweepConfig, run_full_sweep, run_mlx_sweep
from harness.logger import BenchmarkLogger

console = Console()

# ── Configuration ─────────────────────────────────────────────────
# This gives us:
# Ollama: 2 models × 3 prompts × 4 concurrency = 24 conditions
# MLX:    1 model  × 3 prompts × 4 concurrency = 12 conditions
# Total: 36 conditions — well over the 10+ requirement

OLLAMA_MODELS = ["qwen2.5:7b", "llama3.1:8b"]

# MLX community models are pre-quantized to 4-bit for M-series
# mlx-community hosts official quantized versions of popular models
MLX_MODELS = ["mlx-community/Qwen2.5-7B-Instruct-4bit"]

SWEEP_CONFIG = SweepConfig(
    models=OLLAMA_MODELS,
    prompt_lengths={
        "short":  "Explain what machine learning is in one sentence.",
        "medium": "Explain how transformer attention mechanisms work, covering key-query-value structure.",
        "long":   (
            "You are an expert in distributed systems. Explain in detail: "
            "1) The CAP theorem and its implications, "
            "2) How Cassandra and DynamoDB handle consistency vs availability tradeoffs, "
            "3) What eventual consistency means in practice, "
            "4) How you would design a globally distributed key-value store."
        ),
    },
    concurrency_levels=[1, 4, 8, 16],
    max_new_tokens=128,
    warmup_runs=2,
    bench_runs=3,
)


def check_ollama_models():
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        available = [m["name"] for m in r.json().get("models", [])]
        missing = [m for m in OLLAMA_MODELS if m not in available]
        if missing:
            console.print(f"[red]Missing Ollama models: {missing}[/red]")
            console.print("Run: " + " && ".join(f"ollama pull {m}" for m in missing))
            return False
        return True
    except Exception as e:
        console.print(f"[red]Ollama not reachable: {e}[/red]")
        return False


def display_final_table(ollama_results, mlx_results):
    all_results = ollama_results + mlx_results

    table = Table(
        title="Full Suite Results — Apple Silicon M4",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("Framework", style="cyan")
    table.add_column("Model")
    table.add_column("Prompt")
    table.add_column("Concurrency", justify="center")
    table.add_column("Throughput (tok/s)", justify="right", style="green")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Tokens Out", justify="right")

    for r in all_results:
        table.add_row(
            r.framework,
            r.model.split("/")[-1],  # strip org prefix for display
            r.prompt_length,
            str(r.concurrency),
            f"{r.throughput_tok_per_sec:.1f}",
            f"{r.ttft_ms:.0f}",
            f"{r.total_latency_ms:.0f}",
            str(r.tokens_generated),
        )

    console.print(table)
    return all_results


def save_results_json(all_results):
    """Save raw results for README generation."""
    os.makedirs("results", exist_ok=True)
    data = [r.to_dict() for r in all_results]
    with open("results/full_suite.json", "w") as f:
        json.dump(data, f, indent=2)
    console.print("[green]✓ Results saved to results/full_suite.json[/green]")


if __name__ == "__main__":
    console.print("[bold]Apple Silicon LLM Benchmark — Full Suite[/bold]\n")

    if not check_ollama_models():
        sys.exit(1)

    logger = BenchmarkLogger(run_name="full-suite-m4")
    logger.init({
        "checkpoint": "A3-full",
        "ollama_models": OLLAMA_MODELS,
        "mlx_models": MLX_MODELS,
        "concurrency_levels": SWEEP_CONFIG.concurrency_levels,
        "prompt_lengths": list(SWEEP_CONFIG.prompt_lengths.keys()),
        "chip": "M4",
        "max_new_tokens": SWEEP_CONFIG.max_new_tokens,
    })

    try:
        # Phase 1: Ollama sweep
        console.rule("[bold blue]Phase 1: Ollama (Metal/llama.cpp)")
        ollama_results = run_full_sweep(SWEEP_CONFIG, logger=logger)

        # Phase 2: MLX sweep
        console.rule("[bold green]Phase 2: MLX (Apple Native)")
        mlx_results = run_mlx_sweep(SWEEP_CONFIG, MLX_MODELS, logger=logger)

        # Display and save
        all_results = display_final_table(ollama_results, mlx_results)
        save_results_json(all_results)
        logger.log_summary_table(all_results)

        console.print("\n[bold green]✅ Full suite complete.[/bold green]")

    finally:
        logger.finish()