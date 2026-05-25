#!/usr/bin/env python3
"""
Parameter sweep driver (quantization, batching, lengths).

This script automates parameter sweeps across quantization levels,
batch sizes, and other model/run parameters to build performance
matrices for analysis.

Core responsibilities:
- Iterate parameter grid and invoke appropriate benchmark runners.
- Collect structured results for each configuration point.
- Optionally parallelize independent runs and write a combined report.

Recommended notes:
- Keep sweep granularity configurable and results reproducible.
- Limit concurrency to avoid oversubscribing hardware on a single Mac.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import httpx
from rich.console import Console
from rich.table import Table
from harness.runner import SweepConfig, run_full_sweep
from harness.logger import BenchmarkLogger

console = Console()

# ── Sweep configuration ──────────────────────────────────────────
# Start with concurrency [1, 4] to keep runtime reasonable (~15 min)
# Add 8, 16 once baseline is confirmed working
CONFIG = SweepConfig(
    models=["qwen2.5:7b", "llama3.1:8b"],
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


def check_models_available(models):
    """Verify all required models are pulled in Ollama."""
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        available = [m["name"] for m in r.json().get("models", [])]
        missing = [m for m in models if m not in available]
        if missing:
            console.print(f"[red]Missing models: {missing}[/red]")
            console.print("Run: " + " && ".join(f"ollama pull {m}" for m in missing))
            return False
        console.print(f"[green]✓ Models available: {available}[/green]")
        return True
    except Exception as e:
        console.print(f"[red]Ollama not reachable: {e} — run: ollama serve[/red]")
        return False


def display_final_table(results):
    table = Table(
        title="Full Sweep Results — Apple Silicon M4",
        show_header=True,
        header_style="bold magenta"
    )
    table.add_column("Model", style="cyan")
    table.add_column("Prompt")
    table.add_column("Concurrency", justify="center")
    table.add_column("Throughput (tok/s)", justify="right", style="green")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Tokens Out", justify="right")

    for r in results:
        table.add_row(
            r.model,
            r.prompt_length,
            str(r.concurrency),
            f"{r.throughput_tok_per_sec:.1f}",
            f"{r.ttft_ms:.0f}",
            f"{r.total_latency_ms:.0f}",
            str(r.tokens_generated),
        )

    console.print(table)


if __name__ == "__main__":
    console.print("[bold]Apple Silicon LLM Benchmark — Full Sweep[/bold]\n")

    if not check_models_available(CONFIG.models):
        sys.exit(1)

    # Init W&B run
    logger = BenchmarkLogger(run_name="sweep-ollama-m4")
    logger.init({
        "checkpoint": "A3",
        "frameworks": ["ollama"],
        "models": CONFIG.models,
        "concurrency_levels": CONFIG.concurrency_levels,
        "prompt_lengths": list(CONFIG.prompt_lengths.keys()),
        "chip": "M4",
        "max_new_tokens": CONFIG.max_new_tokens,
    })

    try:
        results = run_full_sweep(CONFIG, logger=logger)
        display_final_table(results)
        logger.log_summary_table(results)

        # Save to results/full_suite.json
        import json
        os.makedirs("results", exist_ok=True)
        results_path = "results/full_suite.json"
        existing = []
        if os.path.exists(results_path):
            with open(results_path) as f:
                existing = json.load(f)
        all_results = existing + [r.to_dict() for r in results]
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        console.print(f"[green]✓ {len(results)} results saved to {results_path}[/green]")
        console.print("\n[bold green]✅ Sweep complete.[/bold green]")
    finally:
        logger.finish()