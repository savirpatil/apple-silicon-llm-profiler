#!/usr/bin/env python3
"""Run the Apple Silicon local LLM capacity profiler."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from harness.config import load_benchmark_config
from harness.environment import get_environment_info
from harness.logger import BenchmarkLogger
from harness.runner import run_capacity_suite

load_dotenv()

console = Console()
RUNS_PATH = Path("results/runs")


def check_ollama_models(models: list[str]) -> bool:
    if not models:
        return True
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        available = [m["name"] for m in response.json().get("models", [])]
    except Exception as e:
        console.print(f"[red]Ollama not reachable: {e}[/red]")
        return False

    missing = [m for m in models if m not in available]
    if missing:
        console.print(f"[red]Missing Ollama models: {missing}[/red]")
        console.print("Run: " + " && ".join(f"ollama pull {m}" for m in missing))
        return False
    return True


def display_results(results, summary: dict):
    table = Table(
        title="Apple Silicon LLM Capacity Results",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Framework")
    table.add_column("Model")
    table.add_column("Prompt")
    table.add_column("Conc.", justify="right")
    table.add_column("Tok/s", justify="right")
    table.add_column("TTFT", justify="right")
    table.add_column("p95 TTFT", justify="right")
    table.add_column("BW Util", justify="right")

    for result in results:
        table.add_row(
            result.framework,
            result.model.split("/")[-1],
            result.prompt_length,
            str(result.concurrency),
            f"{result.throughput_tok_per_sec:.1f}",
            f"{result.ttft_ms:.0f}ms",
            f"{result.ttft_p95:.0f}ms",
            f"{result.memory_bandwidth_utilization_pct:.1f}%",
        )
    console.print(table)

    rec = summary.get("recommendations", [])
    if rec:
        rec_table = Table(
            title="Capacity Planner Recommendations",
            show_header=True,
            header_style="bold green",
        )
        rec_table.add_column("Runtime")
        rec_table.add_column("Model")
        rec_table.add_column("Max Safe Conc.", justify="right")
        rec_table.add_column("Bottleneck")
        rec_table.add_column("Recommended")
        for row in rec:
            rec_table.add_row(
                row["framework"],
                row["model"].split("/")[-1],
                str(row["max_safe_concurrency"]),
                row["bottleneck"],
                "yes" if row["recommended"] else "",
            )
        console.print(rec_table)


def save_outputs(results, summary: dict, environment: dict, config) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_PATH / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(
        [r.to_dict() for r in results], indent=2
    ))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        "frameworks": config.frameworks,
        "ollama_models": config.ollama_models,
        "mlx_models": config.mlx_models,
        "concurrency_levels": config.concurrency_levels,
        "prompt_lengths": list(config.prompt_lengths.keys()),
        "max_new_tokens": config.max_new_tokens,
        "warmup_runs": config.warmup_runs,
        "benchmark_runs": config.benchmark_runs,
        "slo": config.slo.__dict__,
    }, indent=2))
    console.print(f"[green]Saved run to {run_dir}[/green]")
    return run_dir


if __name__ == "__main__":
    console.print("[bold]Apple Silicon Local LLM Capacity Profiler[/bold]\n")
    config = load_benchmark_config()
    environment = get_environment_info()

    if "ollama" in config.frameworks and not check_ollama_models(config.ollama_models):
        sys.exit(1)

    logger = BenchmarkLogger(run_name="capacity-profiler")
    if os.getenv("WANDB_API_KEY"):
        logger.init({
            "frameworks": config.frameworks,
            "ollama_models": config.ollama_models,
            "mlx_models": config.mlx_models,
            "concurrency_levels": config.concurrency_levels,
            "prompt_lengths": list(config.prompt_lengths.keys()),
            "max_new_tokens": config.max_new_tokens,
            "slo": config.slo.__dict__,
            "environment": environment,
        })

    try:
        results, summary = run_capacity_suite(config, logger=logger)
        display_results(results, summary)
        save_outputs(results, summary, environment, config)
        if os.getenv("WANDB_API_KEY"):
            logger.log_summary_table(results)
        console.print("\n[bold green]Capacity profiling complete.[/bold green]")
    finally:
        logger.finish()
