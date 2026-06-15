#!/usr/bin/env python3
"""
Driver for advanced or long-running benchmarks.

This script runs extended quantization and thermal workloads that are
slower or heavier than the core capacity profiler.

Core responsibilities:
- Orchestrate multi-stage runs (warmup, sustained load, cooldown).
- Collect interval metrics to observe drift/throttling.
- Save detailed traces and per-interval measurements for analysis.

Recommended notes:
- Run on a machine with sufficient cooling and monitoring.
- Provide flags to limit duration or skip long phases for quick tests.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import json
from rich.console import Console
from rich.table import Table
from harness.logger import BenchmarkLogger

console = Console()
RESULTS_PATH = "results/advanced_results.json"


def save_results(key, data):
    os.makedirs("results", exist_ok=True)
    existing = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            existing = json.load(f)
    existing[key] = data
    with open(RESULTS_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    console.print(f"[green]✓ {key} results saved[/green]")


def run_quant():
    console.rule("[bold blue]1. Quantization Comparison")
    from experiments.quant_bench import QuantBenchmark

    bench = QuantBenchmark()
    results = bench.run_sweep(warmup=2, runs=3)

    table = Table(title="Quantization Results", header_style="bold magenta")
    table.add_column("Model + Quant")
    table.add_column("Throughput (tok/s)", justify="right", style="green")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("Latency (ms)", justify="right")

    for r in results:
        table.add_row(
            r.model,
            f"{r.throughput_tok_per_sec:.1f}",
            f"{r.ttft_ms:.0f}",
            f"{r.total_latency_ms:.0f}",
        )
    console.print(table)
    save_results("quantization", [r.to_dict() for r in results])


def run_thermal():
    console.rule("[bold amber]2. Thermal Throttling")
    from experiments.thermal_bench import ThermalBenchmark

    console.print("[yellow]Running 10-minute sustained benchmark — do not close terminal[/yellow]")
    bench = ThermalBenchmark()
    samples = bench.run_sustained(
        model="llama3.1:8b",
        duration_seconds=600,
        sample_interval=30,
    )

    table = Table(title="Thermal Throttling", header_style="bold magenta")
    table.add_column("Time (min)")
    table.add_column("Throughput (tok/s)", justify="right", style="green")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("CPU Temp", justify="right")

    for s in samples:
        table.add_row(
            f"{s['elapsed_minutes']}",
            f"{s['throughput_tok_per_sec']:.1f}",
            f"{s['ttft_ms']:.0f}",
            f"{s['cpu_temp_c']}°C" if s['cpu_temp_c'] > 0 else "N/A",
        )
    console.print(table)
    save_results("thermal", samples)


if __name__ == "__main__":
    console.print("[bold]Experiments — Apple Silicon M4[/bold]\n")

    run_quant()
    run_thermal()

    console.print("\n[bold green]✅ Experiments complete.[/bold green]")
    console.print(f"Results saved to {RESULTS_PATH}")
