#!/usr/bin/env python3
"""
Advanced benchmarks:
1. MLX (Apple native)
2. Quantization comparison
3. Thermal throttling
4. Thinking mode (Qwen3)
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


def run_mlx():
    console.rule("[bold green]1. MLX Benchmark")
    from harness.runner import SweepConfig, run_mlx_sweep

    config = SweepConfig(
        models=[],
        prompt_lengths={
            "short": "Explain what machine learning is in one sentence.",
            "medium": "Explain how transformer attention mechanisms work.",
        },
        concurrency_levels=[1, 4],
        max_new_tokens=128,
        warmup_runs=2,
        bench_runs=3,
    )
    results = run_mlx_sweep(config, ["mlx-community/Qwen2.5-7B-Instruct-4bit"])

    table = Table(title="MLX Results", header_style="bold magenta")
    table.add_column("Model")
    table.add_column("Prompt")
    table.add_column("Concurrency")
    table.add_column("Throughput (tok/s)", justify="right", style="green")
    table.add_column("TTFT (ms)", justify="right")

    for r in results:
        table.add_row(
            r.model.split("/")[-1],
            r.prompt_length,
            str(r.concurrency),
            f"{r.throughput_tok_per_sec:.1f}",
            f"{r.ttft_ms:.0f}",
        )
    console.print(table)
    save_results("mlx", [r.to_dict() for r in results])


def run_quant():
    console.rule("[bold blue]2. Quantization Comparison")
    from benchmarks.quant_bench import QuantBenchmark

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
    console.rule("[bold amber]3. Thermal Throttling")
    from benchmarks.thermal_bench import ThermalBenchmark

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


def run_thinking():
    console.rule("[bold purple]4. Thinking Mode (Qwen3)")
    from benchmarks.thinking_bench import ThinkingBenchmark

    bench = ThinkingBenchmark()
    results = bench.run_sweep(model="qwen3:8b", warmup=1, runs=3)

    table = Table(title="Thinking Mode Results", header_style="bold magenta")
    table.add_column("Prompt type")
    table.add_column("Thinking")
    table.add_column("Throughput (tok/s)", justify="right", style="green")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("Thinking tokens", justify="right")
    table.add_column("Answer tokens", justify="right")

    for r in results:
        table.add_row(
            r["prompt_type"],
            "ON" if r["thinking"] else "OFF",
            f"{r['throughput_tok_per_sec']:.1f}",
            f"{r['ttft_ms']:.0f}",
            str(int(r["thinking_tokens"])),
            str(int(r["answer_tokens"])),
        )
    console.print(table)
    save_results("thinking", results)


if __name__ == "__main__":
    console.print("[bold]Advanced Benchmarks — Apple Silicon M4[/bold]\n")

    # Run all 4 — comment out any you want to skip
    run_mlx()
    run_quant()
    run_thermal()
    run_thinking()

    console.print("\n[bold green]✅ All advanced benchmarks complete.[/bold green]")
    console.print(f"Results saved to {RESULTS_PATH}")