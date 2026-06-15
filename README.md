# Apple Silicon Local LLM Capacity Profiler

> Measure whether your Mac can serve local LLM workloads, where latency breaks, and whether Ollama or MLX is the better runtime for your use case.

Most public inference benchmarks target NVIDIA data-center GPUs. This project focuses on the local developer reality: Apple Silicon Macs running Ollama and MLX under concurrent load.

## What It Does

- Discovers local Ollama models and benchmarks configured MLX models
- Runs reproducible sweeps across model, prompt length, framework, and concurrency
- Measures throughput, TTFT, total latency, p95 behavior, memory use, and memory-bandwidth utilization
- Converts raw benchmark rows into capacity recommendations: max safe concurrency, recommended runtime, and likely bottleneck
- Streams live benchmark progress through a FastAPI dashboard
- Caches each run separately so historical results stay off the main profiler page

## Inputs

The main input is [config/benchmark_config.yaml](config/benchmark_config.yaml):

```yaml
frameworks: [ollama, mlx]
concurrency_levels: [1, 4, 8, 16]
max_new_tokens: 128
num_warmup_runs: 2
num_benchmark_runs: 3

slo:
  ttft_ms: 1000
  p95_latency_ms: 10000
```

The dashboard also lets you choose frameworks, Ollama models, prompt lengths, concurrency levels, max tokens, and run count.

## Outputs

- `results/runs/<run_id>/results.json`: raw per-condition benchmark results
- `results/runs/<run_id>/summary.json`: max-safe-concurrency recommendations
- `results/runs/<run_id>/environment.json`: chip, memory, OS, Python, Ollama, and MLX metadata
- `results/runs/<run_id>/config.json`: the run inputs used for that benchmark
- FastAPI dashboard at `http://localhost:8000`
- Optional W&B table and metrics when `WANDB_API_KEY` is configured

Example recommendation:

```text
Ollama max safe concurrency under 1s TTFT: 1
MLX max safe concurrency under 1s TTFT: 16
Likely Ollama bottleneck: request_queueing
Recommended runtime: MLX
```

## Quick Start

Requirements: Apple Silicon Mac, Python 3.11+, Ollama, and at least one pulled Ollama model.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
python scripts/verify_setup.py
```

Run the CLI profiler:

```bash
python scripts/run_suite.py
```

Run the dashboard:

```bash
python ui/run_ui.py
# open http://localhost:8000
```

The dashboard has three pages:

- **Profiler:** configure and run Ollama/MLX comparisons, then inspect current-run charts and recommendations
- **Run History:** load cached benchmark runs from `results/runs/`
- **Experiments:** view optional quantization and thermal experiment outputs

## Core Finding

Reference M4 results showed similar single-request throughput for Ollama and MLX, but very different behavior under concurrency. Ollama maintained token throughput while TTFT degraded sharply because requests queued behind one another; MLX stayed near interactive first-token latency in the same benchmark shape.

That makes this project more than a report generator: it acts as a local inference capacity planner for Apple Silicon.

## Project Structure

```text
benchmarks/      Core Ollama and MLX benchmark implementations
harness/         Config loading, runner, metrics, environment capture, summary logic
ui/              FastAPI dashboard with live SSE progress
scripts/         Canonical CLI runner and setup verification
experiments/     Optional quantization and thermal experiments
config/          Benchmark matrix and latency SLOs
results/         Generated local outputs
```

## Optional Experiments

Advanced experiments live outside the core product path:

```bash
python experiments/run_advanced_benchmarks.py
```

They cover quantization and thermal throttling, while the main project story stays focused on local inference capacity.
