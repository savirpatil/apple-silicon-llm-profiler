# Apple Silicon LLM Inference Benchmarking Suite

> A reproducible benchmarking tool for LLM inference on Apple Silicon M-series chips.
> Pull any model from Ollama, run the suite, and get real performance numbers for your Mac.

Almost every public LLM inference benchmark runs on NVIDIA H100s in data centers. This project fills the gap for the millions of developers running models locally on Apple Silicon, and reports throughput, latency, and concurrency numbers for your hardware.

---

## What it does

- Benchmarks any Ollama model you have pulled — whatever shows up in `ollama list` appears as a checkbox in the UI
- Tests two inference frameworks — Ollama (llama.cpp/Metal) and MLX (Apple's native ML framework)
- Sweeps concurrency levels (1, 4, 8, 16 simultaneous requests) to show how each framework handles load
- Measures what actually matters — throughput (tok/s), TTFT (time to first token), latency, and memory bandwidth utilization
- Streams live results to a web dashboard as benchmarks run
- Logs everything to W&B automatically for experiment tracking and visualization
- Exports results to JSON and CSV for your own analysis

---

## Quick start

**Requirements:** Apple M-series Mac · 16GB+ RAM · [Ollama](https://ollama.ai) installed · Python 3.11+

```bash
git clone https://github.com/savirpatil/apple-silicon-llm-benchmarks
cd apple-silicon-llm-benchmarks
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your W&B key (free at wandb.ai)
```

Pull whichever models you want to benchmark:
```bash
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
ollama pull mistral:7b  # or any other model
```

Launch the dashboard and run benchmarks from your browser:
```bash
python ui/run_ui.py
# Open http://localhost:8000
```

Or run the full automated suite from the terminal:
```bash
python scripts/run_full_suite.py
```

---

## Dashboard

The web UI at `http://localhost:8000` has three tabs:

- **Dashboard** — charts comparing frameworks across concurrency levels and prompt lengths, auto-populated from your results
- **Run benchmark** — select models, concurrency levels, and prompt lengths, then watch live tok/s stream in real time
- **Results** — full sortable table of every condition run, with CSV export

---

## What gets measured

| Metric | What it means |
|--------|---------------|
| Throughput (tok/s) | Tokens generated per second — primary speed metric |
| TTFT (ms) | Time to first token — latency before output starts appearing |
| Total latency (ms) | End-to-end response time |
| Memory BW utilization | % of M-series theoretical peak bandwidth used during inference |
| stddev | Stability across runs — low stddev = consistent performance |
| p95 latency | Worst-case latency 1 in 20 requests will see |

---

## Advanced benchmarks

Beyond the main sweep, the suite includes:

```bash
python scripts/run_advanced_benchmarks.py
```

- **Quantization comparison** — Q4 vs Q8 speed and memory tradeoffs on the same model
- **Thermal throttling detection** — sustained 10-minute run showing throughput degradation over time as the chip heats up
- **Thinking mode benchmark** — Qwen3 with thinking on vs off: latency, token usage, and quality tradeoffs

---

## Example results

The following was run on Mac16,1 · 16GB · macOS 15.7.4 as a reference point. Your numbers will differ based on your chip, RAM, and thermal state.

**Peak throughput (concurrency=1):**

| Framework | Model | Throughput (tok/s) | TTFT (ms) | Latency (ms) | BW Util % |
|-----------|-------|-------------------|-----------|--------------|------------|
| mlx | Qwen2.5-7B-Instruct-4bit | 22.2 ±0.0 | 281 | 5782 | 0.0% |
| ollama | qwen2.5:7b | 21.8 ±0.0 | 110 | 6028 | 0.0% |
| ollama | llama3.1:8b | 20.4 ±0.0 | 128 | 6426 | 0.0% |

**Concurrency scaling (medium prompt) — the key finding:**

| Framework | Model | Concurrency | Throughput (tok/s) | TTFT (ms) | Latency (ms) |
|-----------|-------|-------------|-------------------|-----------|-------------- |
| mlx | Qwen2.5-7B-Instruct-4bit | 1 | 22.2 | 281 | 5782 |
| mlx | Qwen2.5-7B-Instruct-4bit | 4 | 22.4 | 277 | 5733 |
| mlx | Qwen2.5-7B-Instruct-4bit | 8 | 22.0 | 281 | 5843 |
| mlx | Qwen2.5-7B-Instruct-4bit | 16 | 22.6 | 267 | 5659 |
| ollama | llama3.1:8b | 1 | 20.2 | 143 | 6508 |
| ollama | llama3.1:8b | 4 | 20.1 | 9834 | 16231 |
| ollama | llama3.1:8b | 8 | 20.4 | 22510 | 28829 |
| ollama | llama3.1:8b | 16 | 20.4 | 48099 | 54411 |
| ollama | qwen2.5:7b | 1 | 21.1 | 112 | 6229 |
| ollama | qwen2.5:7b | 4 | 20.5 | 9699 | 15990 |
| ollama | qwen2.5:7b | 8 | 21.5 | 21432 | 27442 |
| ollama | qwen2.5:7b | 16 | 21.5 | 45629 | 51614 |

> MLX TTFT stays ~300ms at 16x concurrency. Ollama TTFT goes from 112ms → 45629ms (406x degradation) because it queues requests sequentially.

---

## Project structure

```
apple-silicon-llm-benchmarks/
├── benchmarks/     # Per-framework modules (Ollama, MLX, quantization, thermal, thinking)
├── harness/        # Runner, metrics + statistics, W&B logger, environment capture
├── scripts/        # run_full_suite.py, run_advanced_benchmarks.py, generate_readme.py
├── ui/             # FastAPI dashboard with live benchmark streaming
├── results/        # Your results saved here after each run (JSON + CSV)
└── config/         # Concurrency levels, prompt lengths, model registry
```

---

## Methodology

- **Warmup:** 2 runs discarded before timing begins — eliminates cold-start and cache effects
- **Bench runs:** 3 runs per condition — mean, stddev, and p95 reported
- **TTFT:** Wall-clock time from request send to first token received via streaming API
- **Throughput:** Tokens generated / eval time — excludes model load and queue wait time
- **Memory bandwidth utilization:** `(tokens × model_bytes) / (eval_time × M-series peak bandwidth)`
- **Concurrency simulation:** Async HTTP requests fired simultaneously for Ollama; sequential simulation for MLX
- **Statistics:** stddev and p95 computed across bench runs per condition

---

## W&B tracking

Every run is automatically logged to Weights & Biases. Add your free API key to `.env`:
```
WANDB_API_KEY=your_key_here
WANDB_PROJECT=apple-silicon-llm-bench
```
Get a free key at [wandb.ai](https://wandb.ai). View the reference W&B project: https://wandb.ai/savirpatil-purdue-university/apple-silicon-llm-bench
