# Apple Silicon LLM Inference Benchmarking Suite

> Systematic, reproducible benchmarks of LLM inference frameworks on Apple Silicon M-series chips.
> Almost all public LLM inference benchmarks target NVIDIA H100s — this fills that gap.

**Hardware:** Mac16,1 · 16GB Unified Memory  
**OS:** macOS 15.7.4  
**Date:** 2026-05-23  
**Frameworks tested:** Ollama (llama.cpp/Metal), MLX (Apple native)  
**Models:** Qwen2.5-7B, Llama-3.1-8B, Qwen3-8B  
**Tracked:** [W&B Project](https://wandb.ai/savirpatil-purdue-university/apple-silicon-llm-bench)

---

## Key Findings

- **MLX TTFT is stable under concurrency** — stays ~300ms at 16x concurrent requests vs Ollama's 45629ms (171x worse)
- **Throughput is memory-bandwidth-bound** — both frameworks plateau at ~21-22 tok/s regardless of concurrency, model, or prompt length
- **Ollama TTFT collapses under load** — 112ms at concurrency=1 vs 45629ms at concurrency=16 (406x degradation)
- **16GB unified memory handles 7-8B models comfortably** at 4-bit quantization with headroom for the OS
- **Prompt length has no effect on throughput** — the bottleneck is generation, not prefill

---

## Results: Peak Throughput (concurrency=1)

| Framework | Model | Throughput (tok/s) | TTFT (ms) | Latency (ms) | BW Util % |
|-----------|-------|-------------------|-----------|--------------|------------|
| mlx | Qwen2.5-7B-Instruct-4bit | 22.2 ±0.0 | 281 | 5782 | 0.0% |
| ollama | qwen2.5:7b | 21.8 ±0.0 | 110 | 6028 | 0.0% |
| ollama | llama3.1:8b | 20.4 ±0.0 | 128 | 6426 | 0.0% |

---

## Results: Concurrency Scaling (medium prompt)

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

---

## Reproduce This Benchmark

```bash
git clone https://github.com/savirpatil/apple-silicon-llm-bench
cd apple-silicon-llm-bench
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:7b && ollama pull llama3.1:8b
cp .env.example .env  # add your W&B key
python scripts/run_full_suite.py
```

**Requirements:** Apple M-series Mac · 16GB+ RAM · Ollama · Python 3.11+

---

## Project Structure

```
apple-silicon-llm-bench/
├── benchmarks/          # Per-framework modules (Ollama, MLX, quantization, thermal, thinking)
├── harness/             # Runner, metrics + statistics, W&B logger, environment capture
├── scripts/             # Entrypoints: full suite, advanced benchmarks, README gen
├── ui/                  # FastAPI dashboard with live benchmark streaming
├── results/             # JSON results (auto-generated)
└── config/              # Benchmark configuration
```

---

## Methodology

- **Warmup:** 2 runs discarded before timing begins (eliminates cold-start bias)
- **Bench runs:** 3 runs per condition — mean, stddev, and p95 reported
- **TTFT:** Wall-clock time from request send to first token received (streaming)
- **Throughput:** Tokens generated / eval time (excludes model load and queue time)
- **Memory bandwidth utilization:** (tokens × model_bytes) / (eval_time × M4_peak_bandwidth)
- **Concurrency:** Async HTTP requests fired simultaneously (Ollama), sequential simulation (MLX)
- **Statistics:** stddev and p95 computed across bench runs per condition
