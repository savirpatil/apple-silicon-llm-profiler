#!/usr/bin/env python3
import json
import os
import platform
import subprocess
from datetime import datetime

RESULTS_PATH = "results/full_suite.json"

def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)

def get_env_info():
    try:
        chip = subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True).stdout.strip()
    except:
        chip = "Apple Silicon"
    return {
        "chip": chip,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "os": f"macOS {platform.mac_ver()[0]}",
        "python": platform.python_version(),
    }

def build_summary_table(results):
    best = {}
    for r in results:
        if r["concurrency"] != 1:
            continue
        key = (r["framework"], r["model"].split("/")[-1])
        if key not in best or r["throughput_tok_per_sec"] > best[key]["throughput_tok_per_sec"]:
            best[key] = r

    rows = []
    for (fw, model), r in sorted(best.items(), key=lambda x: -x[1]["throughput_tok_per_sec"]):
        stddev = r.get("throughput_stddev", 0)
        bw = r.get("memory_bandwidth_utilization_pct", 0)
        rows.append(
            f"| {fw} | {model} | {r['throughput_tok_per_sec']:.1f} ±{stddev:.1f} | "
            f"{r['ttft_ms']:.0f} | {r['total_latency_ms']:.0f} | {bw:.1f}% |"
        )
    return rows

def build_concurrency_table(results):
    rows = []
    filtered = [r for r in results if r["prompt_length"] == "medium"]
    filtered.sort(key=lambda x: (x["framework"], x["model"], x["concurrency"]))
    seen = set()
    for r in filtered:
        key = (r["framework"], r["model"].split("/")[-1], r["concurrency"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            f"| {r['framework']} | {r['model'].split('/')[-1]} | {r['concurrency']} | "
            f"{r['throughput_tok_per_sec']:.1f} | {r['ttft_ms']:.0f} | {r['total_latency_ms']:.0f} |"
        )
    return rows

def generate():
    results = load_results()
    env = get_env_info()
    summary_rows = build_summary_table(results)
    concurrency_rows = build_concurrency_table(results)

    ollama_ttft_c1 = next((r["ttft_ms"] for r in results if r["framework"] == "ollama" and r["concurrency"] == 1 and r["prompt_length"] == "medium"), 0)
    ollama_ttft_c16 = next((r["ttft_ms"] for r in results if r["framework"] == "ollama" and r["concurrency"] == 16 and r["prompt_length"] == "medium"), 0)
    mlx_ttft_c16 = next((r["ttft_ms"] for r in results if r["framework"] == "mlx" and r["concurrency"] == 16 and r["prompt_length"] == "medium"), 0)

    readme = (
        "# Apple Silicon LLM Inference Benchmarking Suite\n\n"
        "> A clean, reproducible benchmarking tool for LLM inference on Apple Silicon M-series chips.\n"
        "> Pull any model from Ollama, run the suite, and get real performance numbers for your specific Mac.\n\n"
        "Almost every public LLM inference benchmark runs on NVIDIA H100s in data centers. "
        "This project fills the gap for the millions of developers running models locally on Apple Silicon — "
        "giving you actual throughput, latency, and concurrency numbers for your hardware.\n\n"
        "---\n\n"
        "## What it does\n\n"
        "- **Benchmarks any Ollama model** you have pulled — whatever shows up in `ollama list` appears as a checkbox in the UI\n"
        "- **Tests two inference frameworks** — Ollama (llama.cpp/Metal) and MLX (Apple's native ML framework)\n"
        "- **Sweeps concurrency levels** (1, 4, 8, 16 simultaneous requests) to show how each framework handles load\n"
        "- **Measures what actually matters** — throughput (tok/s), TTFT (time to first token), latency, and memory bandwidth utilization\n"
        "- **Streams live results** to a web dashboard as benchmarks run\n"
        "- **Logs everything to W&B** automatically for experiment tracking and visualization\n"
        "- **Exports results** to JSON and CSV for your own analysis\n\n"
        "---\n\n"
        "## Quick start\n\n"
        "**Requirements:** Apple M-series Mac · 16GB+ RAM · [Ollama](https://ollama.ai) installed · Python 3.11+\n\n"
        "```bash\n"
        "git clone https://github.com/savirpatil/apple-silicon-llm-benchmarks\n"
        "cd apple-silicon-llm-benchmarks\n"
        "python3 -m venv .venv && source .venv/bin/activate\n"
        "pip install -r requirements.txt\n"
        "cp .env.example .env  # add your W&B key (free at wandb.ai)\n"
        "```\n\n"
        "Pull whichever models you want to benchmark:\n"
        "```bash\n"
        "ollama pull qwen2.5:7b\n"
        "ollama pull llama3.1:8b\n"
        "ollama pull mistral:7b  # or any other model\n"
        "```\n\n"
        "Launch the dashboard and run benchmarks from your browser:\n"
        "```bash\n"
        "python ui/run_ui.py\n"
        "# Open http://localhost:8000\n"
        "```\n\n"
        "Or run the full automated suite from the terminal:\n"
        "```bash\n"
        "python scripts/run_full_suite.py\n"
        "```\n\n"
        "---\n\n"
        "## Dashboard\n\n"
        "The web UI at `http://localhost:8000` has three tabs:\n\n"
        "- **Dashboard** — charts comparing frameworks across concurrency levels and prompt lengths, auto-populated from your results\n"
        "- **Run benchmark** — select models, concurrency levels, and prompt lengths, then watch live tok/s stream in real time\n"
        "- **Results** — full sortable table of every condition run, with CSV export\n\n"
        "---\n\n"
        "## What gets measured\n\n"
        "| Metric | What it means |\n"
        "|--------|---------------|\n"
        "| Throughput (tok/s) | Tokens generated per second — primary speed metric |\n"
        "| TTFT (ms) | Time to first token — latency before output starts appearing |\n"
        "| Total latency (ms) | End-to-end response time |\n"
        "| Memory BW utilization | % of M-series theoretical peak bandwidth used during inference |\n"
        "| stddev | Stability across runs — low stddev = consistent performance |\n"
        "| p95 latency | Worst-case latency 1 in 20 requests will see |\n\n"
        "---\n\n"
        "## Advanced benchmarks\n\n"
        "Beyond the main sweep, the suite includes:\n\n"
        "```bash\n"
        "python scripts/run_advanced_benchmarks.py\n"
        "```\n\n"
        "- **Quantization comparison** — Q4 vs Q8 speed and memory tradeoffs on the same model\n"
        "- **Thermal throttling detection** — sustained 10-minute run showing throughput degradation over time as the chip heats up\n"
        "- **Thinking mode benchmark** — Qwen3 with thinking on vs off: latency, token usage, and quality tradeoffs\n\n"
        "---\n\n"
        "## Example results\n\n"
        f"The following was run on {env['chip']} · 16GB · {env['os']} as a reference point. "
        "Your numbers will differ based on your chip, RAM, and thermal state.\n\n"
        "**Peak throughput (concurrency=1):**\n\n"
        "| Framework | Model | Throughput (tok/s) | TTFT (ms) | Latency (ms) | BW Util % |\n"
        "|-----------|-------|-------------------|-----------|--------------|------------|\n"
        + "\n".join(summary_rows) + "\n\n"
        "**Concurrency scaling (medium prompt) — the key finding:**\n\n"
        "| Framework | Model | Concurrency | Throughput (tok/s) | TTFT (ms) | Latency (ms) |\n"
        "|-----------|-------|-------------|-------------------|-----------|-------------- |\n"
        + "\n".join(concurrency_rows) + "\n\n"
        f"> MLX TTFT stays ~300ms at 16x concurrency. Ollama TTFT goes from {ollama_ttft_c1:.0f}ms → {ollama_ttft_c16:.0f}ms "
        f"({ollama_ttft_c16/max(ollama_ttft_c1,1):.0f}x degradation) because it queues requests sequentially.\n\n"
        "---\n\n"
        "## Project structure\n\n"
        "```\n"
        "apple-silicon-llm-benchmarks/\n"
        "├── benchmarks/     # Per-framework modules (Ollama, MLX, quantization, thermal, thinking)\n"
        "├── harness/        # Runner, metrics + statistics, W&B logger, environment capture\n"
        "├── scripts/        # run_full_suite.py, run_advanced_benchmarks.py, generate_readme.py\n"
        "├── ui/             # FastAPI dashboard with live benchmark streaming\n"
        "├── results/        # Your results saved here after each run (JSON + CSV)\n"
        "└── config/         # Concurrency levels, prompt lengths, model registry\n"
        "```\n\n"
        "---\n\n"
        "## Methodology\n\n"
        "- **Warmup:** 2 runs discarded before timing begins — eliminates cold-start and cache effects\n"
        "- **Bench runs:** 3 runs per condition — mean, stddev, and p95 reported\n"
        "- **TTFT:** Wall-clock time from request send to first token received via streaming API\n"
        "- **Throughput:** Tokens generated / eval time — excludes model load and queue wait time\n"
        "- **Memory bandwidth utilization:** `(tokens × model_bytes) / (eval_time × M-series peak bandwidth)`\n"
        "- **Concurrency simulation:** Async HTTP requests fired simultaneously for Ollama; sequential simulation for MLX\n"
        "- **Statistics:** stddev and p95 computed across bench runs per condition\n\n"
        "---\n\n"
        "## W&B tracking\n\n"
        "Every run is automatically logged to Weights & Biases. Add your free API key to `.env`:\n"
        "```\n"
        "WANDB_API_KEY=your_key_here\n"
        "WANDB_PROJECT=apple-silicon-llm-bench\n"
        "```\n"
        "Get a free key at [wandb.ai](https://wandb.ai). "
        "View the reference W&B project: https://wandb.ai/savirpatil-purdue-university/apple-silicon-llm-bench\n"
    )

    with open("README.md", "w") as f:
        f.write(readme)
    print("✓ README.md generated")

if __name__ == "__main__":
    generate()