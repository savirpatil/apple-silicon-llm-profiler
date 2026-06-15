"""
Quantization Benchmark: speed vs memory tradeoffs.

This module measures performance and memory use across different
quantization levels (e.g., Q4, Q8, fp16). The goal is to produce
simple, comparable metrics (throughput, latency, peak memory) so
you can choose a quantization strategy that fits your hardware and
accuracy requirements.

Recommended notes:
- Q4 (4-bit): smallest memory footprint and highest throughput.
- Q8 (8-bit): balance between size and fidelity.
- fp16: best numeric fidelity at higher memory cost and lower speed.
- Keep measurement workloads identical across quant levels.
"""

import time
import uuid
import json
import httpx
from harness.metrics import BenchmarkResult, MemoryTracker

OLLAMA_URL = "http://localhost:11434"

QUANT_MODELS = {
    "Q4_K_M": "llama3.1:8b",           # Default 4-bit
    "Q8_0":   "llama3.1:8b-q8_0",      # 8-bit, higher quality
}

TEST_PROMPT = "Explain how neural networks learn through backpropagation."


class QuantBenchmark:
    name = "quant_comparison"

    def __init__(self):
        self.memory_tracker = MemoryTracker()

    def run_single(self, model: str, quant_label: str, max_new_tokens: int = 128) -> BenchmarkResult:
        payload = {
            "model": model,
            "prompt": TEST_PROMPT,
            "stream": True,
            "options": {"num_predict": max_new_tokens, "temperature": 0},
        }

        first_token_time = None
        t_start = time.perf_counter()
        final_chunk = {}
        self.memory_tracker.reset()

        with httpx.stream("POST", f"{OLLAMA_URL}/api/generate", json=payload, timeout=300) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if not chunk.get("done", False) and first_token_time is None:
                    first_token_time = time.perf_counter()
                if chunk.get("done", False):
                    final_chunk = chunk
                self.memory_tracker.sample()

        t_end = time.perf_counter()
        tokens = final_chunk.get("eval_count", 0)
        eval_ms = final_chunk.get("eval_duration", 0) / 1_000_000
        total_ms = (t_end - t_start) * 1000
        ttft_ms = (first_token_time - t_start) * 1000 if first_token_time else total_ms
        throughput = tokens / (eval_ms / 1000) if eval_ms > 0 else 0

        return BenchmarkResult(
            framework="ollama",
            model=f"{model} ({quant_label})",
            concurrency=1,
            prompt_length="medium",
            batch_size=1,
            ttft_ms=ttft_ms,
            total_latency_ms=total_ms,
            tokens_generated=tokens,
            prompt_tokens=final_chunk.get("prompt_eval_count", 0),
            throughput_tok_per_sec=throughput,
            peak_memory_mb=self.memory_tracker.peak_mb,
            run_id=str(uuid.uuid4())[:8],
        )

    def run_sweep(self, warmup=2, runs=3):
        results = []
        for quant_label, model in QUANT_MODELS.items():
            print(f"\n[Quant] Testing {quant_label} ({model})")
            
            try:
                r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
                available = [m["name"] for m in r.json().get("models", [])]
                if model not in available:
                    print(f"  ⚠ {model} not pulled, skipping. Run: ollama pull {model}")
                    continue
            except Exception as e:
                print(f"  ⚠ Ollama not reachable: {e}")
                continue

            print(f"  Warming up...")
            for _ in range(warmup):
                self.run_single(model, quant_label)

            run_results = []
            for i in range(runs):
                r = self.run_single(model, quant_label)
                run_results.append(r)
                print(f"  Run {i+1}/{runs}: {r.throughput_tok_per_sec:.1f} tok/s | TTFT {r.ttft_ms:.0f}ms")

            avg = BenchmarkResult(
                framework="ollama",
                model=f"{model} ({quant_label})",
                concurrency=1,
                prompt_length="medium",
                batch_size=1,
                ttft_ms=sum(r.ttft_ms for r in run_results) / len(run_results),
                total_latency_ms=sum(r.total_latency_ms for r in run_results) / len(run_results),
                tokens_generated=int(sum(r.tokens_generated for r in run_results) / len(run_results)),
                prompt_tokens=run_results[0].prompt_tokens,
                throughput_tok_per_sec=sum(r.throughput_tok_per_sec for r in run_results) / len(run_results),
                peak_memory_mb=max(r.peak_memory_mb for r in run_results),
                run_id=str(uuid.uuid4())[:8],
            )
            results.append(avg)
            print(f"  ✓ AVG: {avg.throughput_tok_per_sec:.1f} tok/s | TTFT {avg.ttft_ms:.0f}ms")

        return results