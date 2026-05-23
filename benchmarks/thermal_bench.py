"""
Thermal throttling benchmark.

Apple Silicon throttles under sustained load — the M4 in a MacBook
runs faster for the first 2 minutes than after 10 minutes as the
chip heats up and the firmware reduces clock speeds to stay within
thermal limits.

This benchmark runs continuous inference for a set duration and
samples throughput every N seconds, revealing the throttling curve.

This is essentially unexplored territory in public benchmarks.
"""

import time
import uuid
import json
import httpx
import subprocess
from harness.metrics import BenchmarkResult, MemoryTracker

OLLAMA_URL = "http://localhost:11434"


def get_cpu_temp() -> float:
    """
    Get CPU temperature on macOS using powermetrics.
    Requires sudo — returns -1 if unavailable.
    We use 'cpu_thermal_level' as a proxy since direct temp
    readings require root on modern macOS.
    """
    try:
        result = subprocess.run(
            ["sudo", "powermetrics", "-n", "1", "-i", "100", "--samplers", "thermal"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.split("\n"):
            if "CPU die temperature" in line:
                return float(line.split(":")[-1].strip().replace("C", ""))
    except Exception:
        pass
    return -1.0


class ThermalBenchmark:
    name = "thermal"

    def __init__(self):
        self.memory_tracker = MemoryTracker()

    def run_sustained(
        self,
        model: str,
        duration_seconds: int = 600,  # 10 minutes
        sample_interval: int = 30,    # sample every 30 seconds
        max_new_tokens: int = 128,
        prompt: str = "Explain the history of neural networks in detail, covering perceptrons, backpropagation, deep learning, and transformers."
    ) -> list:
        """
        Run sustained inference and record throughput over time.
        Returns list of (elapsed_seconds, throughput, ttft_ms) tuples.
        """
        samples = []
        t_start = time.perf_counter()
        last_sample_time = t_start
        request_count = 0

        print(f"[Thermal] Running sustained benchmark for {duration_seconds}s on {model}")
        print(f"  Sampling every {sample_interval}s")

        while True:
            elapsed = time.perf_counter() - t_start
            if elapsed >= duration_seconds:
                break

            # Run one inference request
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {"num_predict": max_new_tokens, "temperature": 0},
            }

            first_token_time = None
            req_start = time.perf_counter()
            final_chunk = {}

            try:
                with httpx.stream("POST", f"{OLLAMA_URL}/api/generate", json=payload, timeout=300) as response:
                    for line in response.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        if not chunk.get("done", False) and first_token_time is None:
                            first_token_time = time.perf_counter()
                        if chunk.get("done", False):
                            final_chunk = chunk
            except Exception as e:
                print(f"  ⚠ Request failed: {e}")
                continue

            req_end = time.perf_counter()
            tokens = final_chunk.get("eval_count", 0)
            eval_ms = final_chunk.get("eval_duration", 0) / 1_000_000
            ttft_ms = (first_token_time - req_start) * 1000 if first_token_time else 0
            throughput = tokens / (eval_ms / 1000) if eval_ms > 0 else 0
            request_count += 1

            # Sample on interval
            now = time.perf_counter()
            if now - last_sample_time >= sample_interval:
                elapsed_min = elapsed / 60
                temp = get_cpu_temp()
                sample = {
                    "elapsed_seconds": round(elapsed),
                    "elapsed_minutes": round(elapsed_min, 1),
                    "throughput_tok_per_sec": round(throughput, 2),
                    "ttft_ms": round(ttft_ms, 1),
                    "request_count": request_count,
                    "cpu_temp_c": temp,
                }
                samples.append(sample)
                last_sample_time = now
                print(f"  t={elapsed_min:.1f}min | {throughput:.1f} tok/s | TTFT {ttft_ms:.0f}ms | temp {temp}°C")

        return samples