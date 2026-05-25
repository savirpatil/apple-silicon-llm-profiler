"""
SGLang Benchmark: short-generation language tasks.

This module runs compact generation workloads (short prompts and
small outputs) to evaluate latency and TTFT for lightweight usage
patterns. It is useful for micro-benchmarks and measuring interactive
responsiveness where single-token latency matters most.

Recommended notes:
- Focus on short prompts and low max_new_tokens to capture TTFT.
- Useful for comparing interactive performance across runners.
- Keep implementations minimal and deterministic for repeatability.
"""

import time
import uuid
import requests
import subprocess
from benchmarks.base import BaseBenchmark
from harness.metrics import BenchmarkResult, MemoryTracker


class SGLangBenchmark(BaseBenchmark):
    """
    SGLang is run as a server; we benchmark via HTTP requests.
    This mirrors real-world usage more accurately.
    """
    name = "sglang"

    def __init__(self, port: int = 30000):
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self.server_proc = None
        self.memory_tracker = MemoryTracker()

    def setup(self, model: str) -> None:
        print(f"[SGLang] Starting server with {model}...")
        self.model = model
        self.server_proc = subprocess.Popen(
            [
                "python", "-m", "sglang.launch_server",
                "--model-path", model,
                "--port", str(self.port),
                "--device", "cpu",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._wait_for_server(timeout=120)
        print(f"[SGLang] Server ready at {self.base_url}")

    def _wait_for_server(self, timeout: int = 120):
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.base_url}/health", timeout=2)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError("SGLang server failed to start")

    def run_single(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        prompt_length: str = "medium",
        concurrency: int = 1,
        batch_size: int = 1,
        model: str = "",
    ) -> BenchmarkResult:
        self.memory_tracker.reset()

        payload = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
            },
        }

        t_start = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=300,
        )
        t_end = time.perf_counter()

        data = response.json()
        output_text = data.get("text", "")
        total_tokens = len(output_text.split())
        total_ms = (t_end - t_start) * 1000

        meta = data.get("meta_info", {})
        ttft_ms = meta.get("ttft_s", total_ms / max(total_tokens, 1) / 1000) * 1000

        return BenchmarkResult(
            framework="sglang",
            model=model,
            concurrency=concurrency,
            prompt_length=prompt_length,
            batch_size=batch_size,
            ttft_ms=ttft_ms,
            total_latency_ms=total_ms,
            tokens_generated=total_tokens,
            throughput_tok_per_sec=total_tokens / (total_ms / 1000),
            peak_memory_mb=self.memory_tracker.peak_mb,
            run_id=str(uuid.uuid4())[:8],
        )

    def teardown(self) -> None:
        if self.server_proc:
            self.server_proc.terminate()
            self.server_proc.wait()
            self.server_proc = None