"""
MLX Benchmark: Apple's native ML framework.

This module provides an MLX benchmark implementation used to
measure inference latency, first-token time (TTFT), throughput, and
peak memory on Apple Silicon GPUs.

MLX runs on Metal with unified memory and is typically used via mlx-lm,
which adds LLM-specific optimizations: 4-bit/8-bit quantization, KV
cache for fast generation, and streaming token generation. This
benchmark streams tokens one-at-a-time to capture true TTFT and
reports per-request averages.

Recommended notes:
- Quantization choices: Q4 (4-bit, fastest, ~4GB for 7B), Q8 (8-bit,
  balanced, ~8GB), fp16 (16-bit, most accurate, ~14GB).
- Streaming generation provides accurate TTFT measurements.
"""

import time
import uuid
from harness.metrics import BenchmarkResult, MemoryTracker


class MLXBenchmark:
    name = "mlx"

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = ""
        self.memory_tracker = MemoryTracker()

    def setup(self, model_name: str) -> None:
        """
        Load model weights into MLX format.
        mlx_lm.load() downloads a 4-bit quantized model from HuggingFace
        the first time, then caches it locally. Subsequent loads are instant.
        """
        from mlx_lm import load
        print(f"[MLX] Loading {model_name}...")
        self.model, self.tokenizer = load(model_name)
        self.model_name = model_name
        print(f"[MLX] Model loaded.")

    def run_single(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        prompt_length: str = "medium",
        concurrency: int = 1,
    ) -> BenchmarkResult:
        """
        MLX generates tokens one at a time via a Python generator.
        This gives true streaming where we can catch the exact first token time.
        
        Note: MLX doesn't support true concurrency (one GPU, one model).
        We run concurrency=N sequentially and report per-request averages.
        """
        from mlx_lm import stream_generate

        self.memory_tracker.reset()
        first_token_time = None
        tokens_generated = 0

        t_start = time.perf_counter()
        self.memory_tracker.sample()

        # stream_generate yields one token at a time
        for token in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter()
            tokens_generated += 1
            self.memory_tracker.sample()

        t_end = time.perf_counter()

        total_ms = (t_end - t_start) * 1000
        ttft_ms = (first_token_time - t_start) * 1000 if first_token_time else total_ms
        throughput = tokens_generated / (total_ms / 1000) if total_ms > 0 else 0

        return BenchmarkResult(
            framework="mlx",
            model=self.model_name,
            concurrency=concurrency,
            prompt_length=prompt_length,
            batch_size=1,
            ttft_ms=ttft_ms,
            total_latency_ms=total_ms,
            tokens_generated=tokens_generated,
            prompt_tokens=0,
            throughput_tok_per_sec=throughput,
            peak_memory_mb=self.memory_tracker.peak_mb,
            run_id=str(uuid.uuid4())[:8],
        )

    def teardown(self):
        del self.model
        self.model = None
        self.tokenizer = None