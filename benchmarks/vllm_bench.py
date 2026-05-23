import time
import uuid
from typing import Optional
from benchmarks.base import BaseBenchmark
from harness.metrics import BenchmarkResult, MemoryTracker


class VLLMBenchmark(BaseBenchmark):
    name = "vllm"

    def __init__(self):
        self.llm = None
        self.memory_tracker = MemoryTracker()

    def setup(self, model: str) -> None:
        from vllm import LLM, SamplingParams
        print(f"[vLLM] Loading {model} in CPU mode...")
        self.llm = LLM(
            model=model,
            device="cpu",
            enforce_eager=True,
            max_model_len=2048,
            dtype="float32",
        )
        self.SamplingParams = SamplingParams
        print(f"[vLLM] Model loaded.")

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
        sampling_params = self.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
        )

        prompts = [prompt] * batch_size
        t_start = time.perf_counter()
        self.memory_tracker.sample()

        outputs = self.llm.generate(prompts, sampling_params)

        t_end = time.perf_counter()
        self.memory_tracker.sample()

        total_tokens = sum(
            len(o.outputs[0].token_ids) for o in outputs
        )
        total_ms = (t_end - t_start) * 1000
        ttft_ms = total_ms / max(total_tokens, 1)

        return BenchmarkResult(
            framework="vllm",
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
        del self.llm
        self.llm = None