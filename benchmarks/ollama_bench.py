import time
import uuid
import json
import httpx
from harness.metrics import BenchmarkResult, MemoryTracker

OLLAMA_URL = "http://localhost:11434"


class OllamaBenchmark:
    """
    Benchmarks Ollama via its streaming HTTP API.
    
    Why streaming? Because it lets us measure TTFT precisely —
    we record the exact moment the first token arrives, not just
    when the whole response finishes. This is the metric that
    matters most for interactive use.
    """
    name = "ollama"

    def __init__(self):
        self.memory_tracker = MemoryTracker()

    def run_single(
        self,
        prompt: str,
        model: str,
        max_new_tokens: int = 128,
        prompt_length: str = "medium",
        concurrency: int = 1,
        batch_size: int = 1,
    ) -> BenchmarkResult:
        self.memory_tracker.reset()

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": 0,
            }
        }

        first_token_time = None
        t_start = time.perf_counter()
        final_chunk = {}

        with httpx.stream(
            "POST",
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=300
        ) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                
                # First non-done chunk = first token arrived
                if not chunk.get("done", False) and first_token_time is None:
                    first_token_time = time.perf_counter()
                
                # Ollama puts final stats in the done=True chunk
                if chunk.get("done", False):
                    final_chunk = chunk
                
                self.memory_tracker.sample()

        t_end = time.perf_counter()

        # Use Ollama's own token counts — these are exact
        # eval_count = tokens generated, prompt_eval_count = prompt tokens
        tokens_generated = final_chunk.get("eval_count", 0)
        prompt_tokens = final_chunk.get("prompt_eval_count", 0)
        
        # eval_duration is in nanoseconds — convert to ms
        eval_duration_ms = final_chunk.get("eval_duration", 0) / 1_000_000

        total_ms = (t_end - t_start) * 1000
        ttft_ms = (first_token_time - t_start) * 1000 if first_token_time else total_ms

        # Throughput = tokens / time spent generating (not including load time)
        throughput = tokens_generated / (eval_duration_ms / 1000) if eval_duration_ms > 0 else 0

        return BenchmarkResult(
            framework="ollama",
            model=model,
            concurrency=concurrency,
            prompt_length=prompt_length,
            batch_size=batch_size,
            ttft_ms=ttft_ms,
            total_latency_ms=total_ms,
            tokens_generated=tokens_generated,
            prompt_tokens=prompt_tokens,
            throughput_tok_per_sec=throughput,
            peak_memory_mb=self.memory_tracker.peak_mb,
            run_id=str(uuid.uuid4())[:8],
        )