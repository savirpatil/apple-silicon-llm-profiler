"""
Thinking mode benchmark — Qwen3 and other modern models support
an extended reasoning mode where the model "thinks" through a
problem before answering, using a <think>...</think> block.

This is controlled via the 'thinking' parameter in Ollama.
Thinking mode uses more tokens internally but produces better
answers on complex problems. The tradeoff: more latency, better quality.

This benchmark measures:
- TTFT with thinking on vs off
- Total latency with thinking on vs off  
- Tokens used in thinking vs answer
- Throughput impact of thinking mode

This is extremely current — Qwen3 shipped thinking mode in early 2025.
"""

import time
import uuid
import json
import httpx
from harness.metrics import BenchmarkResult, MemoryTracker

OLLAMA_URL = "http://localhost:11434"

# These prompts benefit most from thinking mode —
# complex reasoning tasks where careful thought helps
THINKING_PROMPTS = {
    "simple": "What is the capital of France?",
    "reasoning": "If a train travels 120km in 90 minutes, then speeds up by 20%, how long will it take to travel the next 150km?",
    "coding": "Write a Python function that finds all prime numbers up to n using the Sieve of Eratosthenes.",
    "complex": "Explain the tradeoffs between consistency and availability in distributed systems, and give a concrete example of when you'd choose each.",
}


class ThinkingBenchmark:
    name = "thinking_mode"

    def __init__(self):
        self.memory_tracker = MemoryTracker()

    def run_single(
        self,
        model: str,
        prompt: str,
        thinking: bool,
        max_new_tokens: int = 512,
    ) -> dict:
        """
        Run one inference with thinking on or off.
        Returns detailed metrics including thinking token count.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": 0,
            },
        }

        # Qwen3 thinking mode is controlled via /think or /no_think
        # in the system prompt when using Ollama
        if thinking:
            payload["system"] = "/think"
        else:
            payload["system"] = "/no_think"

        first_token_time = None
        t_start = time.perf_counter()
        final_chunk = {}
        full_response = ""
        self.memory_tracker.reset()

        with httpx.stream("POST", f"{OLLAMA_URL}/api/generate", json=payload, timeout=300) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if not chunk.get("done", False):
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    full_response += chunk.get("response", "")
                if chunk.get("done", False):
                    final_chunk = chunk
                self.memory_tracker.sample()

        t_end = time.perf_counter()
        tokens = final_chunk.get("eval_count", 0)
        eval_ms = final_chunk.get("eval_duration", 0) / 1_000_000
        total_ms = (t_end - t_start) * 1000
        ttft_ms = (first_token_time - t_start) * 1000 if first_token_time else total_ms
        throughput = tokens / (eval_ms / 1000) if eval_ms > 0 else 0

        # Count thinking tokens (inside <think>...</think>)
        thinking_tokens = 0
        if "<think>" in full_response and "</think>" in full_response:
            think_start = full_response.find("<think>")
            think_end = full_response.find("</think>") + len("</think>")
            thinking_text = full_response[think_start:think_end]
            thinking_tokens = len(thinking_text.split())

        return {
            "model": model,
            "prompt_type": "",
            "thinking": thinking,
            "ttft_ms": round(ttft_ms, 1),
            "total_latency_ms": round(total_ms, 1),
            "tokens_generated": tokens,
            "thinking_tokens": thinking_tokens,
            "answer_tokens": tokens - thinking_tokens,
            "throughput_tok_per_sec": round(throughput, 2),
            "peak_memory_mb": round(self.memory_tracker.peak_mb, 1),
        }

    def run_sweep(self, model: str = "qwen3:8b", warmup: int = 1, runs: int = 3) -> list:
        results = []

        for prompt_type, prompt in THINKING_PROMPTS.items():
            print(f"\n[Thinking] Prompt: {prompt_type}")

            for thinking in [False, True]:
                mode = "thinking ON" if thinking else "thinking OFF"
                print(f"  Mode: {mode}")

                # Warmup
                for _ in range(warmup):
                    self.run_single(model, prompt, thinking)

                run_results = []
                for i in range(runs):
                    r = self.run_single(model, prompt, thinking)
                    run_results.append(r)
                    print(f"  Run {i+1}/{runs}: {r['throughput_tok_per_sec']:.1f} tok/s | "
                          f"TTFT {r['ttft_ms']:.0f}ms | "
                          f"thinking tokens: {r['thinking_tokens']}")

                avg = {k: (sum(r[k] for r in run_results) / len(run_results)
                           if isinstance(run_results[0][k], (int, float))
                           else run_results[0][k])
                       for k in run_results[0]}
                avg["prompt_type"] = prompt_type
                avg["thinking"] = thinking
                results.append(avg)
                print(f"  ✓ AVG: {avg['throughput_tok_per_sec']:.1f} tok/s | "
                      f"TTFT {avg['ttft_ms']:.0f}ms")

        return results