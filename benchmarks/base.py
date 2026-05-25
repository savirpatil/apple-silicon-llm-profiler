"""
Base benchmark interface used by all benchmark implementations.

This module provides a small BaseBenchmark abstract
class to standardize lifecycle and return types across the project.
Implementations should prepare any runner or tokenizer in setup(),
perform a single measured inference/workload in run_single() and
return a harness.metrics.BenchmarkResult, and release resources in
teardown().

Core responsibilities:
- setup(model): load and warm model/runner/tokenizer or other resources.
- run_single(prompt, max_new_tokens): execute one measured run and
  return a BenchmarkResult (TTFT, latency, throughput, peak memory).
- teardown(): free resources and clear state.
"""

from abc import ABC, abstractmethod
from harness.metrics import BenchmarkResult


class BaseBenchmark(ABC):
    name: str = "base"

    @abstractmethod
    def setup(self, model: str) -> None:
        """Load model, warm up."""
        pass

    @abstractmethod
    def run_single(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> BenchmarkResult:
        """Run a single inference, return metrics."""
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Release resources."""
        pass