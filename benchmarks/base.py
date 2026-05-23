from abc import ABC, abstractmethod
from typing import List
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