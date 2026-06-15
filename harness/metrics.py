"""
Benchmark metrics primitives and helpers.

This module defines the core measurement types and utilities used by
benchmarks: BenchmarkResult (latency, TTFT, throughput, peak memory),
MemoryTracker, and helpers to aggregate and serialize measurements.

Core responsibilities:
- Provide simple, typed metric containers that are easy to log/serialize.
- Track memory and timing samples during runs.
- Offer aggregation helpers (mean, median, per-token throughput).

Recommended notes:
- Measurements should be backend-agnostic and comparable across runs.
- Keep serialization stable (JSON schema) for downstream reporting.
"""

import time
import psutil
import subprocess
import statistics
from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class BenchmarkResult:
    framework: str
    model: str
    concurrency: int
    prompt_length: str
    batch_size: int
    ttft_ms: float
    total_latency_ms: float
    tokens_generated: int
    throughput_tok_per_sec: float
    peak_memory_mb: float
    run_id: str = ""
    error: Optional[str] = None
    raw_output: str = ""
    timestamp: float = field(default_factory=time.time)
    prompt_tokens: int = 0
    # Statistics fields
    throughput_stddev: float = 0.0
    ttft_stddev: float = 0.0
    throughput_p50: float = 0.0
    throughput_p95: float = 0.0
    ttft_p95: float = 0.0
    # Memory bandwidth
    memory_bandwidth_utilization_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "framework": self.framework,
            "model": self.model,
            "concurrency": self.concurrency,
            "prompt_length": self.prompt_length,
            "batch_size": self.batch_size,
            "ttft_ms": self.ttft_ms,
            "total_latency_ms": self.total_latency_ms,
            "tokens_generated": self.tokens_generated,
            "prompt_tokens": self.prompt_tokens,
            "throughput_tok_per_sec": self.throughput_tok_per_sec,
            "throughput_stddev": self.throughput_stddev,
            "throughput_p50": self.throughput_p50,
            "throughput_p95": self.throughput_p95,
            "ttft_p95": self.ttft_p95,
            "ttft_stddev": self.ttft_stddev,
            "memory_bandwidth_utilization_pct": self.memory_bandwidth_utilization_pct,
            "peak_memory_mb": self.peak_memory_mb,
            "run_id": self.run_id,
            "error": self.error,
            "timestamp": self.timestamp,
        }


def compute_statistics(results: List[BenchmarkResult]) -> BenchmarkResult:
    """
    Takes multiple runs of the same condition and returns a single
    result with mean values and statistical spread metrics.
    
    stddev tells you how stable the number is: a low stddev means
    the benchmark is reproducible. A high stddev means the system
    was doing other things (thermal throttling, memory pressure).
    
    p95 = 95th percentile: the worst case 1 in 20 requests sees.
    This is more useful than max (which catches flukes) and more
    honest than mean (which hides tail latency).
    """
    from harness.environment import get_memory_bandwidth_utilization

    n = len(results)
    if n == 0:
        raise ValueError("cannot compute statistics for an empty result set")

    tps_values = [r.throughput_tok_per_sec for r in results]
    ttft_values = [r.ttft_ms for r in results]

    # Model size lookup (approximate GB for bandwidth calculation)
    model_size_map = {
        "7b": 4.0, "8b": 4.5, "1.7b": 1.0,
        "qwen2.5": 4.0, "llama3.1": 4.5, "mlx": 4.0,
    }
    model_lower = results[0].model.lower()
    model_size_gb = 4.0  # default
    for key, size in model_size_map.items():
        if key in model_lower:
            model_size_gb = size
            break

    avg_tokens = sum(r.tokens_generated for r in results) / n
    avg_eval_ms = sum(r.total_latency_ms for r in results) / n
    bw_util = get_memory_bandwidth_utilization(
        int(avg_tokens), avg_eval_ms, model_size_gb
    )

    r = results[0]
    return BenchmarkResult(
        framework=r.framework,
        model=r.model,
        concurrency=r.concurrency,
        prompt_length=r.prompt_length,
        batch_size=r.batch_size,
        ttft_ms=round(sum(ttft_values) / n, 2),
        total_latency_ms=round(sum(r.total_latency_ms for r in results) / n, 2),
        tokens_generated=int(avg_tokens),
        prompt_tokens=r.prompt_tokens,
        throughput_tok_per_sec=round(sum(tps_values) / n, 2),
        throughput_stddev=round(statistics.stdev(tps_values) if n > 1 else 0.0, 2),
        throughput_p50=round(statistics.median(tps_values), 2),
        throughput_p95=round(_percentile(tps_values, 0.95), 2),
        ttft_p95=round(_percentile(ttft_values, 0.95), 2),
        ttft_stddev=round(statistics.stdev(ttft_values) if n > 1 else 0.0, 2),
        peak_memory_mb=max(r.peak_memory_mb for r in results),
        memory_bandwidth_utilization_pct=bw_util,
        run_id=r.run_id,
    )


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


class MemoryTracker:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self._peak_mb = 0.0

    def reset(self):
        self._peak_mb = 0.0

    def sample(self) -> float:
        try:
            mem_mb = self.process.memory_info().rss / (1024 * 1024)
            self._peak_mb = max(self._peak_mb, mem_mb)
            return mem_mb
        except Exception:
            return 0.0

    @property
    def peak_mb(self) -> float:
        return self._peak_mb

    @staticmethod
    def get_system_memory_pressure() -> str:
        try:
            result = subprocess.run(
                ["memory_pressure"],
                capture_output=True, text=True, timeout=5
            )
            if "System-wide memory free percentage" in result.stdout:
                return result.stdout.strip()
        except Exception:
            pass
        return "unavailable"
