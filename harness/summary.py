"""
Capacity-planning summaries for benchmark results.

This module turns raw measurements into the product-level answers the
dashboard and README should lead with: max safe concurrency, recommended
runtime, and likely bottleneck.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from harness.metrics import BenchmarkResult


@dataclass(frozen=True)
class CapacitySLO:
    ttft_ms: float = 1000.0
    p95_latency_ms: float = 10000.0


@dataclass(frozen=True)
class CapacityRecommendation:
    framework: str
    model: str
    max_safe_concurrency: int
    recommended: bool
    bottleneck: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "framework": self.framework,
            "model": self.model,
            "max_safe_concurrency": self.max_safe_concurrency,
            "recommended": self.recommended,
            "bottleneck": self.bottleneck,
            "reason": self.reason,
        }


def result_passes_slo(result: BenchmarkResult, slo: CapacitySLO) -> bool:
    return result.ttft_ms <= slo.ttft_ms and result.total_latency_ms <= slo.p95_latency_ms


def classify_bottleneck(results: List[BenchmarkResult], slo: CapacitySLO) -> str:
    if not results:
        return "unknown"

    ordered = sorted(results, key=lambda r: r.concurrency)
    first = ordered[0]
    last = ordered[-1]

    # Need at least two concurrency levels to detect queueing behavior
    if len(ordered) > 1:
        ttft_ratio = last.ttft_ms / max(first.ttft_ms, 1)
        if ttft_ratio >= 5 and last.ttft_ms > slo.ttft_ms:
            return "request_queueing"

    if last.memory_bandwidth_utilization_pct >= 70:
        return "memory_bandwidth"
    if last.throughput_stddev >= max(last.throughput_tok_per_sec * 0.15, 1):
        return "runtime_variance"
    return "model_execution"


def summarize_capacity(
    results: Iterable[BenchmarkResult],
    slo: CapacitySLO | None = None,
) -> Dict[str, object]:
    slo = slo or CapacitySLO()
    grouped: Dict[Tuple[str, str], List[BenchmarkResult]] = defaultdict(list)
    for result in results:
        grouped[(result.framework, result.model)].append(result)

    recommendations = []
    for (framework, model), group in grouped.items():
        safe = [
            r.concurrency for r in group
            if result_passes_slo(r, slo)
        ]
        max_safe = max(safe) if safe else 0
        bottleneck = classify_bottleneck(group, slo)
        reason = _build_reason(framework, model, max_safe, bottleneck, slo)
        recommendations.append(
            CapacityRecommendation(
                framework=framework,
                model=model,
                max_safe_concurrency=max_safe,
                recommended=False,
                bottleneck=bottleneck,
                reason=reason,
            )
        )

    best = _choose_best(recommendations)
    output = []
    for rec in recommendations:
        output.append(
            CapacityRecommendation(
                framework=rec.framework,
                model=rec.model,
                max_safe_concurrency=rec.max_safe_concurrency,
                recommended=best is not None and rec == best,
                bottleneck=rec.bottleneck,
                reason=rec.reason,
            ).to_dict()
        )

    return {
        "slo": {
            "ttft_ms": slo.ttft_ms,
            "p95_latency_ms": slo.p95_latency_ms,
        },
        "recommendations": output,
        "recommended_runtime": best.framework if best else None,
        "recommended_model": best.model if best else None,
    }


def _choose_best(
    recommendations: List[CapacityRecommendation],
) -> CapacityRecommendation | None:
    if not recommendations:
        return None
    return max(
        recommendations,
        key=lambda r: (
            r.max_safe_concurrency,
            1 if r.bottleneck == "memory_bandwidth" else 0,
            r.framework == "mlx",
        ),
    )


def _build_reason(
    framework: str,
    model: str,
    max_safe_concurrency: int,
    bottleneck: str,
    slo: CapacitySLO,
) -> str:
    short_model = model.split("/")[-1]
    if max_safe_concurrency == 0:
        return (
            f"{framework}/{short_model} exceeds the {slo.ttft_ms:.0f}ms TTFT SLO "
            "at all tested concurrency levels — not suitable for concurrent serving."
        )
    return (
        f"{framework}/{short_model} stays within the {slo.ttft_ms:.0f}ms TTFT SLO "
        f"through concurrency={max_safe_concurrency}; likely bottleneck: {bottleneck}."
    )