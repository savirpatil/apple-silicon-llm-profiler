"""
Configuration loading for the Apple Silicon LLM capacity profiler.

The YAML file is the single source of truth for CLI and UI benchmark
defaults: frameworks, model mappings, prompts, concurrency, run counts,
and latency SLOs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "benchmark_config.yaml"


@dataclass(frozen=True)
class PromptConfig:
    name: str
    text: str
    target_tokens: int


@dataclass(frozen=True)
class ModelConfig:
    key: str
    ollama_name: Optional[str] = None
    mlx_name: Optional[str] = None


@dataclass(frozen=True)
class SLOConfig:
    ttft_ms: float = 1000.0
    p95_latency_ms: float = 10000.0


@dataclass(frozen=True)
class BenchmarkConfig:
    frameworks: List[str]
    concurrency_levels: List[int]
    prompt_lengths: Dict[str, PromptConfig]
    models: Dict[str, ModelConfig]
    max_new_tokens: int
    warmup_runs: int
    benchmark_runs: int
    slo: SLOConfig

    @property
    def ollama_models(self) -> List[str]:
        return [
            model.ollama_name for model in self.models.values()
            if model.ollama_name
        ]

    @property
    def mlx_models(self) -> List[str]:
        return [
            model.mlx_name for model in self.models.values()
            if model.mlx_name
        ]


def load_benchmark_config(path: str | Path = DEFAULT_CONFIG_PATH) -> BenchmarkConfig:
    config_path = Path(path)
    with config_path.open() as f:
        raw = yaml.safe_load(f) or {}

    prompt_lengths = {
        name: PromptConfig(
            name=name,
            text=values["text"],
            target_tokens=int(values.get("tokens", 0)),
        )
        for name, values in raw.get("prompt_lengths", {}).items()
    }

    models = {
        key: ModelConfig(
            key=key,
            ollama_name=values.get("ollama_name"),
            mlx_name=values.get("mlx_name"),
        )
        for key, values in raw.get("models", {}).items()
    }

    slo_raw = raw.get("slo", {})
    config = BenchmarkConfig(
        frameworks=list(raw.get("frameworks", ["ollama", "mlx"])),
        concurrency_levels=[int(c) for c in raw.get("concurrency_levels", [1])],
        prompt_lengths=prompt_lengths,
        models=models,
        max_new_tokens=int(raw.get("max_new_tokens", 128)),
        warmup_runs=int(raw.get("num_warmup_runs", 2)),
        benchmark_runs=int(raw.get("num_benchmark_runs", 3)),
        slo=SLOConfig(
            ttft_ms=float(slo_raw.get("ttft_ms", 1000)),
            p95_latency_ms=float(slo_raw.get("p95_latency_ms", 10000)),
        ),
    )
    validate_benchmark_config(config)
    return config


def validate_benchmark_config(config: BenchmarkConfig) -> None:
    if not config.frameworks:
        raise ValueError("config must include at least one framework")
    unknown = sorted(set(config.frameworks) - {"ollama", "mlx"})
    if unknown:
        raise ValueError(f"unsupported frameworks: {unknown}")
    if not config.models:
        raise ValueError("config must include at least one model")
    if not config.prompt_lengths:
        raise ValueError("config must include at least one prompt length")
    if not config.concurrency_levels:
        raise ValueError("config must include at least one concurrency level")
    if config.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if config.benchmark_runs <= 0:
        raise ValueError("num_benchmark_runs must be positive")
