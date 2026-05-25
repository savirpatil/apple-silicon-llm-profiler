"""
Lightweight logging / structured output utilities.

This module wraps Python logging to provide consistent
logs across scripts and benchmarks. It exposes a small API
for console and file logging as well as JSON lines
output for experiment metadata.

Core responsibilities:
- Configure root logger with consistent format and levels.
- Provide helpers for structured JSON logs and experiment headers.
- Utility to capture and redirect external stdout/stderr when needed.

Recommended notes:
- Keep log format stable for automated parsing.
- Use the JSON-lines helper for benchmark result exports.
"""

import wandb
import os
from typing import List, Optional
from dotenv import load_dotenv
from harness.metrics import BenchmarkResult

load_dotenv()


class BenchmarkLogger:
    def __init__(self, project: Optional[str] = None, run_name: Optional[str] = None):
        self.project = project or os.getenv("WANDB_PROJECT", "apple-silicon-llm-bench")
        self.entity = os.getenv("WANDB_ENTITY")
        self.run_name = run_name
        self._run = None

    def init(self, config: dict):
        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.run_name,
            config=config,
            tags=["apple-silicon", "m4", "inference-benchmark"],
        )
        return self._run

    def log_result(self, result: BenchmarkResult):
        """
        Log a single result. We include all dimensions as metrics
        so W&B lets us plot throughput vs concurrency, TTFT vs prompt_length, etc.
        """
        if self._run is None:
            return
        wandb.log({
            **result.to_dict(),
            f"{result.framework}/throughput_tok_per_sec": result.throughput_tok_per_sec,
            f"{result.framework}/ttft_ms": result.ttft_ms,
            f"{result.framework}/total_latency_ms": result.total_latency_ms,
            f"{result.framework}/tokens_generated": result.tokens_generated,
        })

    def log_summary_table(self, results: List[BenchmarkResult]):
        """Log the full results as a W&B Table for easy comparison."""
        if self._run is None or not results:
            return
        columns = list(results[0].to_dict().keys())
        data = [list(r.to_dict().values()) for r in results]
        table = wandb.Table(columns=columns, data=data)
        wandb.log({"sweep_results": table})

    def finish(self):
        if self._run:
            wandb.finish()