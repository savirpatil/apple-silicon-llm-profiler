#!/usr/bin/env python3
"""Verify the core Apple Silicon Ollama/MLX benchmark environment."""

import os
import platform
import subprocess
import sys


results = {}


def check(name, fn):
    try:
        fn()
        results[name] = "OK"
    except Exception as e:
        results[name] = f"FAIL: {e}"


def check_apple_silicon():
    assert platform.system() == "Darwin", "requires macOS"
    machine = platform.machine().lower()
    assert machine == "arm64", f"expected arm64 Apple Silicon, got {machine}"


def check_core_imports():
    import httpx
    import numpy
    import psutil
    import rich
    import yaml

    assert httpx.__version__
    assert numpy.__version__
    assert psutil.virtual_memory().total > 0
    assert rich
    assert yaml


def check_mlx():
    import mlx.core as mx

    x = mx.array([1.0, 2.0])
    assert x.shape == (2,)


def check_wandb_optional():
    import wandb

    assert wandb.__version__
    if not os.getenv("WANDB_API_KEY"):
        print("  WANDB_API_KEY not set; W&B logging will be skipped unless configured.")


def check_ollama():
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr.strip() or "ollama list failed"
    model_lines = [
        line for line in result.stdout.splitlines()[1:]
        if line.strip()
    ]
    assert model_lines, "no Ollama models found; run `ollama pull qwen2.5:7b`"
    print(f"  Ollama models found: {len(model_lines)}")


check("Apple Silicon", check_apple_silicon)
check("Core Python deps", check_core_imports)
check("MLX", check_mlx)
check("W&B optional", check_wandb_optional)
check("Ollama + models", check_ollama)

print("\n=== Setup Verification ===")
for name, status in results.items():
    print(f"  {name:20s} {status}")

failed = [k for k, v in results.items() if v.startswith("FAIL")]
if failed:
    print(f"\nFix these before benchmarking: {', '.join(failed)}")
    sys.exit(1)

print("\nAll core checks passed. Ready to benchmark.")
