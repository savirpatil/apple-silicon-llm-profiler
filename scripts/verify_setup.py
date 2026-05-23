#!/usr/bin/env python3
"""Run this before any benchmarks to confirm everything is installed."""

import sys
import subprocess

results = {}

def check(name, fn):
    try:
        fn()
        results[name] = "✅"
    except Exception as e:
        results[name] = f"❌ {e}"

def check_vllm():
    import vllm
    assert hasattr(vllm, "LLM"), "LLM class not found"

def check_sglang():
    import sglang
    _ = sglang.__version__

def check_mlx():
    import mlx.core as mx
    x = mx.array([1.0, 2.0])
    assert x.shape == (2,)

def check_wandb():
    import wandb
    assert wandb.__version__

def check_ollama():
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    assert "qwen3" in result.stdout.lower() or "llama" in result.stdout.lower(), \
        f"Expected models not found. Got:\n{result.stdout}"

def check_psutil():
    import psutil
    mem = psutil.virtual_memory()
    print(f"  System RAM: {mem.total / (1024**3):.1f} GB")

check("vLLM", check_vllm)
check("SGLang", check_sglang)
check("MLX", check_mlx)
check("W&B", check_wandb)
check("Ollama + models", check_ollama)
check("psutil/memory", check_psutil)

print("\n=== Setup Verification ===")
for name, status in results.items():
    print(f"  {name:20s} {status}")

failed = [k for k, v in results.items() if v.startswith("❌")]
if failed:
    print(f"\n⚠️  Fix these before benchmarking: {', '.join(failed)}")
    sys.exit(1)
else:
    print("\n🎉 All checks passed. Ready to benchmark.")