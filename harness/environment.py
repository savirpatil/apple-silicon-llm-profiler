"""
Environment helpers and runtime detection for benchmarks.

This module provides utilities to detect and configure the local
execution environment (macOS / Apple Silicon), query GPU/CPU
capabilities, and set recommended environment variables for
ML inference frameworks used by the harness.

Core responsibilities:
- Query device info (CPU model, chip family, amount of RAM, GPU/Metal info).
- Provide helper functions to set or validate env vars (e.g., MLX, vLLM).
- Small convenience helpers used by setup/verify scripts and runners.

Recommended notes:
- Keep platform-specific checks isolated here.
- Return simple, serializable dicts for logging and result metadata.
"""

import platform
import subprocess
import psutil


def get_environment_info() -> dict:
    """Capture full system context at benchmark time."""
    
    info = {
        "os": platform.system(),
        "os_version": platform.mac_ver()[0],
        "python_version": platform.python_version(),
        "chip": _get_chip_info(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ollama_version": _get_ollama_version(),
        "mlx_version": _get_mlx_version(),
    }
    return info


def _get_chip_info() -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            return result.stdout.strip()
        result = subprocess.run(
            ["sysctl", "-n", "hw.model"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return platform.processor() or "unknown"


def _get_ollama_version() -> str:
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _get_mlx_version() -> str:
    try:
        import mlx.core as mx
        return mx.__version__
    except Exception:
        return "not installed"


def get_memory_bandwidth_utilization(
    tokens_generated: int,
    eval_time_ms: float,
    model_size_gb: float,
    context_length: int = 128,
) -> float:
    """
    Calculate what percentage of Apple Silicon's theoretical peak
    memory bandwidth is being utilized during inference.
    
    Why this matters: LLM inference on Apple Silicon is memory-bandwidth-bound,
    not compute-bound. The M4 has ~120GB/s theoretical bandwidth.
    Every token generated requires loading the full model weights from memory.
    
    Formula: (tokens × model_bytes) / (time × peak_bandwidth)
    
    A utilization of 50-70% is typical for well-optimized inference.
    Below 30% suggests overhead or inefficiency.
    """
    if eval_time_ms <= 0 or tokens_generated <= 0:
        return 0.0
    
    M4_PEAK_BANDWIDTH_GBs = 120.0  # GB/s for M4
    model_bytes = model_size_gb * (1024**3)
    eval_time_s = eval_time_ms / 1000
    
    # Bytes moved = tokens × model size (each token requires one full model pass)
    bytes_moved = tokens_generated * model_bytes
    actual_bandwidth = bytes_moved / eval_time_s / (1024**3)  # GB/s
    
    utilization_pct = (actual_bandwidth / M4_PEAK_BANDWIDTH_GBs) * 100
    return round(min(utilization_pct, 100.0), 2)


def print_environment():
    info = get_environment_info()
    print("\n=== Environment ===")
    for k, v in info.items():
        print(f"  {k:20s}: {v}")
    print()