"""
Metrics utilities: CUDA event timing, statistics, and memory capture.
"""
from typing import Dict, List
import torch
import numpy as np


def compute_stats(times_ms: List[float]) -> Dict:
    arr = np.array(times_ms, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


def measure_kernel(
    fn,
    warmup: int,
    measure: int,
    trials: int,
    device: str = "cuda",
) -> Dict:
    """
    Run fn multiple times with CUDA events.
    Returns raw times_ms across all trials and stats.
    """
    assert torch.cuda.is_available(), "CUDA required"
    torch.cuda.synchronize(device)
    times_ms: List[float] = []

    for _ in range(trials):
        # warmup
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)

        # measure
        for _ in range(measure):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device)
            times_ms.append(start.elapsed_time(end))

    stats = compute_stats(times_ms)
    return {
        "times_ms": times_ms,
        "stats": stats,
    }


def memory_stats() -> Dict:
    torch.cuda.synchronize()
    return {
        "max_memory_allocated": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved": int(torch.cuda.max_memory_reserved()),
    }


def reset_memory():
    torch.cuda.reset_peak_memory_stats()
