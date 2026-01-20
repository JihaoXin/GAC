"""High-precision CUDA event timing and performance metrics."""
import torch
from typing import List, Dict, Callable
import numpy as np

from .utils import compute_statistics


def benchmark_kernel(
    kernel_fn: Callable,
    warmup_iterations: int = 10,
    measurement_iterations: int = 100,
    device: str = "cuda:0"
) -> Dict[str, float]:
    """
    Benchmark a CUDA kernel using CUDA events for high-precision timing.
    
    Args:
        kernel_fn: Function to benchmark (should be a no-arg callable that performs the operation)
        warmup_iterations: Number of warmup iterations
        measurement_iterations: Number of measurement iterations
        device: CUDA device string
    
    Returns:
        Dictionary with timing statistics (in milliseconds)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    # Synchronize before starting
    torch.cuda.synchronize(device)
    
    # Warmup
    for _ in range(warmup_iterations):
        kernel_fn()
    
    # Synchronize after warmup
    torch.cuda.synchronize(device)
    
    # Measurement
    events = []
    for _ in range(measurement_iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        kernel_fn()
        end_event.record()
        
        events.append((start_event, end_event))
    
    # Synchronize and extract timings
    torch.cuda.synchronize(device)
    
    times_ms = []
    for start_event, end_event in events:
        elapsed_ms = start_event.elapsed_time(end_event)
        times_ms.append(elapsed_ms)
    
    # Compute statistics
    stats = compute_statistics(times_ms)
    
    # Convert to seconds for consistency
    stats["times_ms"] = times_ms
    stats["times_s"] = [t / 1000.0 for t in times_ms]
    
    return stats


def compute_gemm_tflops(m: int, n: int, k: int, time_s: float) -> float:
    """Compute achieved TFLOPs for GEMM operation."""
    # GEMM: C = A @ B where A is (M, K) and B is (K, N)
    # Operations: 2 * M * N * K (multiply-add)
    flops = 2 * m * n * k
    tflops = flops / (time_s * 1e12)
    return tflops


def compute_gemm_bandwidth(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    time_s: float
) -> float:
    """Compute achieved memory bandwidth in GB/s for GEMM."""
    # Memory access: read A (M*K), read B (K*N), write C (M*N)
    dtype_size_bytes = torch.tensor([], dtype=dtype).element_size()
    total_bytes = (m * k + k * n + m * n) * dtype_size_bytes
    bandwidth_gbs = total_bytes / (time_s * 1e9)
    return bandwidth_gbs


def benchmark_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    warmup_iterations: int = 10,
    measurement_iterations: int = 100,
) -> Dict:
    """
    Benchmark a GEMM operation and return comprehensive metrics.
    
    Args:
        a: Left tensor (M, K)
        b: Right tensor (K, N)
        warmup_iterations: Number of warmup iterations
        measurement_iterations: Number of measurement iterations
    
    Returns:
        Dictionary with timing and performance metrics
    """
    m, k_a = a.shape
    k_b, n = b.shape
    
    if k_a != k_b:
        raise ValueError(f"Dimension mismatch: k_a={k_a}, k_b={k_b}")
    
    k = k_a
    
    def kernel_fn():
        torch.matmul(a, b)
    
    stats = benchmark_kernel(
        kernel_fn,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        device=str(a.device)
    )
    
    # Compute performance metrics for mean time
    mean_time_s = stats["mean"] / 1000.0
    tflops = compute_gemm_tflops(m, n, k, mean_time_s)
    bandwidth_gbs = compute_gemm_bandwidth(m, n, k, a.dtype, mean_time_s)
    
    # Compute metrics for each measurement
    tflops_list = [compute_gemm_tflops(m, n, k, t / 1000.0) for t in stats["times_ms"]]
    bandwidth_list = [compute_gemm_bandwidth(m, n, k, a.dtype, t / 1000.0) for t in stats["times_ms"]]
    
    return {
        "shape": {"m": m, "n": n, "k": k},
        "dtype": str(a.dtype),
        "timing": stats,
        "performance": {
            "tflops_mean": tflops,
            "tflops_stats": compute_statistics(tflops_list),
            "bandwidth_gbs_mean": bandwidth_gbs,
            "bandwidth_gbs_stats": compute_statistics(bandwidth_list),
        }
    }
