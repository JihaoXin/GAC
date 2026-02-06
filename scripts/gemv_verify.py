#!/usr/bin/env python3
"""Verify GEMV latency measurement accuracy."""

import torch
import numpy as np
import time

def benchmark_gemv_careful(N, K, repeats=500, warmup=100):
    """More careful GEMV benchmark with multiple measurement methods."""
    A = torch.randn(N, K, dtype=torch.float16, device='cuda')
    x = torch.randn(K, dtype=torch.float16, device='cuda')

    # Warmup
    for _ in range(warmup):
        y = A @ x
    torch.cuda.synchronize()

    # Method 1: CUDA events
    times_cuda = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        y = A @ x
        end.record()
        torch.cuda.synchronize()
        times_cuda.append(start.elapsed_time(end))

    # Method 2: Python time with sync
    times_python = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = A @ x
        torch.cuda.synchronize()
        times_python.append((time.perf_counter() - t0) * 1000)

    return {
        'cuda_mean': np.mean(times_cuda),
        'cuda_std': np.std(times_cuda),
        'cuda_min': np.min(times_cuda),
        'cuda_max': np.max(times_cuda),
        'python_mean': np.mean(times_python),
        'python_std': np.std(times_python),
    }

print("=" * 70)
print("GEMV Latency Verification")
print("=" * 70)

# Test specific dimensions
test_dims = [256, 512, 1024, 1536, 2048, 3072, 3500, 3800, 4000, 4096, 4200]

print(f"\n{'K':>6} | {'CUDA mean':>10} | {'CUDA std':>9} | {'CUDA min':>9} | {'Python mean':>11}")
print("-" * 65)

for K in test_dims:
    result = benchmark_gemv_careful(4096, K, repeats=500)
    print(f"{K:>6} | {result['cuda_mean']:>8.3f}μs | {result['cuda_std']:>7.3f}μs | {result['cuda_min']:>7.3f}μs | {result['python_mean']:>9.3f}μs")

# Check kernel launch overhead
print("\n" + "=" * 70)
print("Kernel Launch Overhead Test (empty kernel vs tiny GEMV)")
print("=" * 70)

# Tiny GEMV
A_tiny = torch.randn(8, 8, dtype=torch.float16, device='cuda')
x_tiny = torch.randn(8, dtype=torch.float16, device='cuda')

for _ in range(100):
    _ = A_tiny @ x_tiny
torch.cuda.synchronize()

times = []
for _ in range(1000):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _ = A_tiny @ x_tiny
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

print(f"Tiny GEMV (8x8): mean={np.mean(times):.3f}μs, min={np.min(times):.3f}μs")

# Check if cuBLAS switches kernel at certain sizes
print("\n" + "=" * 70)
print("Fine-grained sweep around 4000 (checking for kernel switch)")
print("=" * 70)

print(f"{'K':>6} | {'Latency':>10} | {'Diff from prev':>14}")
print("-" * 40)

prev = None
for K in range(3900, 4150, 10):
    result = benchmark_gemv_careful(4096, K, repeats=200)
    lat = result['cuda_mean']
    diff = f"{lat - prev:+.3f}μs" if prev else "---"
    print(f"{K:>6} | {lat:>8.3f}μs | {diff:>14}")
    prev = lat
