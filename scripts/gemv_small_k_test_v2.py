#!/usr/bin/env python3
"""GEMV staircase test - compare aligned vs unaligned at same scale."""

import torch
import numpy as np

def benchmark_gemv(N, K, repeats=500):
    """Benchmark GEMV: y = A @ x where A is (N, K), x is (K,)"""
    A = torch.randn(N, K, dtype=torch.float16, device='cuda')
    x = torch.randn(K, dtype=torch.float16, device='cuda')

    # Warmup
    for _ in range(50):
        y = A @ x
    torch.cuda.synchronize()

    # Timing
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        y = A @ x
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return np.mean(times), np.std(times)

print("=" * 70)
print("GEMV Staircase: Aligned vs Unaligned at Same Scale")
print("=" * 70)

# Test pairs: (unaligned, aligned) - same approximate size
test_pairs = [
    (327, 328),    # ~320 range
    (519, 520),    # ~512 range
    (1007, 1008),  # ~1000 range
    (1500, 1504),  # ~1500 range
    (1999, 2000),  # ~2000 range
    (2500, 2504),  # ~2500 range
    (2999, 3000),  # ~3000 range
    (3185, 3192),  # max ASVD rank range
]

print("\n[1] K Dimension (N=4096 fixed) - ASVD GEMV2: x @ A where A is (r, 4096)")
print(f"{'Unaligned K':>12} | {'Aligned K':>10} | {'Unalign lat':>12} | {'Align lat':>10} | {'Penalty':>8}")
print("-" * 70)

for k_unalign, k_align in test_pairs:
    mean_unalign, std_unalign = benchmark_gemv(4096, k_unalign)
    mean_align, std_align = benchmark_gemv(4096, k_align)
    penalty = (mean_unalign / mean_align - 1) * 100
    print(f"{k_unalign:>12} | {k_align:>10} | {mean_unalign*1000:>9.2f} μs | {mean_align*1000:>7.2f} μs | {penalty:>+7.1f}%")

print("\n[2] N Dimension (K=4096 fixed) - ASVD GEMV1: x @ B where B is (4096, r)")
print(f"{'Unaligned N':>12} | {'Aligned N':>10} | {'Unalign lat':>12} | {'Align lat':>10} | {'Penalty':>8}")
print("-" * 70)

for n_unalign, n_align in test_pairs:
    mean_unalign, std_unalign = benchmark_gemv(n_unalign, 4096)
    mean_align, std_align = benchmark_gemv(n_align, 4096)
    penalty = (mean_unalign / mean_align - 1) * 100
    print(f"{n_unalign:>12} | {n_align:>10} | {mean_unalign*1000:>9.2f} μs | {mean_align*1000:>7.2f} μs | {penalty:>+7.1f}%")

# Summary with actual ASVD rank distribution
print("\n" + "=" * 70)
print("Summary: Weighted penalty based on ASVD rank distribution")
print("=" * 70)
