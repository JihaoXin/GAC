#!/usr/bin/env python3
"""Quick test: GEMV staircase for small K (ASVD rank range)."""

import torch
import numpy as np

def benchmark_gemv(N, K, repeats=100):
    """Benchmark GEMV: y = A @ x where A is (N, K), x is (K,)"""
    A = torch.randn(N, K, dtype=torch.float16, device='cuda')
    x = torch.randn(K, dtype=torch.float16, device='cuda')

    # Warmup
    for _ in range(20):
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

print("=" * 60)
print("GEMV Staircase Test for Small K (ASVD rank range)")
print("=" * 60)

# Test K values in ASVD rank range
print("\n[1] K sweep (N=4096 fixed) - ASVD GEMV2 scenario")
print(f"{'K':>6} | {'K%8':>4} | {'Latency':>12} | {'Penalty':>8}")
print("-" * 45)

baseline_latencies = {}
for K in [320, 327, 328, 512, 519, 520, 1000, 1007, 1008, 2000, 2003, 2008, 3000, 3001, 3008]:
    mean_ms, _ = benchmark_gemv(4096, K, repeats=200)
    K_aligned = (K // 8) * 8
    if K % 8 == 0:
        baseline_latencies[K] = mean_ms
        print(f"{K:>6} | {K%8:>4} | {mean_ms*1000:>8.2f} μs | baseline")
    else:
        # Find nearest aligned baseline
        base = baseline_latencies.get(K_aligned, mean_ms)
        penalty = (mean_ms / base - 1) * 100
        print(f"{K:>6} | {K%8:>4} | {mean_ms*1000:>8.2f} μs | {penalty:>+6.1f}%")

print("\n[2] N sweep (K=4096 fixed) - ASVD GEMV1 scenario")
print(f"{'N':>6} | {'N%8':>4} | {'Latency':>12} | {'Penalty':>8}")
print("-" * 45)

baseline_latencies = {}
for N in [320, 327, 328, 512, 519, 520, 1000, 1007, 1008, 2000, 2003, 2008, 3000, 3001, 3008]:
    mean_ms, _ = benchmark_gemv(N, 4096, repeats=200)
    N_aligned = (N // 8) * 8
    if N % 8 == 0:
        baseline_latencies[N] = mean_ms
        print(f"{N:>6} | {N%8:>4} | {mean_ms*1000:>8.2f} μs | baseline")
    else:
        base = baseline_latencies.get(N_aligned, mean_ms)
        penalty = (mean_ms / base - 1) * 100
        print(f"{N:>6} | {N%8:>4} | {mean_ms*1000:>8.2f} μs | {penalty:>+6.1f}%")
