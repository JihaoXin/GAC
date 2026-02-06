#!/usr/bin/env python3
"""GEMV fine-grained sweep: step=1 for each dimension."""

import torch
import numpy as np
import json
from pathlib import Path

def benchmark_gemv(N, K, repeats=50):
    """Benchmark GEMV: y = A @ x"""
    A = torch.randn(N, K, dtype=torch.float16, device='cuda')
    x = torch.randn(K, dtype=torch.float16, device='cuda')

    # Warmup
    for _ in range(10):
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


def main():
    print("=" * 70)
    print("GEMV Fine-grained Sweep (step=1)")
    print("=" * 70)

    results = {}

    # K sweep: 3800 to 4300, step=1, N=4096 fixed
    print("\n[K Sweep] N=4096, K from 3800 to 4300 (501 points)")
    K_results = []
    for K in range(3800, 4301):
        mean_ms, std_ms = benchmark_gemv(4096, K, repeats=50)
        K_results.append({'K': K, 'mean_ms': mean_ms, 'std_ms': std_ms})
        if K % 50 == 0:
            print(f"  K={K}: {mean_ms*1000:.2f}μs")
    results['K_sweep'] = K_results

    # N sweep: 3800 to 4300, step=1, K=4096 fixed
    print("\n[N Sweep] K=4096, N from 3800 to 4300 (501 points)")
    N_results = []
    for N in range(3800, 4301):
        mean_ms, std_ms = benchmark_gemv(N, 4096, repeats=50)
        N_results.append({'N': N, 'mean_ms': mean_ms, 'std_ms': std_ms})
        if N % 50 == 0:
            print(f"  N={N}: {mean_ms*1000:.2f}μs")
    results['N_sweep'] = N_results

    # Save
    Path('results').mkdir(exist_ok=True)
    with open('results/gemv_fine_sweep.json', 'w') as f:
        json.dump(results, f)
    print(f"\nSaved to results/gemv_fine_sweep.json")


if __name__ == "__main__":
    main()
