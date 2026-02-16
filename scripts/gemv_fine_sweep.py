#!/usr/bin/env python3
"""GEMV fine-grained sweep: step=1 for each dimension."""

import torch
import numpy as np
import json
from pathlib import Path
import argparse

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
    parser = argparse.ArgumentParser(description="GEMV fine-grained sweep")
    parser.add_argument('--start', type=int, default=3800, help='Sweep start dimension')
    parser.add_argument('--end', type=int, default=4300, help='Sweep end dimension (inclusive)')
    parser.add_argument('--fixed', type=int, default=4096, help='Fixed dimension for the other axis')
    parser.add_argument('--repeats', type=int, default=50, help='Benchmark repeats per point')
    parser.add_argument('--output', type=str, default='results/gemv_fine_sweep.json', help='Output JSON path')
    args = parser.parse_args()

    print("=" * 70)
    print("GEMV Fine-grained Sweep (step=1)")
    print("=" * 70)

    results = {}

    # K sweep: [start, end], step=1, N=fixed
    n_points = args.end - args.start + 1
    print(f"\n[K Sweep] N={args.fixed}, K from {args.start} to {args.end} ({n_points} points)")
    K_results = []
    for K in range(args.start, args.end + 1):
        mean_ms, std_ms = benchmark_gemv(args.fixed, K, repeats=args.repeats)
        K_results.append({'K': K, 'mean_ms': mean_ms, 'std_ms': std_ms})
        if K % 50 == 0:
            print(f"  K={K}: {mean_ms*1000:.2f}μs")
    results['K_sweep'] = K_results

    # N sweep: [start, end], step=1, K=fixed
    print(f"\n[N Sweep] K={args.fixed}, N from {args.start} to {args.end} ({n_points} points)")
    N_results = []
    for N in range(args.start, args.end + 1):
        mean_ms, std_ms = benchmark_gemv(N, args.fixed, repeats=args.repeats)
        N_results.append({'N': N, 'mean_ms': mean_ms, 'std_ms': std_ms})
        if N % 50 == 0:
            print(f"  N={N}: {mean_ms*1000:.2f}μs")
    results['N_sweep'] = N_results

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
