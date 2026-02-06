#!/usr/bin/env python3
"""
GeMV K and N alignment sweep using cuBLAS.
Similar to GEMM sweep to compare alignment effects.
"""

import torch
import numpy as np
import json
import time


def benchmark_gemv_cublas(M, N, K, repeats=100):
    """
    Benchmark GeMV: y = A @ x
    A: (M, K), x: (K,), y: (M,)
    For GeMV, M=1 (single output row), so it's (1, K) @ (K,) -> (1,)
    But we use (N, K) @ (K,) -> (N,) to test N dimension effect
    """
    # A: (N, K), x: (K,)
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

    mean_ms = np.mean(times)
    std_ms = np.std(times)

    # Compute bandwidth
    bytes_read = (N * K + K) * 2  # A + x in fp16
    bytes_write = N * 2  # y in fp16
    total_bytes = bytes_read + bytes_write
    bandwidth_gbs = (total_bytes / 1e9) / (mean_ms / 1000)

    return {
        'N': N,
        'K': K,
        'mean_ms': mean_ms,
        'std_ms': std_ms,
        'bandwidth_gbs': bandwidth_gbs,
        'N_mod_8': N % 8,
        'K_mod_8': K % 8,
        'N_mod_16': N % 16,
        'K_mod_16': K % 16,
    }


def main():
    results = []

    # Test configurations similar to GEMM sweep
    # Vary K around 4096
    print("=" * 70)
    print("GeMV K Sweep (N=1024 fixed)")
    print("=" * 70)

    N_fixed = 1024
    K_values = list(range(3900, 4101))  # K from 3900 to 4100 (201 points)

    print(f"{'K':>6} | {'K%16':>5} | {'Latency':>10} | {'BW (GB/s)':>10} | Note")
    print("-" * 60)

    for K in K_values:
        result = benchmark_gemv_cublas(1, N_fixed, K)
        result['sweep'] = 'K_sweep'
        results.append(result)

        aligned = "ALIGNED" if K % 16 == 0 else ""
        print(f"{K:>6} | {K%16:>5} | {result['mean_ms']:>8.4f}ms | {result['bandwidth_gbs']:>10.1f} | {aligned}")

    # Vary N around 1024
    print()
    print("=" * 70)
    print("GeMV N Sweep (K=4096 fixed)")
    print("=" * 70)

    K_fixed = 4096
    N_values = list(range(900, 1101))  # N from 900 to 1100 (201 points)

    print(f"{'N':>6} | {'N%16':>5} | {'Latency':>10} | {'BW (GB/s)':>10} | Note")
    print("-" * 60)

    for N in N_values:
        result = benchmark_gemv_cublas(1, N, K_fixed)
        result['sweep'] = 'N_sweep'
        results.append(result)

        aligned = "ALIGNED" if N % 16 == 0 else ""
        print(f"{N:>6} | {N%16:>5} | {result['mean_ms']:>8.4f}ms | {result['bandwidth_gbs']:>10.1f} | {aligned}")

    # Summary statistics
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    # K sweep summary
    k_results = [r for r in results if r['sweep'] == 'K_sweep']
    k_aligned = [r for r in k_results if r['K_mod_16'] == 0]
    k_misaligned = [r for r in k_results if r['K_mod_16'] != 0]

    if k_aligned and k_misaligned:
        k_aligned_bw = np.mean([r['bandwidth_gbs'] for r in k_aligned])
        k_misaligned_bw = np.mean([r['bandwidth_gbs'] for r in k_misaligned])
        k_penalty = (k_aligned_bw - k_misaligned_bw) / k_aligned_bw * 100
        print(f"K sweep: Aligned={k_aligned_bw:.1f} GB/s, Misaligned={k_misaligned_bw:.1f} GB/s, Penalty={k_penalty:.1f}%")

    # N sweep summary
    n_results = [r for r in results if r['sweep'] == 'N_sweep']
    n_aligned = [r for r in n_results if r['N_mod_16'] == 0]
    n_misaligned = [r for r in n_results if r['N_mod_16'] != 0]

    if n_aligned and n_misaligned:
        n_aligned_bw = np.mean([r['bandwidth_gbs'] for r in n_aligned])
        n_misaligned_bw = np.mean([r['bandwidth_gbs'] for r in n_misaligned])
        n_penalty = (n_aligned_bw - n_misaligned_bw) / n_aligned_bw * 100
        print(f"N sweep: Aligned={n_aligned_bw:.1f} GB/s, Misaligned={n_misaligned_bw:.1f} GB/s, Penalty={n_penalty:.1f}%")

    # Save results
    with open('results/gemv_alignment_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to results/gemv_alignment_sweep.json")


if __name__ == "__main__":
    main()
