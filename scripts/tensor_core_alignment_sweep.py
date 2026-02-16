#!/usr/bin/env python3
"""
Tensor Core MMA alignment sweep.
Test m16n8k16 alignment requirements for GEMM.

Tensor Core constraints:
- K % 16 = 0 for m16n8k16
- M % 16 = 0 for efficient tiling
- N % 8 = 0 for efficient tiling
"""

import torch
import numpy as np
import json
import argparse


def benchmark_gemm_cublas(M, N, K, repeats=100):
    """Benchmark GEMM: C = A @ B, A:(M,K), B:(K,N), C:(M,N)"""
    A = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(K, N, dtype=torch.float16, device='cuda')

    # Warmup
    for _ in range(20):
        C = A @ B
    torch.cuda.synchronize()

    # Timing
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        C = A @ B
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mean_ms = np.mean(times)
    std_ms = np.std(times)

    # Compute TFLOPS
    flops = 2 * M * N * K
    tflops = (flops / 1e12) / (mean_ms / 1000)

    return {
        'M': M,
        'N': N,
        'K': K,
        'mean_ms': mean_ms,
        'std_ms': std_ms,
        'tflops': tflops,
        'M_mod_16': M % 16,
        'N_mod_8': N % 8,
        'K_mod_16': K % 16,
    }


def main():
    parser = argparse.ArgumentParser(description="Tensor Core MMA alignment sweep")
    parser.add_argument('--start', type=int, default=4000, help='Sweep start dimension')
    parser.add_argument('--end', type=int, default=4100, help='Sweep end dimension (inclusive)')
    parser.add_argument('--m-fixed', type=int, default=512, help='Fixed M for K/N sweeps')
    parser.add_argument('--n-fixed', type=int, default=4096, help='Fixed N for K/M sweeps')
    parser.add_argument('--k-fixed', type=int, default=4096, help='Fixed K for N/M sweeps')
    parser.add_argument('--m-start', type=int, default=500, help='M sweep start')
    parser.add_argument('--m-end', type=int, default=600, help='M sweep end (inclusive)')
    parser.add_argument('--repeats', type=int, default=100, help='Repeats per point')
    parser.add_argument('--output', type=str, default='results/tensor_core_alignment_sweep.json', help='Output JSON path')
    args = parser.parse_args()

    results = []

    # Fixed dimensions for sweep
    M_fixed = args.m_fixed
    N_fixed = args.n_fixed

    # K sweep (most important for Tensor Core)
    print("=" * 70)
    print(f"GEMM K Sweep (M={M_fixed}, N={N_fixed} fixed)")
    print("=" * 70)

    K_values = list(range(args.start, args.end + 1))

    print(f"{'K':>6} | {'K%16':>5} | {'Latency':>10} | {'TFLOPS':>8} | Note")
    print("-" * 60)

    for K in K_values:
        result = benchmark_gemm_cublas(M_fixed, N_fixed, K, repeats=args.repeats)
        result['sweep'] = 'K_sweep'
        results.append(result)

        aligned = "ALIGNED" if K % 16 == 0 else ""
        print(f"{K:>6} | {K%16:>5} | {result['mean_ms']:>8.4f}ms | {result['tflops']:>8.1f} | {aligned}")

    # N sweep
    print()
    print("=" * 70)
    print(f"GEMM N Sweep (M={M_fixed}, K=4096 fixed)")
    print("=" * 70)

    K_fixed = args.k_fixed
    N_values = list(range(args.start, args.end + 1))

    print(f"{'N':>6} | {'N%8':>4} | {'Latency':>10} | {'TFLOPS':>8} | Note")
    print("-" * 60)

    for N in N_values:
        result = benchmark_gemm_cublas(M_fixed, N, K_fixed, repeats=args.repeats)
        result['sweep'] = 'N_sweep'
        results.append(result)

        aligned = "ALIGNED" if N % 8 == 0 else ""
        print(f"{N:>6} | {N%8:>4} | {result['mean_ms']:>8.4f}ms | {result['tflops']:>8.1f} | {aligned}")

    # M sweep
    print()
    print("=" * 70)
    print(f"GEMM M Sweep (N={N_fixed}, K=4096 fixed)")
    print("=" * 70)

    M_values = list(range(args.m_start, args.m_end + 1))

    print(f"{'M':>6} | {'M%16':>5} | {'Latency':>10} | {'TFLOPS':>8} | Note")
    print("-" * 60)

    for M in M_values:
        result = benchmark_gemm_cublas(M, N_fixed, K_fixed, repeats=args.repeats)
        result['sweep'] = 'M_sweep'
        results.append(result)

        aligned = "ALIGNED" if M % 16 == 0 else ""
        print(f"{M:>6} | {M%16:>5} | {result['mean_ms']:>8.4f}ms | {result['tflops']:>8.1f} | {aligned}")

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    for sweep_name, mod_key, mod_val in [('K_sweep', 'K_mod_16', 16),
                                          ('N_sweep', 'N_mod_8', 8),
                                          ('M_sweep', 'M_mod_16', 16)]:
        sweep_results = [r for r in results if r['sweep'] == sweep_name]
        aligned = [r for r in sweep_results if r[mod_key] == 0]
        misaligned = [r for r in sweep_results if r[mod_key] != 0]

        if aligned and misaligned:
            aligned_tflops = np.mean([r['tflops'] for r in aligned])
            misaligned_tflops = np.mean([r['tflops'] for r in misaligned])
            penalty = (aligned_tflops - misaligned_tflops) / aligned_tflops * 100
            print(f"{sweep_name}: Aligned={aligned_tflops:.1f} TFLOPS, Misaligned={misaligned_tflops:.1f} TFLOPS, Penalty={penalty:.1f}%")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
