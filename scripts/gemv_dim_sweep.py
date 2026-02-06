#!/usr/bin/env python3
"""
GEMV dimension sweep: y = A @ x where A is (N, K) and x is (K,)
M=1 for decode, so we only sweep K and N.
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path


def benchmark_gemv(N, K, repeats=100):
    """Benchmark GEMV: y = A @ x, A: (N, K), x: (K,)"""
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

    # Bandwidth
    bytes_read = (N * K + K) * 2
    bytes_write = N * 2
    bandwidth_gbs = ((bytes_read + bytes_write) / 1e9) / (mean_ms / 1000)

    return mean_ms, std_ms, bandwidth_gbs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--output', type=str, default='results/gemv_dim_sweep.json')
    args = parser.parse_args()

    print("=" * 80)
    print("GEMV Dimension Sweep")
    print("=" * 80)

    results = {'K_sweep': [], 'N_sweep': []}

    # ========================================
    # K sweep (N fixed at 4096)
    # ========================================
    N_fixed = 4096
    K_values = list(range(256, 4352, 64))  # 256 to 4288 step 64

    print(f"\n[K Sweep] N={N_fixed} fixed")
    print(f"{'K':>6} | {'K%8':>4} | {'K%16':>5} | {'Latency (ms)':>12} | {'BW (GB/s)':>10}")
    print("-" * 55)

    for K in K_values:
        mean_ms, std_ms, bw = benchmark_gemv(N_fixed, K, args.repeats)
        results['K_sweep'].append({
            'N': N_fixed, 'K': K,
            'mean_ms': mean_ms, 'std_ms': std_ms,
            'bandwidth_gbs': bw,
            'K_mod_8': K % 8, 'K_mod_16': K % 16,
        })
        mark = "*" if K % 16 != 0 else ""
        print(f"{K:>6} | {K%8:>4} | {K%16:>5} | {mean_ms:>10.4f}ms | {bw:>10.1f} {mark}")

    # ========================================
    # N sweep (K fixed at 4096)
    # ========================================
    K_fixed = 4096
    N_values = list(range(256, 4352, 64))

    print(f"\n[N Sweep] K={K_fixed} fixed")
    print(f"{'N':>6} | {'N%8':>4} | {'N%16':>5} | {'Latency (ms)':>12} | {'BW (GB/s)':>10}")
    print("-" * 55)

    for N in N_values:
        mean_ms, std_ms, bw = benchmark_gemv(N, K_fixed, args.repeats)
        results['N_sweep'].append({
            'N': N, 'K': K_fixed,
            'mean_ms': mean_ms, 'std_ms': std_ms,
            'bandwidth_gbs': bw,
            'N_mod_8': N % 8, 'N_mod_16': N % 16,
        })
        mark = "*" if N % 16 != 0 else ""
        print(f"{N:>6} | {N%8:>4} | {N%16:>5} | {mean_ms:>10.4f}ms | {bw:>10.1f} {mark}")

    # ========================================
    # Fine-grained sweep around 4096
    # ========================================
    print(f"\n[Fine K Sweep] N=4096, K from 4000 to 4100")
    print(f"{'K':>6} | {'K%8':>4} | {'K%16':>5} | {'Latency (ms)':>12} | {'BW (GB/s)':>10}")
    print("-" * 55)

    results['K_fine_sweep'] = []
    for K in range(4000, 4101):
        mean_ms, std_ms, bw = benchmark_gemv(4096, K, args.repeats)
        results['K_fine_sweep'].append({
            'N': 4096, 'K': K,
            'mean_ms': mean_ms, 'std_ms': std_ms,
            'bandwidth_gbs': bw,
        })
        if K % 16 == 0 or K == 4000 or K == 4100:
            print(f"{K:>6} | {K%8:>4} | {K%16:>5} | {mean_ms:>10.4f}ms | {bw:>10.1f}")

    # ========================================
    # Analysis
    # ========================================
    print("\n" + "=" * 80)
    print("Analysis")
    print("=" * 80)

    # K sweep linearity check
    K_data = results['K_sweep']
    K_vals = np.array([d['K'] for d in K_data])
    K_times = np.array([d['mean_ms'] for d in K_data])

    # Linear fit
    coeffs = np.polyfit(K_vals, K_times, 1)
    K_pred = np.polyval(coeffs, K_vals)
    r2 = 1 - np.sum((K_times - K_pred)**2) / np.sum((K_times - np.mean(K_times))**2)

    print(f"\nK sweep: slope = {coeffs[0]*1e6:.4f} ns/dim, intercept = {coeffs[1]*1000:.2f} us, R² = {r2:.4f}")
    print(f"  -> Latency is {'linear' if r2 > 0.99 else 'NOT purely linear'} with K dimension")

    # N sweep linearity check
    N_data = results['N_sweep']
    N_vals = np.array([d['N'] for d in N_data])
    N_times = np.array([d['mean_ms'] for d in N_data])

    coeffs_n = np.polyfit(N_vals, N_times, 1)
    N_pred = np.polyval(coeffs_n, N_vals)
    r2_n = 1 - np.sum((N_times - N_pred)**2) / np.sum((N_times - np.mean(N_times))**2)

    print(f"\nN sweep: slope = {coeffs_n[0]*1e6:.4f} ns/dim, intercept = {coeffs_n[1]*1000:.2f} us, R² = {r2_n:.4f}")
    print(f"  -> Latency is {'linear' if r2_n > 0.99 else 'NOT purely linear'} with N dimension")

    # Check for stair pattern in fine sweep
    fine_data = results['K_fine_sweep']
    fine_times = [d['mean_ms'] for d in fine_data]
    fine_std = np.std(fine_times)
    fine_mean = np.mean(fine_times)
    cv = fine_std / fine_mean * 100

    print(f"\nFine K sweep (4000-4100): mean={fine_mean:.4f}ms, std={fine_std:.4f}ms, CV={cv:.2f}%")
    print(f"  -> {'Stair pattern detected' if cv > 5 else 'No significant stair pattern'}")

    # Save results
    results['analysis'] = {
        'K_sweep_slope_ns_per_dim': coeffs[0] * 1e6,
        'K_sweep_r2': r2,
        'N_sweep_slope_ns_per_dim': coeffs_n[0] * 1e6,
        'N_sweep_r2': r2_n,
        'fine_sweep_cv_pct': cv,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
