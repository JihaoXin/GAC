#!/usr/bin/env python3
"""
GEMV benchmark using real dimensions from ASVD compressed model.
Tests whether irregular ranks cause decode latency degradation.
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path


def get_asvd_ranks(hidden_size=4096, intermediate_size=14336,
                   num_kv_heads=8, head_dim=128, param_ratio=0.85, rank_align=1):
    """
    Calculate ASVD ranks for Llama-3-8B layers.

    For SVD decomposition W ≈ A @ B:
    - W: (out, in), params = out * in
    - A: (out, rank), B: (rank, in), params = rank * (out + in)
    - param_ratio = rank * (out + in) / (out * in)
    - rank = param_ratio * out * in / (out + in)
    """
    layers = {}

    # Attention projections
    # q_proj: (hidden, hidden)
    out, inp = hidden_size, hidden_size
    rank = param_ratio * out * inp / (out + inp)
    rank = int(rank // rank_align) * rank_align if rank_align > 1 else int(rank)
    layers['q_proj'] = {'out': out, 'in': inp, 'rank': rank}

    # k_proj, v_proj: (num_kv_heads * head_dim, hidden)
    out, inp = num_kv_heads * head_dim, hidden_size
    rank = param_ratio * out * inp / (out + inp)
    rank = int(rank // rank_align) * rank_align if rank_align > 1 else int(rank)
    layers['k_proj'] = {'out': out, 'in': inp, 'rank': rank}
    layers['v_proj'] = {'out': out, 'in': inp, 'rank': rank}

    # o_proj: (hidden, hidden)
    out, inp = hidden_size, hidden_size
    rank = param_ratio * out * inp / (out + inp)
    rank = int(rank // rank_align) * rank_align if rank_align > 1 else int(rank)
    layers['o_proj'] = {'out': out, 'in': inp, 'rank': rank}

    # MLP projections
    # gate_proj, up_proj: (intermediate, hidden)
    out, inp = intermediate_size, hidden_size
    rank = param_ratio * out * inp / (out + inp)
    rank = int(rank // rank_align) * rank_align if rank_align > 1 else int(rank)
    layers['gate_proj'] = {'out': out, 'in': inp, 'rank': rank}
    layers['up_proj'] = {'out': out, 'in': inp, 'rank': rank}

    # down_proj: (hidden, intermediate)
    out, inp = hidden_size, intermediate_size
    rank = param_ratio * out * inp / (out + inp)
    rank = int(rank // rank_align) * rank_align if rank_align > 1 else int(rank)
    layers['down_proj'] = {'out': out, 'in': inp, 'rank': rank}

    return layers


def benchmark_gemv(N, K, repeats=100):
    """
    Benchmark GEMV: y = A @ x
    A: (N, K), x: (K,), y: (N,)
    """
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
        'mean_ms': mean_ms,
        'std_ms': std_ms,
        'bandwidth_gbs': bandwidth_gbs,
    }


def benchmark_svd_layer(out_dim, in_dim, rank, repeats=100):
    """
    Benchmark SVD layer decode: x @ B @ A
    B: (rank, in_dim), A: (out_dim, rank)
    For decode, x is (1, in_dim)

    Operation 1: x @ B.T -> (1, rank)  [actually x @ B where B stored as (rank, in_dim)]
    Operation 2: result @ A.T -> (1, out_dim)
    """
    # ASVD stores: A (out, rank), B (rank, in)
    # Decode: x (1, in) @ B.T (in, rank) = (1, rank)
    #         then (1, rank) @ A.T (rank, out) = (1, out)

    A = torch.randn(out_dim, rank, dtype=torch.float16, device='cuda')
    B = torch.randn(rank, in_dim, dtype=torch.float16, device='cuda')
    x = torch.randn(1, in_dim, dtype=torch.float16, device='cuda')

    # Warmup
    for _ in range(20):
        tmp = x @ B.T  # (1, rank)
        y = tmp @ A.T  # (1, out)
    torch.cuda.synchronize()

    # Timing
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        tmp = x @ B.T
        y = tmp @ A.T
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mean_ms = np.mean(times)
    std_ms = np.std(times)

    # Compute bandwidth for both operations
    # Op1: B (rank * in) + x (in) = (rank * in + in) * 2 bytes
    # Op2: A (out * rank) + tmp (rank) = (out * rank + rank) * 2 bytes
    bytes_total = ((rank * in_dim + in_dim) + (out_dim * rank + rank) + (rank + out_dim)) * 2
    bandwidth_gbs = (bytes_total / 1e9) / (mean_ms / 1000)

    return {
        'mean_ms': mean_ms,
        'std_ms': std_ms,
        'bandwidth_gbs': bandwidth_gbs,
    }


def benchmark_original_layer(out_dim, in_dim, repeats=100):
    """Benchmark original dense layer: x @ W.T"""
    W = torch.randn(out_dim, in_dim, dtype=torch.float16, device='cuda')
    x = torch.randn(1, in_dim, dtype=torch.float16, device='cuda')

    # Warmup
    for _ in range(20):
        y = x @ W.T
    torch.cuda.synchronize()

    # Timing
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        y = x @ W.T
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mean_ms = np.mean(times)
    std_ms = np.std(times)

    bytes_total = (out_dim * in_dim + in_dim + out_dim) * 2
    bandwidth_gbs = (bytes_total / 1e9) / (mean_ms / 1000)

    return {
        'mean_ms': mean_ms,
        'std_ms': std_ms,
        'bandwidth_gbs': bandwidth_gbs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_ratio', type=float, default=0.85)
    parser.add_argument('--repeats', type=int, default=200)
    parser.add_argument('--output', type=str, default='results/gemv_real_dims.json')
    args = parser.parse_args()

    print("=" * 80)
    print("GEMV Benchmark with Real ASVD Dimensions (Llama-3-8B)")
    print("=" * 80)

    # Get dimensions for unaligned (rank_align=1) and aligned (rank_align=8)
    unaligned = get_asvd_ranks(param_ratio=args.param_ratio, rank_align=1)
    aligned = get_asvd_ranks(param_ratio=args.param_ratio, rank_align=8)

    print(f"\nParam ratio: {args.param_ratio}")
    print("\nCalculated ranks:")
    print(f"{'Layer':<12} | {'Out':>6} | {'In':>6} | {'Rank(1)':>8} | {'Rank(8)':>8} | {'Aligned?':>8}")
    print("-" * 70)
    for name in unaligned:
        u = unaligned[name]
        a = aligned[name]
        is_aligned = "YES" if u['rank'] % 8 == 0 else "NO"
        print(f"{name:<12} | {u['out']:>6} | {u['in']:>6} | {u['rank']:>8} | {a['rank']:>8} | {is_aligned:>8}")

    results = {
        'param_ratio': args.param_ratio,
        'layers': {},
    }

    print("\n" + "=" * 80)
    print("Benchmarking decode latency per layer")
    print("=" * 80)
    print(f"\n{'Layer':<12} | {'Original':>10} | {'SVD(r=1)':>10} | {'SVD(r=8)':>10} | {'Overhead(1)':>12} | {'Overhead(8)':>12}")
    print("-" * 90)

    for name in unaligned:
        u = unaligned[name]
        a = aligned[name]

        # Original layer
        orig = benchmark_original_layer(u['out'], u['in'], args.repeats)

        # SVD unaligned
        svd_u = benchmark_svd_layer(u['out'], u['in'], u['rank'], args.repeats)

        # SVD aligned
        svd_a = benchmark_svd_layer(a['out'], a['in'], a['rank'], args.repeats)

        overhead_u = (svd_u['mean_ms'] - orig['mean_ms']) / orig['mean_ms'] * 100
        overhead_a = (svd_a['mean_ms'] - orig['mean_ms']) / orig['mean_ms'] * 100

        print(f"{name:<12} | {orig['mean_ms']:>8.4f}ms | {svd_u['mean_ms']:>8.4f}ms | {svd_a['mean_ms']:>8.4f}ms | {overhead_u:>+10.1f}% | {overhead_a:>+10.1f}%")

        results['layers'][name] = {
            'dims': {'out': u['out'], 'in': u['in']},
            'rank_unaligned': u['rank'],
            'rank_aligned': a['rank'],
            'original_ms': orig['mean_ms'],
            'svd_unaligned_ms': svd_u['mean_ms'],
            'svd_aligned_ms': svd_a['mean_ms'],
            'overhead_unaligned_pct': overhead_u,
            'overhead_aligned_pct': overhead_a,
        }

    # Summary
    print("\n" + "=" * 80)
    print("Summary (total for one transformer layer)")
    print("=" * 80)

    # Sum up all layers for one transformer block
    total_orig = sum(r['original_ms'] for r in results['layers'].values())
    total_svd_u = sum(r['svd_unaligned_ms'] for r in results['layers'].values())
    total_svd_a = sum(r['svd_aligned_ms'] for r in results['layers'].values())

    print(f"Original dense:     {total_orig:.4f} ms")
    print(f"SVD unaligned:      {total_svd_u:.4f} ms ({(total_svd_u/total_orig - 1)*100:+.1f}%)")
    print(f"SVD aligned (r=8):  {total_svd_a:.4f} ms ({(total_svd_a/total_orig - 1)*100:+.1f}%)")

    results['summary'] = {
        'total_original_ms': total_orig,
        'total_svd_unaligned_ms': total_svd_u,
        'total_svd_aligned_ms': total_svd_a,
    }

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
