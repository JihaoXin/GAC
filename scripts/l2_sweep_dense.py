#!/usr/bin/env python3
"""
Dense sweep of K values to get continuous data for L2 alignment plot.
"""

import torch
import triton
import triton.language as tl
import numpy as np
import json


@triton.jit
def row_load_kernel(
    ptr, out_ptr,
    N, K, stride,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_N
    row_idx = row_start + tl.arange(0, BLOCK_N)
    row_mask = row_idx < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K
        ptrs = ptr + row_idx[:, None] * stride + k_idx[None, :]
        mask = row_mask[:, None] & k_mask[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=1)

    tl.store(out_ptr + row_idx, acc.to(tl.float16), mask=row_mask)


def benchmark_k(n_rows, k, repeats=50):
    """Benchmark bandwidth for a given K."""
    data = torch.randn(n_rows, k, dtype=torch.float16, device='cuda')
    out = torch.empty(n_rows, dtype=torch.float16, device='cuda')

    grid = (triton.cdiv(n_rows, 32),)

    # Warmup
    for _ in range(10):
        row_load_kernel[grid](data, out, n_rows, k, k, BLOCK_N=32, BLOCK_K=64)
    torch.cuda.synchronize()

    # Timing
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        row_load_kernel[grid](data, out, n_rows, k, k, BLOCK_N=32, BLOCK_K=64)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mean_ms = np.mean(times)
    bytes_total = (n_rows * k + n_rows) * 2
    bandwidth = (bytes_total / 1e9) / (mean_ms / 1000)

    return bandwidth


def main():
    n_rows = 1024

    # Dense sweep from 4000 to 4100
    k_values = list(range(4000, 4101))

    results = []

    print("K dim | K%16 | Bandwidth (GB/s)")
    print("-" * 40)

    for k in k_values:
        bw = benchmark_k(n_rows, k)
        results.append({'K': k, 'K_mod_16': k % 16, 'bandwidth': bw})

        marker = "***" if k % 16 == 0 else ""
        print(f"{k:5d} | {k%16:4d} | {bw:6.1f} {marker}")

    # Save results
    with open('results/l2_dense_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} data points to results/l2_dense_sweep.json")


if __name__ == "__main__":
    main()
