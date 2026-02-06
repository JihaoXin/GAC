#!/usr/bin/env python3
"""
Debug LDG alignment effects - find the real constraint.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def ldg_row_kernel(
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


def benchmark(n_rows, row_width, block_n=32, block_k=64, warmup=50, repeats=200):
    data = torch.randn(n_rows, row_width, dtype=torch.float16, device='cuda')
    out = torch.empty(n_rows, dtype=torch.float16, device='cuda')

    grid = (triton.cdiv(n_rows, block_n),)

    for _ in range(warmup):
        ldg_row_kernel[grid](data, out, n_rows, row_width, row_width,
                            BLOCK_N=block_n, BLOCK_K=block_k)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        ldg_row_kernel[grid](data, out, n_rows, row_width, row_width,
                            BLOCK_N=block_n, BLOCK_K=block_k)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    import numpy as np
    mean_ms = np.mean(times)
    bytes_total = (n_rows * row_width + n_rows) * 2
    bandwidth = (bytes_total / 1e9) / (mean_ms / 1000)
    return mean_ms, bandwidth


print("=" * 70)
print("Experiment 1: Power-of-2 vs non-power-of-2 (all mod8=0)")
print("=" * 70)
n_rows = 1024
for k in [4096, 4088, 4080, 4072, 4064, 4032, 4000, 3968, 3072, 2048, 1024]:
    ms, bw = benchmark(n_rows, k)
    print(f"K={k:5d}  K%8={k%8}  K%64={k%64:2d}  K%128={k%128:3d}  "
          f"pow2={'Y' if (k & (k-1)) == 0 else 'N'}  "
          f"{bw:6.1f} GB/s  {ms:.4f} ms")

print("\n" + "=" * 70)
print("Experiment 2: Different BLOCK_K values")
print("=" * 70)
n_rows = 1024
for k in [4096, 4088]:
    print(f"\nK={k}:")
    for block_k in [32, 64, 128]:
        ms, bw = benchmark(n_rows, k, block_k=block_k)
        print(f"  BLOCK_K={block_k:3d}: {bw:6.1f} GB/s  (K%BLOCK_K = {k % block_k})")

print("\n" + "=" * 70)
print("Experiment 3: Around 4096 boundary")
print("=" * 70)
n_rows = 1024
for k in range(4088, 4105):
    ms, bw = benchmark(n_rows, k)
    marker = " <-- FAST" if bw > 100 else ""
    print(f"K={k}  K%8={k%8}  K%64={k%64:2d}  {bw:6.1f} GB/s{marker}")

print("\n" + "=" * 70)
print("Experiment 4: Other power-of-2 values")
print("=" * 70)
n_rows = 1024
for k in [1024, 1023, 2048, 2047, 4096, 4095, 8192, 8191]:
    ms, bw = benchmark(n_rows, k)
    print(f"K={k:5d}  K%8={k%8}  {bw:6.1f} GB/s")
