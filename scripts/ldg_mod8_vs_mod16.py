#!/usr/bin/env python3
"""Test K%8 vs K%16 alignment effects"""
import torch
import triton
import triton.language as tl
import numpy as np

@triton.jit
def ldg_row_kernel(ptr, out_ptr, N, K, stride, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    row_idx = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = row_idx < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K
        ptrs = ptr + row_idx[:, None] * stride + k_idx[None, :]
        vals = tl.load(ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        acc += tl.sum(vals, axis=1)
    tl.store(out_ptr + row_idx, acc.to(tl.float16), mask=row_mask)

def bench(n, k, warmup=30, repeats=100):
    data = torch.randn(n, k, dtype=torch.float16, device='cuda')
    out = torch.empty(n, dtype=torch.float16, device='cuda')
    grid = (triton.cdiv(n, 32),)
    for _ in range(warmup):
        ldg_row_kernel[grid](data, out, n, k, k, BLOCK_N=32, BLOCK_K=64)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        ldg_row_kernel[grid](data, out, n, k, k, BLOCK_N=32, BLOCK_K=64)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    ms = np.mean(times)
    bw = ((n * k + n) * 2 / 1e9) / (ms / 1000)
    return bw

print('=' * 50)
print('K%8 vs K%16 Alignment Effects')
print('=' * 50)
print(f'{"K":>5}  {"K%8":>3}  {"K%16":>4}  {"BW (GB/s)":>10}  Note')
print('-' * 50)
for k in range(4080, 4097):
    bw = bench(1024, k)
    note = ''
    if k % 16 == 0:
        note = 'K%16=0 (32B aligned)'
    elif k % 8 == 0:
        note = 'K%8=0 only (16B aligned)'
    print(f'{k:>5}  {k%8:>3}  {k%16:>4}  {bw:>10.1f}  {note}')

print('\n' + '=' * 50)
print('Summary by alignment class')
print('=' * 50)
# Group by alignment
mod16_0 = []
mod8_0_only = []
others = []

for k in range(4080, 4097):
    bw = bench(1024, k)
    if k % 16 == 0:
        mod16_0.append((k, bw))
    elif k % 8 == 0:
        mod8_0_only.append((k, bw))
    else:
        others.append((k, bw))

print(f'\nK % 16 = 0 (32-byte aligned rows):')
for k, bw in mod16_0:
    print(f'  K={k}: {bw:.1f} GB/s')
avg_16 = np.mean([bw for _, bw in mod16_0])
print(f'  Average: {avg_16:.1f} GB/s')

print(f'\nK % 8 = 0 but K % 16 != 0 (16-byte aligned only):')
for k, bw in mod8_0_only:
    print(f'  K={k}: {bw:.1f} GB/s')
if mod8_0_only:
    avg_8 = np.mean([bw for _, bw in mod8_0_only])
    print(f'  Average: {avg_8:.1f} GB/s')
    print(f'  Penalty vs K%16=0: {(avg_16 - avg_8) / avg_16 * 100:.1f}%')

print(f'\nK % 8 != 0 (misaligned):')
avg_other = np.mean([bw for _, bw in others])
print(f'  Average: {avg_other:.1f} GB/s')
print(f'  Penalty vs K%16=0: {(avg_16 - avg_other) / avg_16 * 100:.1f}%')
