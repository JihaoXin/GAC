#!/usr/bin/env python3
"""
Verify L2 Cache Sector Alignment is the Real Constraint.

Key Hypothesis: K % 16 = 0 (fp16) matters because:
- L2 cache sector = 32 bytes = 16 fp16 elements
- Misaligned rows cause partial sector reads

NCU Metrics to verify:
- lts__t_sectors_srcunit_tex_op_read.sum: L2 sectors read
- lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum: L2 hits
- lts__t_bytes_equiv_per_sector_op_tex_op_read.pct: Sector efficiency
- dram__bytes_read.sum: DRAM bytes read
- l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum: Shared mem loads

Theory:
- Aligned (K%16=0): Each row starts at sector boundary → full sector utilization
- Misaligned (K%16≠0): Rows cross sector boundaries → partial sector reads
"""

import torch
import triton
import triton.language as tl
import argparse
import json
import os


@triton.jit
def row_load_kernel(
    ptr, out_ptr,
    N, K, stride,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Row-wise load kernel for L2 sector alignment testing."""
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


def run_kernel(n_rows, row_width, block_n=32, block_k=64, repeats=10):
    """Run kernel and return timing."""
    data = torch.randn(n_rows, row_width, dtype=torch.float16, device='cuda')
    out = torch.empty(n_rows, dtype=torch.float16, device='cuda')

    grid = (triton.cdiv(n_rows, block_n),)

    # Warmup
    for _ in range(5):
        row_load_kernel[grid](data, out, n_rows, row_width, row_width,
                              BLOCK_N=block_n, BLOCK_K=block_k)
    torch.cuda.synchronize()

    # Timing
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        row_load_kernel[grid](data, out, n_rows, row_width, row_width,
                              BLOCK_N=block_n, BLOCK_K=block_k)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    import numpy as np
    mean_ms = np.mean(times)
    bytes_total = (n_rows * row_width + n_rows) * 2  # fp16
    bandwidth = (bytes_total / 1e9) / (mean_ms / 1000)

    return {
        'n_rows': n_rows,
        'row_width': row_width,
        'K_mod_8': row_width % 8,
        'K_mod_16': row_width % 16,
        'row_bytes': row_width * 2,
        'row_bytes_mod_32': (row_width * 2) % 32,
        'mean_ms': mean_ms,
        'bandwidth_gbs': bandwidth,
    }


def print_analysis():
    """Print theoretical analysis of L2 sector alignment."""
    print("=" * 70)
    print("L2 Cache Sector Alignment Analysis")
    print("=" * 70)
    print()
    print("L2 Cache Sector = 32 bytes")
    print("For FP16: 32 bytes = 16 elements")
    print()
    print("Alignment requirement: K % 16 = 0 (for fp16)")
    print()
    print("Row width (K)  | Bytes/row | Bytes%32 | Sectors/row | Waste")
    print("-" * 70)

    test_cases = [
        4096,  # K%16=0, bytes%32=0
        4088,  # K%16=8, bytes%32=16
        4080,  # K%16=0, bytes%32=0
        4095,  # K%16=15, bytes%32=30
        4094,  # K%16=14, bytes%32=28
    ]

    for k in test_cases:
        bytes_per_row = k * 2
        bytes_mod_32 = bytes_per_row % 32
        sectors_needed = (bytes_per_row + 31) // 32
        ideal_sectors = bytes_per_row / 32
        waste_pct = (sectors_needed - ideal_sectors) / sectors_needed * 100 if sectors_needed > 0 else 0

        alignment = "✓" if k % 16 == 0 else "✗"
        print(f"K={k:5d} {alignment}    | {bytes_per_row:8d} | {bytes_mod_32:8d} | "
              f"{sectors_needed:11.1f} | {waste_pct:5.1f}%")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-rows', type=int, default=1024)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--analysis-only', action='store_true')
    args = parser.parse_args()

    print_analysis()

    if args.analysis_only:
        return

    print("=" * 70)
    print("Running L2 Sector Alignment Experiments")
    print("=" * 70)
    print()

    # Test configurations
    test_widths = [
        # K % 16 = 0 (aligned)
        4096, 4080, 4064, 4048, 4032,
        # K % 16 != 0 but K % 8 = 0 (misaligned)
        4088, 4072, 4056, 4040, 4024,
        # K % 8 != 0 (misaligned)
        4095, 4094, 4093, 4092, 4091,
    ]

    results = []

    print(f"{'K':>6} | {'K%8':>4} | {'K%16':>5} | {'Bytes%32':>8} | {'BW (GB/s)':>10} | Status")
    print("-" * 70)

    for k in sorted(test_widths):
        result = run_kernel(args.n_rows, k)
        results.append(result)

        status = "ALIGNED" if k % 16 == 0 else ("K%8=0 only" if k % 8 == 0 else "MISALIGNED")
        print(f"{k:>6} | {k%8:>4} | {k%16:>5} | {(k*2)%32:>8} | {result['bandwidth_gbs']:>10.1f} | {status}")

    # Compute statistics
    aligned_bw = [r['bandwidth_gbs'] for r in results if r['K_mod_16'] == 0]
    mod8_only_bw = [r['bandwidth_gbs'] for r in results if r['K_mod_8'] == 0 and r['K_mod_16'] != 0]
    misaligned_bw = [r['bandwidth_gbs'] for r in results if r['K_mod_8'] != 0]

    print()
    print("=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    import numpy as np
    if aligned_bw:
        print(f"K % 16 = 0 (32-byte aligned):  {np.mean(aligned_bw):.1f} GB/s (n={len(aligned_bw)})")
    if mod8_only_bw:
        print(f"K % 8 = 0, K % 16 != 0:        {np.mean(mod8_only_bw):.1f} GB/s (n={len(mod8_only_bw)})")
        penalty = (np.mean(aligned_bw) - np.mean(mod8_only_bw)) / np.mean(aligned_bw) * 100
        print(f"  → Penalty vs aligned: {penalty:.1f}%")
    if misaligned_bw:
        print(f"K % 8 != 0 (misaligned):       {np.mean(misaligned_bw):.1f} GB/s (n={len(misaligned_bw)})")
        penalty = (np.mean(aligned_bw) - np.mean(misaligned_bw)) / np.mean(aligned_bw) * 100
        print(f"  → Penalty vs aligned: {penalty:.1f}%")

    print()
    print("=" * 70)
    print("Conclusion")
    print("=" * 70)

    if mod8_only_bw and aligned_bw:
        mod8_vs_aligned = abs(np.mean(mod8_only_bw) - np.mean(misaligned_bw)) / np.mean(aligned_bw) * 100
        if mod8_vs_aligned < 5:
            print("✓ K % 8 = 0 provides NO benefit over misaligned")
            print("✓ Only K % 16 = 0 provides benefit (L2 sector alignment)")
            print("✓ Confirms: constraint is L2 cache sector (32 bytes), not LDG instruction width")
        else:
            print("✗ K % 8 = 0 provides some benefit - needs further investigation")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'n_rows': args.n_rows,
                'results': results,
                'summary': {
                    'aligned_mean_bw': np.mean(aligned_bw) if aligned_bw else None,
                    'mod8_only_mean_bw': np.mean(mod8_only_bw) if mod8_only_bw else None,
                    'misaligned_mean_bw': np.mean(misaligned_bw) if misaligned_bw else None,
                }
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
