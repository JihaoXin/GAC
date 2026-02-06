#!/usr/bin/env python3
"""
NCU profiling script to measure L2 cache sector efficiency.

Usage:
    # Run basic experiment (no NCU, just timing)
    python scripts/ncu_l2_profile.py --K 4096

    # Run with NCU profiling (requires sudo or ncu in path)
    ncu --set full --target-processes all -o l2_profile \
        python scripts/ncu_l2_profile.py --K 4096 --ncu-mode

Key NCU metrics for L2 sector analysis:
- lts__t_sectors_srcunit_tex_op_read.sum: Total L2 sectors read
- lts__t_requests_srcunit_tex_op_read.sum: Total L2 read requests
- lts__average_t_sectors_per_request_srcunit_tex_op_read.ratio: Sectors/request
- lts__t_sector_hit_rate.pct: L2 sector hit rate
- dram__bytes_read.sum: Total DRAM bytes read
- lts__t_bytes_equiv_l2_to_fb_read.sum: L2 → DRAM bytes
"""

import torch
import triton
import triton.language as tl
import argparse


@triton.jit
def row_load_kernel(
    ptr, out_ptr,
    N, K, stride,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Row-wise load kernel."""
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


def run_single_kernel(n_rows, row_width, block_n=32, block_k=64):
    """Run a single kernel invocation for NCU profiling."""
    data = torch.randn(n_rows, row_width, dtype=torch.float16, device='cuda')
    out = torch.empty(n_rows, dtype=torch.float16, device='cuda')

    grid = (triton.cdiv(n_rows, block_n),)

    # Single invocation for NCU
    row_load_kernel[grid](data, out, n_rows, row_width, row_width,
                          BLOCK_N=block_n, BLOCK_K=block_k)
    torch.cuda.synchronize()

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1024, help='Number of rows')
    parser.add_argument('--K', type=int, default=4096, help='Row width')
    parser.add_argument('--ncu-mode', action='store_true',
                        help='Single invocation mode for NCU profiling')
    parser.add_argument('--compare', action='store_true',
                        help='Compare aligned vs misaligned')
    args = parser.parse_args()

    print(f"Configuration: N={args.N}, K={args.K}")
    print(f"K % 8 = {args.K % 8}")
    print(f"K % 16 = {args.K % 16}")
    print(f"Row bytes = {args.K * 2}")
    print(f"Row bytes % 32 = {(args.K * 2) % 32}")
    print()

    if args.ncu_mode:
        print("Running single kernel invocation for NCU profiling...")
        run_single_kernel(args.N, args.K)
        print("Done. Check NCU output for L2 metrics.")
    elif args.compare:
        print("Comparing aligned (K=4096) vs misaligned (K=4095)...")
        print()

        # Aligned
        print("=== K=4096 (K%16=0, aligned) ===")
        run_single_kernel(args.N, 4096)
        print("Kernel executed.")

        # Misaligned
        print()
        print("=== K=4095 (K%16=15, misaligned) ===")
        run_single_kernel(args.N, 4095)
        print("Kernel executed.")

        print()
        print("To compare with NCU, run:")
        print("  ncu --set full -o aligned python scripts/ncu_l2_profile.py --K 4096 --ncu-mode")
        print("  ncu --set full -o misaligned python scripts/ncu_l2_profile.py --K 4095 --ncu-mode")
    else:
        # Regular timing mode
        import numpy as np

        data = torch.randn(args.N, args.K, dtype=torch.float16, device='cuda')
        out = torch.empty(args.N, dtype=torch.float16, device='cuda')
        grid = (triton.cdiv(args.N, 32),)

        # Warmup
        for _ in range(10):
            row_load_kernel[grid](data, out, args.N, args.K, args.K,
                                  BLOCK_N=32, BLOCK_K=64)
        torch.cuda.synchronize()

        # Timing
        times = []
        for _ in range(100):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            row_load_kernel[grid](data, out, args.N, args.K, args.K,
                                  BLOCK_N=32, BLOCK_K=64)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        mean_ms = np.mean(times)
        bytes_total = (args.N * args.K + args.N) * 2
        bandwidth = (bytes_total / 1e9) / (mean_ms / 1000)

        print(f"Mean latency: {mean_ms:.4f} ms")
        print(f"Bandwidth: {bandwidth:.1f} GB/s")


if __name__ == "__main__":
    main()
