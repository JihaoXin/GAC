#!/usr/bin/env python3
"""
NCU Profiling Wrapper for Triton GeMV Kernel

Usage:
    # Basic profiling (all metrics)
    ncu --set full -o triton_gemv_profile python scripts/ncu_profile_triton_gemv.py --N 1024 --K 4096

    # Profile specific metrics for LDG analysis
    ncu --metrics \
        sm__sass_inst_executed_op_ld_pred_on_any.sum,\
        sm__sass_inst_executed_op_ld_pred_on_any.sum.per_second,\
        l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
        lts__t_sectors_srcunit_tex_op_read.sum,\
        dram__throughput.avg.pct_of_peak_sustained_elapsed \
        -o triton_gemv_ldg python scripts/ncu_profile_triton_gemv.py --N 1024 --K 4096

    # Compare aligned vs misaligned
    ncu --set full -o aligned python scripts/ncu_profile_triton_gemv.py --N 1024 --K 4096
    ncu --set full -o misaligned python scripts/ncu_profile_triton_gemv.py --N 1023 --K 4096

Key NCU Metrics to Analyze:
    Memory Load Instructions:
    - sm__sass_inst_executed_op_ldg_32.sum     # LDG.32 (4 bytes, misaligned)
    - sm__sass_inst_executed_op_ldg_64.sum     # LDG.64 (8 bytes)
    - sm__sass_inst_executed_op_ldg_128.sum    # LDG.128 (16 bytes, optimal)

    Memory Throughput:
    - dram__throughput.avg.pct_of_peak_sustained_elapsed  # HBM utilization %
    - lts__t_sectors_srcunit_tex_op_read.sum   # L2 cache read sectors
    - l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum  # L1 global load sectors

    Compute vs Memory Bound:
    - sm__throughput.avg.pct_of_peak_sustained_elapsed    # SM utilization
    - gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
"""

import argparse
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def gemv_kernel(
        W_ptr, x_ptr, y_ptr,
        N, K,
        stride_wn, stride_wk,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        GeMV kernel: y[n] = sum_k(W[n,k] * x[k])
        """
        pid = tl.program_id(0)
        n_offset = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offset < N

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_offset = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offset < K

            x_vals = tl.load(x_ptr + k_offset, mask=k_mask, other=0.0)

            w_ptrs = W_ptr + n_offset[:, None] * stride_wn + k_offset[None, :] * stride_wk
            w_mask = n_mask[:, None] & k_mask[None, :]
            w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

            acc += tl.sum(w_vals * x_vals[None, :], axis=1)

        tl.store(y_ptr + n_offset, acc.to(tl.float16), mask=n_mask)


    def triton_gemv(W, x, block_n=64, block_k=64):
        if x.dim() == 2:
            x = x.squeeze(0)

        N, K = W.shape
        y = torch.empty(N, dtype=torch.float16, device=W.device)

        grid = (triton.cdiv(N, block_n),)

        gemv_kernel[grid](
            W, x, y,
            N, K,
            W.stride(0), W.stride(1),
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )

        return y


def main():
    parser = argparse.ArgumentParser(description="NCU Profiling Wrapper for Triton GeMV")
    parser.add_argument("--N", type=int, required=True, help="N dimension (output size)")
    parser.add_argument("--K", type=int, required=True, help="K dimension (input size)")
    parser.add_argument("--block-n", type=int, default=64, help="BLOCK_N")
    parser.add_argument("--block-k", type=int, default=64, help="BLOCK_K")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--kernel", type=str, default="triton", choices=["triton", "cublas"],
                       help="Kernel to profile")
    args = parser.parse_args()

    print(f"NCU Profiling: {args.kernel} GeMV")
    print(f"  N={args.N}, K={args.K}")
    print(f"  N % 8 = {args.N % 8} ({'aligned' if args.N % 8 == 0 else 'misaligned'})")
    print(f"  K % 8 = {args.K % 8} ({'aligned' if args.K % 8 == 0 else 'misaligned'})")

    # Create tensors
    W = torch.randn(args.N, args.K, dtype=torch.float16, device="cuda")
    x = torch.randn(args.K, dtype=torch.float16, device="cuda")

    # Warmup
    print(f"\nWarmup ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        if args.kernel == "triton" and HAS_TRITON:
            y = triton_gemv(W, x, block_n=args.block_n, block_k=args.block_k)
        else:
            x_2d = x.unsqueeze(0)
            y = torch.nn.functional.linear(x_2d, W)
    torch.cuda.synchronize()

    # Target kernel for NCU capture
    print("Running target kernel for NCU capture...")
    if args.kernel == "triton" and HAS_TRITON:
        y = triton_gemv(W, x, block_n=args.block_n, block_k=args.block_k)
    else:
        x_2d = x.unsqueeze(0)
        y = torch.nn.functional.linear(x_2d, W)
    torch.cuda.synchronize()

    print("Done.")


if __name__ == "__main__":
    main()
