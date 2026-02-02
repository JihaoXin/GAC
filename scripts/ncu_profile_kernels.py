"""Minimal script for ncu profiling: run one GEMM per invocation.

Usage:
    ncu --target-processes all --kernel-name regex:gemm \
        python scripts/ncu_profile_kernels.py --M 1088 --N 2048 --K 128
"""
import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")

    # Warmup (ncu will skip these via --launch-skip)
    for _ in range(3):
        torch.mm(A, B)
    torch.cuda.synchronize()

    # Target kernel (ncu captures this one)
    torch.mm(A, B)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
