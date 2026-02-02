"""Minimal script for ncu profiling: run one SDPA call per invocation.

Usage:
    ncu --target-processes all --kernel-name regex:'flash|fmha|attention' \
        python scripts/ncu_profile_sdpa.py --head-dim 96 --batch 4 --seq-len 2048 --n-heads 32
"""
import argparse
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--n-heads", type=int, default=32)
    args = parser.parse_args()

    B, S, H, D = args.batch, args.seq_len, args.n_heads, args.head_dim
    Q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
    K = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")
    V = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda")

    # Warmup (ncu will skip these via --launch-skip)
    for _ in range(3):
        F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    # Target kernel (ncu captures this one)
    F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
