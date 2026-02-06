#!/usr/bin/env python3
"""
Inspect Triton compiled kernel to see actual load instructions.
This extracts PTX and SASS from Triton's JIT cache.
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
    """Simple row-wise load kernel - same as our benchmark."""
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


def inspect_kernel(n_rows, row_width, block_n=32, block_k=64):
    """Compile kernel and extract assembly."""

    # Create dummy tensors to trigger compilation
    data = torch.randn(n_rows, row_width, dtype=torch.float16, device='cuda')
    out = torch.empty(n_rows, dtype=torch.float16, device='cuda')

    grid = (triton.cdiv(n_rows, block_n),)

    # Run once to compile
    ldg_row_kernel[grid](data, out, n_rows, row_width, row_width,
                         BLOCK_N=block_n, BLOCK_K=block_k)
    torch.cuda.synchronize()

    # Get the compiled kernel
    # Triton stores compiled kernels in its cache
    key = (n_rows, row_width, row_width, block_n, block_k)

    # Try to access the compiled binary through Triton's API
    print(f"\n{'='*70}")
    print(f"Kernel for N={n_rows}, K={row_width}, BLOCK_N={block_n}, BLOCK_K={block_k}")
    print(f"K % 8 = {row_width % 8}, K % 16 = {row_width % 16}")
    print(f"{'='*70}")

    # Method 1: Use the kernel's asm property if available
    try:
        # Get the kernel function object
        kernel_fn = ldg_row_kernel

        # Try to get compiled info
        # In newer Triton versions, we can access the cache
        import triton.compiler as tc

        # The kernel should have cached its compilation
        # Let's try to print any available info
        print("\nTriton kernel info:")
        print(f"  Function name: {kernel_fn.fn.__name__}")

    except Exception as e:
        print(f"Could not get kernel info: {e}")

    # Method 2: Look at Triton's cache directory
    try:
        import os
        cache_dir = os.path.expanduser("~/.triton/cache")
        if os.path.exists(cache_dir):
            print(f"\nTriton cache directory: {cache_dir}")
            # List recent files
            import glob
            files = glob.glob(f"{cache_dir}/**/*", recursive=True)
            files = sorted(files, key=os.path.getmtime, reverse=True)[:10]
            print(f"Recent cache files: {len(files)}")
            for f in files[:5]:
                print(f"  {f}")
    except Exception as e:
        print(f"Could not inspect cache: {e}")

    return out


def dump_ptx_sass():
    """Use triton's built-in tools to dump PTX/SASS."""

    # Create a simple test case
    n, k = 1024, 4096
    data = torch.randn(n, k, dtype=torch.float16, device='cuda')
    out = torch.empty(n, dtype=torch.float16, device='cuda')

    grid = (triton.cdiv(n, 32),)

    print("\n" + "="*70)
    print("Attempting to extract PTX/SASS from Triton kernel")
    print("="*70)

    # Compile and run
    compiled = ldg_row_kernel[grid](data, out, n, k, k, BLOCK_N=32, BLOCK_K=64)
    torch.cuda.synchronize()

    # Try to get the PTX
    try:
        # In Triton 2.x, we can compile separately
        from triton.compiler import compile as triton_compile
        from triton.runtime import driver

        # Get device properties
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        print(f"\nCUDA capability: {capability}")

    except ImportError as e:
        print(f"Triton compile import failed: {e}")

    # Method: Use cuobjdump on the cached binary
    try:
        import subprocess
        import os

        cache_dir = os.path.expanduser("~/.triton/cache")
        if os.path.exists(cache_dir):
            # Find the most recent .cubin or .ptx file
            import glob
            cubins = glob.glob(f"{cache_dir}/**/*.cubin", recursive=True)
            cubins = sorted(cubins, key=os.path.getmtime, reverse=True)

            if cubins:
                latest = cubins[0]
                print(f"\nFound cubin: {latest}")

                # Use cuobjdump to extract SASS
                result = subprocess.run(
                    ["cuobjdump", "-sass", latest],
                    capture_output=True, text=True, timeout=30
                )

                if result.returncode == 0:
                    sass = result.stdout

                    # Look for load instructions
                    print("\n" + "="*70)
                    print("SASS Load Instructions Found:")
                    print("="*70)

                    # Count different load instruction types
                    ldg_128 = sass.count("LDG.E.128")
                    ldg_64 = sass.count("LDG.E.64") + sass.count("LDG.64")
                    ldg_32 = sass.count("LDG.E.32") + sass.count("LDG.32")
                    ldg_other = sass.count("LDG.E") - ldg_128 - ldg_64 - ldg_32
                    lds = sass.count("LDS")  # shared memory loads

                    print(f"  LDG.E.128: {ldg_128}")
                    print(f"  LDG.E.64:  {ldg_64}")
                    print(f"  LDG.E.32:  {ldg_32}")
                    print(f"  LDG other: {ldg_other}")
                    print(f"  LDS (shared): {lds}")

                    # Also look for store instructions
                    stg_128 = sass.count("STG.E.128")
                    stg_64 = sass.count("STG.E.64")
                    stg_32 = sass.count("STG.E.32")

                    print(f"\n  STG.E.128: {stg_128}")
                    print(f"  STG.E.64:  {stg_64}")
                    print(f"  STG.E.32:  {stg_32}")

                    # Print relevant lines
                    print("\n" + "-"*70)
                    print("Load instruction lines:")
                    print("-"*70)
                    for line in sass.split('\n'):
                        if 'LDG' in line or 'LDS' in line:
                            print(f"  {line.strip()}")

                else:
                    print(f"cuobjdump error: {result.stderr}")
            else:
                print("No .cubin files found in cache")

    except Exception as e:
        print(f"Could not run cuobjdump: {e}")


if __name__ == "__main__":
    import sys

    print("="*70)
    print("Triton Kernel Assembly Inspector")
    print("="*70)

    # Test with both aligned and misaligned K
    for k in [4096, 4088, 4095]:
        inspect_kernel(1024, k)

    # Try to dump PTX/SASS
    dump_ptx_sass()
