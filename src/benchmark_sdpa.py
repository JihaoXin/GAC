"""SDPA (Scaled Dot Product Attention) benchmark implementation."""
import json
from pathlib import Path
from typing import List, Dict, Optional
import torch

from .config import SDPAConfig
from .measurement import benchmark_kernel
from .utils import set_deterministic, get_dtype


def detect_sdpa_backend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> Optional[str]:
    """
    Attempt to detect which SDPA backend is being used.
    
    This is heuristic-based since PyTorch doesn't always expose backend selection directly.
    """
    # Try to use torch.backends.cuda.sdp_kernel context manager
    # This allows us to see which backends are available and potentially force one
    
    available_backends = []
    if hasattr(torch.backends.cuda, "sdp_kernel"):
        # Check which backends are available
        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                # Math backend should always work
                available_backends.append("math")
        except Exception:
            pass
        
        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                try:
                    _ = torch.nn.functional.scaled_dot_product_attention(query, key, value)
                    available_backends.append("flash")
                except Exception:
                    pass
        except Exception:
            pass
        
        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
                try:
                    _ = torch.nn.functional.scaled_dot_product_attention(query, key, value)
                    available_backends.append("mem_efficient")
                except Exception:
                    pass
        except Exception:
            pass
    
    # Default: try to infer from error messages or timing patterns
    # For now, return None and let the benchmark record what happens
    return None if not available_backends else ",".join(available_backends)


def run_sdpa_benchmark(config: SDPAConfig, output_dir: Path) -> Dict:
    """
    Run SDPA benchmark sweep according to configuration.
    
    Returns:
        Dictionary with all benchmark results
    """
    set_deterministic(config.seed)
    device = torch.device(config.device)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    results = {
        "config": config.to_dict(),
        "experiments": [],
    }
    
    print("Running SDPA benchmark")
    
    for batch_size in config.batch_sizes:
        for seq_len in config.seq_lengths:
            for head_dim in config.head_dims:
                for dtype_str in config.dtypes:
                    dtype = get_dtype(dtype_str)
                    
                    # Skip if dtype not supported (e.g., bfloat16 on older GPUs)
                    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                        print(f"  Skipping {dtype_str} (not supported on this GPU)")
                        continue
                    
                    n_heads = config.n_heads
                    
                    print(f"  Testing batch={batch_size}, seq_len={seq_len}, n_heads={n_heads}, head_dim={head_dim}, dtype={dtype_str}")
                    
                    # Create attention tensors: (batch, n_heads, seq_len, head_dim)
                    torch.manual_seed(config.seed)
                    query = torch.randn(
                        batch_size, n_heads, seq_len, head_dim,
                        dtype=dtype, device=device
                    )
                    key = torch.randn(
                        batch_size, n_heads, seq_len, head_dim,
                        dtype=dtype, device=device
                    )
                    value = torch.randn(
                        batch_size, n_heads, seq_len, head_dim,
                        dtype=dtype, device=device
                    )
                    
                    # Detect available backends
                    available_backends = detect_sdpa_backend(query, key, value)
                    
                    # Benchmark with default backend selection
                    def kernel_fn():
                        return torch.nn.functional.scaled_dot_product_attention(
                            query, key, value,
                            is_causal=False,  # Not using causal mask for simplicity
                        )
                    
                    try:
                        stats = benchmark_kernel(
                            kernel_fn,
                            warmup_iterations=config.warmup_iterations,
                            measurement_iterations=config.measurement_iterations,
                            device=str(device)
                        )
                        
                        # Try to infer backend by attempting to force each one
                        backend_used = None
                        backend_timings = {}
                        
                        if config.detect_backend and hasattr(torch.backends.cuda, "sdp_kernel"):
                            # Try each backend individually
                            for backend_name in ["flash", "mem_efficient", "math"]:
                                try:
                                    if backend_name == "flash":
                                        ctx = torch.backends.cuda.sdp_kernel(
                                            enable_flash=True,
                                            enable_math=False,
                                            enable_mem_efficient=False
                                        )
                                    elif backend_name == "mem_efficient":
                                        ctx = torch.backends.cuda.sdp_kernel(
                                            enable_flash=False,
                                            enable_math=False,
                                            enable_mem_efficient=True
                                        )
                                    else:  # math
                                        ctx = torch.backends.cuda.sdp_kernel(
                                            enable_flash=False,
                                            enable_math=True,
                                            enable_mem_efficient=False
                                        )
                                    
                                    with ctx:
                                        backend_stats = benchmark_kernel(
                                            kernel_fn,
                                            warmup_iterations=5,  # Fewer for backend detection
                                            measurement_iterations=20,
                                            device=str(device)
                                        )
                                        backend_timings[backend_name] = backend_stats["mean"]
                                        
                                        # Heuristic: if this backend's timing matches the default, it's likely what was used
                                        if backend_used is None:
                                            if abs(backend_stats["mean"] - stats["mean"]) < stats["mean"] * 0.1:
                                                backend_used = backend_name
                                except Exception as e:
                                    backend_timings[backend_name] = f"error: {str(e)}"
                        
                        result = {
                            "batch_size": batch_size,
                            "seq_len": seq_len,
                            "n_heads": n_heads,
                            "head_dim": head_dim,
                            "dtype": dtype_str,
                            "available_backends": available_backends,
                            "backend_used": backend_used,
                            "backend_timings": backend_timings,
                            "timing": stats,
                        }
                        results["experiments"].append(result)
                        
                    except Exception as e:
                        print(f"    ERROR: {e}")
                        results["experiments"].append({
                            "batch_size": batch_size,
                            "seq_len": seq_len,
                            "n_heads": n_heads,
                            "head_dim": head_dim,
                            "dtype": dtype_str,
                            "error": str(e),
                        })
                    
                    # Free memory
                    del query, key, value
                    torch.cuda.empty_cache()
    
    return results


def save_sdpa_results(results: Dict, output_path: Path):
    """Save SDPA benchmark results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
