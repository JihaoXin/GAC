"""GEMM benchmark implementation."""
import json
from pathlib import Path
from typing import List, Dict
import torch

from .config import GEMMConfig
from .measurement import benchmark_gemm
from .utils import set_deterministic, get_dtype, allocate_tensors


def run_gemm_benchmark(config: GEMMConfig, output_dir: Path) -> Dict:
    """
    Run GEMM benchmark sweep according to configuration.
    
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
    
    # Experiment A: Projection-like shapes
    if config.test_projection_qkv:
        print("Running GEMM projection (QKV-like): (M, K) @ (K, N)")
        
        for m in config.m_values:
            for k in config.k_values:
                for n in config.n_values:
                    for dtype_str in config.dtypes:
                        dtype = get_dtype(dtype_str)
                        
                        print(f"  Testing M={m}, K={k}, N={n}, dtype={dtype_str}")
                        
                        # Allocate tensors
                        a, b = allocate_tensors(
                            (m, k),
                            (k, n),
                            dtype=dtype,
                            device=str(device),
                            seed=config.seed
                        )
                        
                        # Benchmark
                        try:
                            result = benchmark_gemm(
                                a, b,
                                warmup_iterations=config.warmup_iterations,
                                measurement_iterations=config.measurement_iterations,
                            )
                            result["experiment_type"] = "projection_qkv"
                            result["m"] = m
                            result["k"] = k
                            result["n"] = n
                            result["dtype"] = dtype_str
                            results["experiments"].append(result)
                        except Exception as e:
                            print(f"    ERROR: {e}")
                            results["experiments"].append({
                                "experiment_type": "projection_qkv",
                                "m": m,
                                "k": k,
                                "n": n,
                                "dtype": dtype_str,
                                "error": str(e),
                            })
                        
                        # Free memory
                        del a, b
                        torch.cuda.empty_cache()
    
    # Experiment B: Reduction dimension focus (K as reduction dim)
    if config.reduction_k:
        print("Running GEMM reduction dimension test: (M, K) @ (K, N) with fixed M, N")
        
        m = config.reduction_m
        n = config.reduction_n
        
        for k in config.k_values:
            for dtype_str in config.dtypes:
                dtype = get_dtype(dtype_str)
                
                print(f"  Testing M={m}, K={k}, N={n}, dtype={dtype_str}")
                
                # Allocate tensors
                a, b = allocate_tensors(
                    (m, k),
                    (k, n),
                    dtype=dtype,
                    device=str(device),
                    seed=config.seed
                )
                
                # Benchmark
                try:
                    result = benchmark_gemm(
                        a, b,
                        warmup_iterations=config.warmup_iterations,
                        measurement_iterations=config.measurement_iterations,
                    )
                    result["experiment_type"] = "reduction_k"
                    result["m"] = m
                    result["k"] = k
                    result["n"] = n
                    result["dtype"] = dtype_str
                    results["experiments"].append(result)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    results["experiments"].append({
                        "experiment_type": "reduction_k",
                        "m": m,
                        "k": k,
                        "n": n,
                        "dtype": dtype_str,
                        "error": str(e),
                    })
                
                # Free memory
                del a, b
                torch.cuda.empty_cache()
    
    # Experiment: Output projection-like (N as reduction dim)
    if config.test_projection_output:
        print("Running GEMM projection (Output-like): (M, N) @ (N, K)")
        
        for m in config.m_values:
            for n in config.n_values:
                for k in config.k_values:
                    for dtype_str in config.dtypes:
                        dtype = get_dtype(dtype_str)
                        
                        print(f"  Testing M={m}, N={n}, K={k}, dtype={dtype_str}")
                        
                        # Allocate tensors: (M, N) @ (N, K)
                        a, b = allocate_tensors(
                            (m, n),
                            (n, k),
                            dtype=dtype,
                            device=str(device),
                            seed=config.seed
                        )
                        
                        # Benchmark
                        try:
                            result = benchmark_gemm(
                                a, b,
                                warmup_iterations=config.warmup_iterations,
                                measurement_iterations=config.measurement_iterations,
                            )
                            result["experiment_type"] = "projection_output"
                            result["m"] = m
                            result["n"] = n
                            result["k"] = k
                            result["dtype"] = dtype_str
                            results["experiments"].append(result)
                        except Exception as e:
                            print(f"    ERROR: {e}")
                            results["experiments"].append({
                                "experiment_type": "projection_output",
                                "m": m,
                                "n": n,
                                "k": k,
                                "dtype": dtype_str,
                                "error": str(e),
                            })
                        
                        # Free memory
                        del a, b
                        torch.cuda.empty_cache()
    
    return results


def save_gemm_results(results: Dict, output_path: Path):
    """Save GEMM benchmark results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
