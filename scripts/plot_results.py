"""Plotting script for benchmark results."""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import argparse

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: Path) -> Dict:
    """Load all results from a results directory."""
    results = {}
    
    gemm_path = results_dir / "gemm_results.json"
    if gemm_path.exists():
        with open(gemm_path) as f:
            results["gemm"] = json.load(f)
    
    sdpa_path = results_dir / "sdpa_results.json"
    if sdpa_path.exists():
        with open(sdpa_path) as f:
            results["sdpa"] = json.load(f)
    
    return results


def plot_gemm_latency_vs_dimension(results: Dict, output_dir: Path):
    """Plot GEMM latency vs dimension (head_dim-like)."""
    experiments = results.get("experiments", [])
    if not experiments:
        print("No GEMM experiments found")
        return
    
    # Group by experiment type and M value
    data = {}
    for exp in experiments:
        if "error" in exp:
            continue
        
        exp_type = exp.get("experiment_type", "unknown")
        m = exp.get("m", exp.get("shape", {}).get("m", 0))
        n = exp.get("n", exp.get("shape", {}).get("n", 0))
        k = exp.get("k", exp.get("shape", {}).get("k", 0))
        dtype = exp.get("dtype", "unknown")
        
        # Use N as the dimension to plot (head_dim-like)
        key = (exp_type, m, dtype)
        if key not in data:
            data[key] = {"dims": [], "latency_mean": [], "latency_std": []}
        
        timing = exp.get("timing", {})
        data[key]["dims"].append(n)
        data[key]["latency_mean"].append(timing.get("mean", 0))
        data[key]["latency_std"].append(timing.get("std", 0))
    
    if not data:
        print("No valid GEMM data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for (exp_type, m, dtype), values in data.items():
        # Sort by dimension
        sorted_indices = np.argsort(values["dims"])
        dims = np.array(values["dims"])[sorted_indices]
        latency_mean = np.array(values["latency_mean"])[sorted_indices]
        latency_std = np.array(values["latency_std"])[sorted_indices]
        
        label = f"{exp_type}, M={m}, {dtype}"
        ax.errorbar(
            dims, latency_mean, yerr=latency_std,
            marker='o', label=label, capsize=3, capthick=1
        )
    
    ax.set_xlabel("Dimension (N, head_dim-like)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("GEMM Latency vs Dimension")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "plots" / "gemm_latency_vs_dimension.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_gemm_tflops_vs_dimension(results: Dict, output_dir: Path):
    """Plot GEMM achieved TFLOPs vs dimension."""
    experiments = results.get("experiments", [])
    if not experiments:
        print("No GEMM experiments found")
        return
    
    # Group by experiment type and M value
    data = {}
    for exp in experiments:
        if "error" in exp:
            continue
        
        exp_type = exp.get("experiment_type", "unknown")
        m = exp.get("m", exp.get("shape", {}).get("m", 0))
        n = exp.get("n", exp.get("shape", {}).get("n", 0))
        dtype = exp.get("dtype", "unknown")
        
        key = (exp_type, m, dtype)
        if key not in data:
            data[key] = {"dims": [], "tflops_mean": [], "tflops_std": []}
        
        perf = exp.get("performance", {})
        tflops_stats = perf.get("tflops_stats", {})
        data[key]["dims"].append(n)
        data[key]["tflops_mean"].append(perf.get("tflops_mean", 0))
        data[key]["tflops_std"].append(tflops_stats.get("std", 0))
    
    if not data:
        print("No valid GEMM data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for (exp_type, m, dtype), values in data.items():
        # Sort by dimension
        sorted_indices = np.argsort(values["dims"])
        dims = np.array(values["dims"])[sorted_indices]
        tflops_mean = np.array(values["tflops_mean"])[sorted_indices]
        tflops_std = np.array(values["tflops_std"])[sorted_indices]
        
        label = f"{exp_type}, M={m}, {dtype}"
        ax.errorbar(
            dims, tflops_mean, yerr=tflops_std,
            marker='o', label=label, capsize=3, capthick=1
        )
    
    # Add theoretical peak line (A100: ~312 TFLOPS for FP16)
    ax.axhline(y=312, color='r', linestyle='--', alpha=0.5, label='A100 Peak (FP16)')
    
    ax.set_xlabel("Dimension (N, head_dim-like)")
    ax.set_ylabel("Achieved TFLOPs")
    ax.set_title("GEMM Achieved TFLOPs vs Dimension")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "plots" / "gemm_tflops_vs_dimension.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_sdpa_backend_vs_head_dim(results: Dict, output_dir: Path):
    """Plot SDPA backend selection vs head_dim."""
    experiments = results.get("experiments", [])
    if not experiments:
        print("No SDPA experiments found")
        return
    
    # Group by batch_size and seq_len
    data = {}
    for exp in experiments:
        if "error" in exp:
            continue
        
        batch_size = exp.get("batch_size", 0)
        seq_len = exp.get("seq_len", 0)
        head_dim = exp.get("head_dim", 0)
        backend = exp.get("backend_used", "unknown")
        
        key = (batch_size, seq_len)
        if key not in data:
            data[key] = {"head_dims": [], "backends": []}
        
        data[key]["head_dims"].append(head_dim)
        data[key]["backends"].append(backend)
    
    if not data:
        print("No valid SDPA data to plot")
        return
    
    fig, axes = plt.subplots(len(data), 1, figsize=(10, 4 * len(data)))
    if len(data) == 1:
        axes = [axes]
    
    backend_colors = {
        "flash": "green",
        "mem_efficient": "orange",
        "math": "red",
        "unknown": "gray",
    }
    
    for idx, ((batch_size, seq_len), values) in enumerate(data.items()):
        ax = axes[idx]
        
        # Sort by head_dim
        sorted_indices = np.argsort(values["head_dims"])
        head_dims = np.array(values["head_dims"])[sorted_indices]
        backends = np.array(values["backends"])[sorted_indices]
        
        # Create categorical plot
        backend_numeric = []
        for b in backends:
            if b == "flash":
                backend_numeric.append(0)
            elif b == "mem_efficient":
                backend_numeric.append(1)
            elif b == "math":
                backend_numeric.append(2)
            else:
                backend_numeric.append(3)
        
        ax.scatter(head_dims, backend_numeric, s=100, alpha=0.7)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["Flash", "MemEfficient", "Math", "Unknown"])
        ax.set_xlabel("Head Dimension")
        ax.set_ylabel("Backend")
        ax.set_title(f"SDPA Backend Selection (batch={batch_size}, seq_len={seq_len})")
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "plots" / "sdpa_backend_vs_head_dim.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_sdpa_latency_vs_head_dim(results: Dict, output_dir: Path):
    """Plot SDPA latency vs head_dim."""
    experiments = results.get("experiments", [])
    if not experiments:
        print("No SDPA experiments found")
        return
    
    # Group by batch_size and seq_len
    data = {}
    for exp in experiments:
        if "error" in exp:
            continue
        
        batch_size = exp.get("batch_size", 0)
        seq_len = exp.get("seq_len", 0)
        head_dim = exp.get("head_dim", 0)
        
        key = (batch_size, seq_len)
        if key not in data:
            data[key] = {"head_dims": [], "latency_mean": [], "latency_std": []}
        
        timing = exp.get("timing", {})
        data[key]["head_dims"].append(head_dim)
        data[key]["latency_mean"].append(timing.get("mean", 0))
        data[key]["latency_std"].append(timing.get("std", 0))
    
    if not data:
        print("No valid SDPA data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for (batch_size, seq_len), values in data.items():
        # Sort by head_dim
        sorted_indices = np.argsort(values["head_dims"])
        head_dims = np.array(values["head_dims"])[sorted_indices]
        latency_mean = np.array(values["latency_mean"])[sorted_indices]
        latency_std = np.array(values["latency_std"])[sorted_indices]
        
        label = f"batch={batch_size}, seq_len={seq_len}"
        ax.errorbar(
            head_dims, latency_mean, yerr=latency_std,
            marker='o', label=label, capsize=3, capthick=1
        )
    
    ax.set_xlabel("Head Dimension")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("SDPA Latency vs Head Dimension")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "plots" / "sdpa_latency_vs_head_dim.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Results directory containing gemm_results.json and/or sdpa_results.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: same as results_dir)"
    )
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"Error: Results directory does not exist: {args.results_dir}")
        sys.exit(1)
    
    output_dir = args.output_dir if args.output_dir else args.results_dir
    
    print(f"Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)
    
    if "gemm" in results:
        print("\nGenerating GEMM plots...")
        plot_gemm_latency_vs_dimension(results["gemm"], output_dir)
        plot_gemm_tflops_vs_dimension(results["gemm"], output_dir)
    
    if "sdpa" in results:
        print("\nGenerating SDPA plots...")
        plot_sdpa_latency_vs_head_dim(results["sdpa"], output_dir)
        plot_sdpa_backend_vs_head_dim(results["sdpa"], output_dir)
    
    print("\n✅ Plotting completed!")


if __name__ == "__main__":
    main()
