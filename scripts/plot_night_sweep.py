"""Plotting script for night sweep experiments."""
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results(run_dir: Path):
    """Load results from a run directory."""
    raw_path = run_dir / "raw.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"raw.json not found in {run_dir}")
    
    with open(raw_path) as f:
        return json.load(f)


def plot_s1_sdpa_dense(results: dict, output_dir: Path):
    """Plot S1: SDPA dense sweep."""
    measurements = results.get("measurements", [])
    if not measurements:
        return
    
    # Group by shape
    by_shape = {}
    for m in measurements:
        if "error" in m:
            continue
        shape_key = f"B={m['shape']['batch']},S={m['shape']['seq_len']}"
        if shape_key not in by_shape:
            by_shape[shape_key] = {"head_dims": [], "latencies": [], "stds": []}
        
        by_shape[shape_key]["head_dims"].append(m["shape"]["head_dim"])
        by_shape[shape_key]["latencies"].append(m["timing"]["mean"])
        by_shape[shape_key]["stds"].append(m["timing"]["std"])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for shape_key, data in by_shape.items():
        sorted_idx = np.argsort(data["head_dims"])
        head_dims = np.array(data["head_dims"])[sorted_idx]
        latencies = np.array(data["latencies"])[sorted_idx]
        stds = np.array(data["stds"])[sorted_idx]
        
        ax.errorbar(head_dims, latencies, yerr=stds, marker='o', label=shape_key, capsize=3)
    
    ax.set_xlabel("Head Dimension")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("S1: SDPA Dense Sweep - Latency vs Head Dimension")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "plots" / "s1_latency_vs_head_dim.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_s2_backend_forced(results: dict, output_dir: Path):
    """Plot S2: SDPA backend forced."""
    measurements = results.get("measurements", [])
    if not measurements:
        return
    
    # Group by backend and head_dim
    data = {}
    for m in measurements:
        if "error" in m:
            continue
        backend = m["backend"]
        hd = m["head_dim"]
        if backend not in data:
            data[backend] = {}
        data[backend][hd] = m["timing"]["mean"]
    
    head_dims = sorted(set(m["head_dim"] for m in measurements if "error" not in m))
    backends = sorted(data.keys())
    
    x = np.arange(len(head_dims))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, backend in enumerate(backends):
        values = [data[backend].get(hd, 0) for hd in head_dims]
        ax.bar(x + i * width, values, width, label=backend)
    
    ax.set_xlabel("Head Dimension")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("S2: SDPA Backend Forced - Latency by Backend and Head Dimension")
    ax.set_xticks(x + width * (len(backends) - 1) / 2)
    ax.set_xticklabels(head_dims)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "plots" / "s2_backend_latency.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_g3_gemm_k(results: dict, output_dir: Path):
    """Plot G3: GEMM K dimension sweep."""
    measurements = results.get("measurements", [])
    if not measurements:
        return
    
    # Group by dtype
    by_dtype = {}
    for m in measurements:
        if "error" in m:
            continue
        dtype = m["dtype"]
        if dtype not in by_dtype:
            by_dtype[dtype] = {"K": [], "latency": [], "tflops": []}
        
        by_dtype[dtype]["K"].append(m["shape"]["K"])
        by_dtype[dtype]["latency"].append(m["timing"]["mean"])
        if "derived" in m:
            by_dtype[dtype]["tflops"].append(m["derived"]["tflops_mean"])
    
    # Latency plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for dtype, data in by_dtype.items():
        sorted_idx = np.argsort(data["K"])
        K = np.array(data["K"])[sorted_idx]
        latency = np.array(data["latency"])[sorted_idx]
        ax.plot(K, latency, marker='o', label=f"{dtype} latency")
    ax.set_xlabel("K Dimension")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("G3: GEMM K Dimension Sweep - Latency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / "plots" / "g3_latency_vs_K.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # TFLOPs plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for dtype, data in by_dtype.items():
        if data["tflops"]:
            sorted_idx = np.argsort(data["K"])
            K = np.array(data["K"])[sorted_idx]
            tflops = np.array(data["tflops"])[sorted_idx]
            ax.plot(K, tflops, marker='o', label=f"{dtype} TFLOPs")
    ax.set_xlabel("K Dimension")
    ax.set_ylabel("TFLOPs")
    ax.set_title("G3: GEMM K Dimension Sweep - TFLOPs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / "plots" / "g3_tflops_vs_K.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_g4_gemm_n(results: dict, output_dir: Path):
    """Plot G4: GEMM N dimension sweep."""
    measurements = results.get("measurements", [])
    if not measurements:
        return
    
    # Group by M and dtype
    by_m_dtype = {}
    for m in measurements:
        if "error" in m:
            continue
        M = m["shape"]["M"]
        dtype = m["dtype"]
        key = (M, dtype)
        if key not in by_m_dtype:
            by_m_dtype[key] = {"N": [], "latency": []}
        by_m_dtype[key]["N"].append(m["shape"]["N"])
        by_m_dtype[key]["latency"].append(m["timing"]["mean"])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for (M, dtype), data in sorted(by_m_dtype.items()):
        sorted_idx = np.argsort(data["N"])
        N = np.array(data["N"])[sorted_idx]
        latency = np.array(data["latency"])[sorted_idx]
        ax.plot(N, latency, marker='o', label=f"M={M}, {dtype}")
    
    ax.set_xlabel("N Dimension")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("G4: GEMM N Dimension Sweep - Latency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / "plots" / "g4_latency_vs_N.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_p1_padding_rescue(results: dict, output_dir: Path):
    """Plot P1: Padding rescue."""
    measurements = results.get("measurements", [])
    if not measurements:
        return
    
    # Group by operation
    by_op = {}
    for m in measurements:
        if "error" in m:
            continue
        op = m.get("operation", "unknown")
        if op not in by_op:
            by_op[op] = {"physical_dim": [], "latency": [], "overhead": []}
        by_op[op]["physical_dim"].append(m["physical_dim"])
        by_op[op]["latency"].append(m["timing"]["mean"])
        by_op[op]["overhead"].append(m.get("memory_overhead_pct", 0))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Latency plot
    for op, data in by_op.items():
        sorted_idx = np.argsort(data["physical_dim"])
        dims = np.array(data["physical_dim"])[sorted_idx]
        latency = np.array(data["latency"])[sorted_idx]
        ax1.bar([f"{op}\n{d}" for d in dims], latency, label=op)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("P1: Padding Rescue - Latency")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Overhead plot
    for op, data in by_op.items():
        sorted_idx = np.argsort(data["physical_dim"])
        dims = np.array(data["physical_dim"])[sorted_idx]
        overhead = np.array(data["overhead"])[sorted_idx]
        ax2.bar([f"{op}\n{d}" for d in dims], overhead, label=op)
    ax2.set_ylabel("Memory Overhead (%)")
    ax2.set_title("P1: Padding Rescue - Memory Overhead")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "plots" / "p1_padding_rescue.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_het1_hetero_batching(results: dict, output_dir: Path):
    """Plot HET1: Heterogeneous batching."""
    measurements = results.get("measurements", [])
    if not measurements:
        return
    
    patterns = []
    latencies = []
    num_calls = []
    
    for m in measurements:
        if "error" in m:
            continue
        patterns.append(m["pattern"])
        if "total_latency_ms" in m:
            latencies.append(m["total_latency_ms"])
        else:
            latencies.append(m["timing"]["mean"])
        num_calls.append(m.get("num_gemm_calls", 1))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.bar(patterns, latencies)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("HET1: Heterogeneous Batching - Latency")
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.bar(patterns, num_calls)
    ax2.set_ylabel("Number of GEMM Calls")
    ax2.set_title("HET1: Heterogeneous Batching - GEMM Calls")
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "plots" / "het1_hetero_batching.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


PLOTTERS = {
    "S1_sdpa_dense_sweep": plot_s1_sdpa_dense,
    "S2_sdpa_backend_forced": plot_s2_backend_forced,
    "G3_gemm_k_dense": plot_g3_gemm_k,
    "G4_gemm_n_dense_projectionlike": plot_g4_gemm_n,
    "P1_padding_rescue": plot_p1_padding_rescue,
    "HET1_head_hetero_batching_penalty": plot_het1_hetero_batching,
}


def main():
    parser = argparse.ArgumentParser(description="Plot night sweep experiment results")
    parser.add_argument("run_dir", type=Path, help="Run directory containing raw.json")
    args = parser.parse_args()
    
    results = load_results(args.run_dir)
    exp_name = results.get("experiment", "")
    
    if exp_name not in PLOTTERS:
        print(f"Warning: No plotter for experiment {exp_name}")
        return
    
    print(f"Generating plots for {exp_name}...")
    PLOTTERS[exp_name](results, args.run_dir)
    print("✅ Plotting completed!")


if __name__ == "__main__":
    main()
