"""
Plot baseline vs palu results for infer_sweep, ppl, lmeval.
Usage:
  python scripts/plot_llm_results.py --runs <baseline_dir> <palu_dir> --out <out_dir>
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_run(run_dir: Path):
    with open(run_dir / "raw.json") as f:
        raw = json.load(f)
    with open(run_dir / "summary.json") as f:
        summary = json.load(f)
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    return raw, summary, config


def plot_prefill(baseline_raw, palu_raw, out_dir: Path):
    base = baseline_raw.get("prefill", [])
    palu = palu_raw.get("prefill", [])
    def extract(data):
        xs, ys = [], []
        for m in data:
            if "error" in m:
                continue
            xs.append(m["seq_len"])
            ys.append(m["timing"]["mean"])
        order = np.argsort(xs)
        return np.array(xs)[order], np.array(ys)[order]
    x_b, y_b = extract(base)
    x_p, y_p = extract(palu)
    plt.figure(figsize=(8,5))
    plt.plot(x_b, y_b, marker='o', label="baseline")
    plt.plot(x_p, y_p, marker='o', label="palu")
    plt.xlabel("Sequence length")
    plt.ylabel("Latency (ms)")
    plt.title("Prefill Latency vs Seq Len")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    out = out_dir / "prefill_latency.png"; out_dir.mkdir(parents=True, exist_ok=True); plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")


def plot_prefill_throughput(baseline_raw, palu_raw, out_dir: Path):
    base = baseline_raw.get("prefill", [])
    palu = palu_raw.get("prefill", [])
    def extract(data):
        xs, ys = [], []
        for m in data:
            if "error" in m:
                continue
            xs.append(m["seq_len"])
            ys.append(m["throughput_toks_per_s"]["mean"])
        order = np.argsort(xs)
        return np.array(xs)[order], np.array(ys)[order]
    x_b, y_b = extract(base)
    x_p, y_p = extract(palu)
    plt.figure(figsize=(8,5))
    plt.plot(x_b, y_b, marker='o', label="baseline")
    plt.plot(x_p, y_p, marker='o', label="palu")
    plt.xlabel("Sequence length")
    plt.ylabel("Throughput (tokens/s)")
    plt.title("Prefill Throughput vs Seq Len")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    out = out_dir / "prefill_throughput.png"; out_dir.mkdir(parents=True, exist_ok=True); plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")


def plot_decode_throughput(baseline_raw, palu_raw, out_dir: Path):
    base = baseline_raw.get("decode", [])
    palu = palu_raw.get("decode", [])
    def extract(data):
        ctxs, thr = [], []
        for m in data:
            if "error" in m:
                continue
            ctxs.append(m["context_len"])
            thr.append(m["throughput_toks_per_s"]["mean"])
        return ctxs, thr
    cb, tb = extract(base)
    cp, tp = extract(palu)
    plt.figure(figsize=(8,5))
    plt.scatter(cb, tb, label="baseline", marker='o')
    plt.scatter(cp, tp, label="palu", marker='x')
    plt.xlabel("Context length")
    plt.ylabel("Throughput (tokens/s)")
    plt.title("Decode Throughput vs Context Length")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    out = out_dir / "decode_throughput.png"; out_dir.mkdir(parents=True, exist_ok=True); plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")


def plot_memory(baseline_raw, palu_raw, out_dir: Path):
    base = baseline_raw.get("prefill", [])
    palu = palu_raw.get("prefill", [])
    def extract(data):
        xs, ys = [], []
        for m in data:
            if "error" in m:
                continue
            xs.append(m["seq_len"])
            ys.append(m["memory"]["max_memory_allocated"] / (1024**3))
        order = np.argsort(xs)
        return np.array(xs)[order], np.array(ys)[order]
    x_b, y_b = extract(base)
    x_p, y_p = extract(palu)
    plt.figure(figsize=(8,5))
    plt.plot(x_b, y_b, marker='o', label="baseline")
    plt.plot(x_p, y_p, marker='o', label="palu")
    plt.xlabel("Sequence length")
    plt.ylabel("Peak allocated memory (GB)")
    plt.title("Peak Memory vs Seq Len (Prefill)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    out = out_dir / "prefill_memory.png"; out_dir.mkdir(parents=True, exist_ok=True); plt.savefig(out, dpi=150); plt.close()
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs=2, required=True, help="baseline_dir palu_dir")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base_raw, base_sum, _ = load_run(Path(args.runs[0]))
    palu_raw, palu_sum, _ = load_run(Path(args.runs[1]))

    args.out.mkdir(parents=True, exist_ok=True)
    plot_prefill(base_raw, palu_raw, args.out)
    plot_prefill_throughput(base_raw, palu_raw, args.out)
    plot_decode_throughput(base_raw, palu_raw, args.out)
    plot_memory(base_raw, palu_raw, args.out)
    print("✅ Plotting done.")


if __name__ == "__main__":
    main()
