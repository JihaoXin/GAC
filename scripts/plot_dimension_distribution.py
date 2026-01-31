"""
Load per-layer importance scores and plot dimension distributions for various
compression ratios, for all projection matrices (Q, K, V, O).

For each retain ratio and projection, allocates ranks proportional to layer
importance using greedy algorithm, then plots scatter + alignment bands.

Usage:
  python scripts/plot_dimension_distribution.py \
    --scores results/rank_scores/llama3_8b.json results/rank_scores/mistral_7b.json \
    --ratios 0.5 0.6 0.7 0.8 0.9 \
    --out results/rank_distribution/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ── Constants ────────────────────────────────────────────────────────────

METHODS = ["fisher", "magnitude", "activation", "gradient"]
METHOD_LABELS = {"fisher": "Fisher", "magnitude": "Magnitude", "activation": "Activation", "gradient": "Gradient"}
PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj"]
PROJ_LABELS = {"q_proj": "Q", "k_proj": "K", "v_proj": "V", "o_proj": "O"}
PROJ_MARKERS = {"q_proj": "o", "k_proj": "s", "v_proj": "^", "o_proj": "D"}
PROJ_OFFSETS = {"q_proj": -0.3, "k_proj": -0.1, "v_proj": 0.1, "o_proj": 0.3}


# ── Rank allocation (greedy, matches PaLU style) ─────────────────────────

def allocate_ranks(scores: list[float], head_dim: int, retain_ratio: float) -> list[int]:
    """
    Allocate per-layer rank proportional to layer score.
    Returns list of per-layer head dimensions (one int per layer).
    """
    num_layers = len(scores)
    total_budget = int(num_layers * head_dim * retain_ratio)

    scores_arr = np.array(scores, dtype=np.float64)
    scores_arr = np.maximum(scores_arr, 1e-12)
    total_score = scores_arr.sum()

    rank_float = scores_arr / total_score * total_budget

    ranks = np.floor(rank_float).astype(int)
    ranks = np.clip(ranks, 1, head_dim)
    remainder = rank_float - ranks

    deficit = total_budget - ranks.sum()
    if deficit > 0:
        order = np.argsort(-remainder)
        for idx in order[:deficit]:
            ranks[idx] = min(ranks[idx] + 1, head_dim)

    return ranks.tolist()


def compute_alignment_stats(dims) -> dict:
    arr = np.array(dims)
    n = len(arr)
    a8 = int(np.sum(arr % 8 == 0))
    a16 = int(np.sum(arr % 16 == 0))
    mis = n - a8
    return {
        "total": n,
        "8_aligned": a8,
        "8_aligned_pct": round(100.0 * a8 / n, 1),
        "16_aligned": a16,
        "16_aligned_pct": round(100.0 * a16 / n, 1),
        "misaligned": mis,
        "misaligned_pct": round(100.0 * mis / n, 1),
    }


# ── Score access helpers ─────────────────────────────────────────────────

def get_scores(data: dict, method: str, proj: str):
    """Get per-layer score list, handling both old and new format."""
    scores = data["scores"][method]
    if isinstance(scores, dict):
        return scores.get(proj)
    # Old format: only k_proj
    if proj == "k_proj":
        return scores
    return None


def detect_projections(all_data: list[dict]) -> list[str]:
    """Detect available projections from score format."""
    sample = all_data[0]["scores"]["fisher"]
    if isinstance(sample, dict):
        return list(sample.keys())
    return ["k_proj"]


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_scatter_single_model(data: dict, retain_ratio: float, out_dir: Path, projections: list[str]):
    """
    Single-model, 1×4 scatter with alignment bands — all projections overlaid.
    Shared Y axis, no legend. Sized at print width (~7in) so fonts render 1:1.
    """
    model_name = data["model"]
    model_tag = model_name.lower().replace("-", "_").replace(".", "_")
    head_dim = data["head_dim"]
    n_methods = len(METHODS)

    fig, axes = plt.subplots(1, n_methods, figsize=(7.0, 1.8),
                             sharey=True, squeeze=False)
    fig.subplots_adjust(wspace=0.05)

    # First pass: compute global y range across all methods
    global_min, global_max = float("inf"), float("-inf")
    all_proj_ranks = {}
    for method in METHODS:
        proj_ranks = {}
        for proj in projections:
            scores = get_scores(data, method, proj)
            if scores is None:
                continue
            ranks = allocate_ranks(scores, head_dim, retain_ratio)
            proj_ranks[proj] = ranks
            arr = np.array(ranks)
            global_min = min(global_min, arr.min())
            global_max = max(global_max, arr.max())
        all_proj_ranks[method] = proj_ranks

    ymin = max(global_min - 5, 0)
    ymax = global_max + 5

    for col, method in enumerate(METHODS):
        ax = axes[0, col]
        proj_ranks = all_proj_ranks[method]

        if not proj_ranks:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_title(METHOD_LABELS[method], fontsize=8, fontweight="bold")
            continue

        # Draw alignment bands
        for mult in range(int(ymin // 8), int(ymax // 8) + 2):
            center = mult * 8
            ax.axhspan(center - 0.5, center + 0.5, color="#2ecc71", alpha=0.13, zorder=0)
            ax.axhline(center, color="#2ecc71", alpha=0.25, linewidth=0.3, zorder=0)
        for mult in range(int(ymin // 16), int(ymax // 16) + 2):
            center = mult * 16
            ax.axhline(center, color="#27ae60", alpha=0.45, linewidth=0.6, zorder=0)
        for mult in range(int(ymin // 32), int(ymax // 32) + 2):
            center = mult * 32
            ax.axhline(center, color="#1a6b30", alpha=0.85, linewidth=1.2, zorder=1)

        # Plot each projection and collect all dims for overall stat
        all_method_dims = []
        for proj, ranks in proj_ranks.items():
            layers = np.arange(len(ranks)) + PROJ_OFFSETS[proj]
            dims = np.array(ranks)
            marker = PROJ_MARKERS[proj]
            colors = ["#2ecc71" if d % 8 == 0 else "#e74c3c" for d in dims]
            ax.scatter(layers, dims, c=colors, s=10, marker=marker,
                       edgecolors="black", linewidths=0.25, zorder=5)
            all_method_dims.extend(ranks)

        overall_stats = compute_alignment_stats(all_method_dims)
        ax.set_title(f"({chr(97+col)}) {METHOD_LABELS[method]}-{overall_stats['misaligned_pct']:.0f}% misaligned",
                     fontsize=7, pad=4)
        ax.set_xlabel("Layer", fontsize=8)
        ax.tick_params(axis="both", labelsize=7, length=2, pad=2)
        if col == 0:
            ax.set_ylabel("Head Dim", fontsize=8)
            ax.set_yticks([32, 64, 96, 128])
        ax.set_xlim(-1, len(list(proj_ranks.values())[0]))
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    out_path = out_dir / f"scatter_{model_tag}_r{retain_ratio}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}  +  {pdf_path.name}")


def plot_grid(all_data: list[dict], retain_ratio: float, out_dir: Path, proj: str):
    """Bar chart of head dimension counts for a specific projection."""
    n_models = len(all_data)
    n_methods = len(METHODS)
    proj_label = PROJ_LABELS.get(proj, proj)
    fig, axes = plt.subplots(n_models, n_methods, figsize=(5 * n_methods, 4.5 * n_models), squeeze=False)
    fig.suptitle(f"Head Dim Distribution — W{proj_label} (retain={retain_ratio*100:.0f}%)", fontsize=15, y=1.01)

    for row, data in enumerate(all_data):
        model_name = data["model"]
        head_dim = data["head_dim"]
        num_kv_heads = data["num_kv_heads"]

        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            scores = get_scores(data, method, proj)
            if scores is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=14)
                ax.set_title(f"{model_name}\n{METHOD_LABELS[method]}", fontsize=10)
                continue

            ranks = allocate_ranks(scores, head_dim, retain_ratio)
            dims_all = [r for r in ranks for _ in range(num_kv_heads * 2)]

            unique, counts = np.unique(dims_all, return_counts=True)
            colors = ["#2ecc71" if u % 8 == 0 else "#e74c3c" for u in unique]
            ax.bar(unique, counts, width=0.8, color=colors, edgecolor="black", linewidth=0.5)

            stats = compute_alignment_stats(dims_all)
            ax.set_title(f"{model_name}\n{METHOD_LABELS[method]}-{stats['misaligned_pct']:.0f}% misaligned", fontsize=10)
            ax.set_xlabel("Head Dimension")
            ax.set_ylabel("Count")
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"dim_dist_{proj}_r{retain_ratio}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_alignment_heatmap(all_data: list[dict], ratios: list[float], out_dir: Path, proj: str):
    """Heatmap: methods x ratios, one per model, for a specific projection."""
    proj_label = PROJ_LABELS.get(proj, proj)
    for data in all_data:
        model_name = data["model"]
        head_dim = data["head_dim"]
        num_kv_heads = data["num_kv_heads"]

        mat = np.zeros((len(METHODS), len(ratios)))
        has_data = False
        for i, method in enumerate(METHODS):
            scores = get_scores(data, method, proj)
            if scores is None:
                continue
            has_data = True
            for j, ratio in enumerate(ratios):
                ranks = allocate_ranks(scores, head_dim, ratio)
                dims_all = [r for r in ranks for _ in range(num_kv_heads * 2)]
                stats = compute_alignment_stats(dims_all)
                mat[i, j] = stats["misaligned_pct"]

        if not has_data:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(ratios)))
        ax.set_xticklabels([f"{r*100:.0f}%" for r in ratios])
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS])

        for i in range(len(METHODS)):
            for j in range(len(ratios)):
                ax.text(j, i, f"{mat[i,j]:.0f}%", ha="center", va="center", fontsize=10,
                        color="white" if mat[i, j] > 60 else "black")

        ax.set_title(f"Misalignment % — {model_name} — W{proj_label}")
        ax.set_xlabel("Retain Ratio")
        plt.colorbar(im, ax=ax, label="Misaligned %")
        plt.tight_layout()
        out_path = out_dir / f"alignment_heatmap_{proj}_{model_name.lower().replace('-', '_')}.png"
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")


def save_summary_json(all_data: list[dict], ratios: list[float], out_dir: Path, projections: list[str]):
    """Save machine-readable summary of all allocations."""
    summary = []
    for data in all_data:
        for proj in projections:
            for method in METHODS:
                scores = get_scores(data, method, proj)
                if scores is None:
                    continue
                for ratio in ratios:
                    ranks = allocate_ranks(scores, data["head_dim"], ratio)
                    stats = compute_alignment_stats(ranks)
                    summary.append({
                        "model": data["model"],
                        "projection": proj,
                        "method": method,
                        "retain_ratio": ratio,
                        "ranks_per_layer": ranks,
                        "alignment": stats,
                    })
    out_path = out_dir / "allocation_summary.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, nargs="+", required=True, help="Score JSON files from compute_rank_scores.py")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    all_data = []
    for p in args.scores:
        with p.open() as f:
            all_data.append(json.load(f))
        print(f"Loaded: {p} ({all_data[-1]['model']})")

    projections = detect_projections(all_data)
    print(f"Projections: {projections}")

    args.out.mkdir(parents=True, exist_ok=True)

    # Per-model scatter bands (one row per figure, no suptitle)
    print("\nScatter + alignment bands (per model)...")
    for data in all_data:
        print(f"\n  {data['model']}:")
        for ratio in args.ratios:
            plot_scatter_single_model(data, ratio, args.out, projections)

    print("\n  Saving summary JSON...")
    save_summary_json(all_data, args.ratios, args.out, projections)

    print("\nDone!")


if __name__ == "__main__":
    main()
