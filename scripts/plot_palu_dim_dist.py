"""
Plot PaLU compressed dimension distribution (estimated from attention-size JSON).

We do NOT have per-head rank logs in the checked-in PaLU speedup JSONs (they only contain latency).
Instead, we estimate an "effective dimension" per head from per-layer attention parameter ratio:

  eff_dim_per_head ~= round( head_dim * (pruned_params / baseline_params) )

Then we present the number as "4 heads total dim" (= 4 * eff_dim_per_head), and follow the user's
instruction to divide by 4 with rounding to get the displayed per-head dimension.

Usage:
  python scripts/plot_palu_dim_dist.py \
    --json third_party/RAP/results/attention_size/Mistral-7B-v0.3_PALU_retain0.8.json \
    --out results/palu_dim_dist
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_attention_size_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _estimate_dims_per_layer(payload: dict, head_dim: int = 128) -> tuple[list[int], list[int]]:
    baseline = int(payload["baseline_avg_attention_params"])
    pruned = payload["pruned_all_layer_params"]
    if not isinstance(pruned, list) or not pruned:
        raise ValueError("JSON missing non-empty 'pruned_all_layer_params'")

    # Effective per-head dim estimate from parameter ratio.
    dims_per_head = [int(round(head_dim * (int(p) / baseline))) for p in pruned]

    # Convert to "4 heads dim" numbers, then apply the user's rule: divide by 4 with rounding.
    dims_4_heads = [int(round(d * 4)) for d in dims_per_head]
    dims_per_head_from_4 = [int(round(d4 / 4.0)) for d4 in dims_4_heads]
    return dims_4_heads, dims_per_head_from_4


def _plot_hist(values: list[int], title: str, out_path: Path) -> None:
    vmin, vmax = min(values), max(values)
    # Bin step 2 makes the plot less noisy for dims like 128/126/124...
    bins = range(vmin, vmax + 2, 2) if vmax > vmin else [vmin - 1, vmin + 1]
    plt.figure(figsize=(7, 4.5))
    plt.hist(values, bins=bins, edgecolor="black")
    plt.xlabel("Dimension")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_bar_counts(values: list[int], title: str, out_path: Path) -> None:
    uniq, counts = np.unique(np.array(values), return_counts=True)
    plt.figure(figsize=(7, 4.5))
    plt.bar(uniq, counts, width=1.5, edgecolor="black")
    plt.xlabel("Dimension")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_count_scatter(values: list[int], title: str, out_path: Path) -> None:
    uniq, counts = np.unique(np.array(values), return_counts=True)
    plt.figure(figsize=(7, 4.5))
    plt.scatter(uniq, counts, s=35)
    plt.xlabel("Dimension (per head)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_scatter(values: list[int], title: str, out_path: Path) -> None:
    xs = np.arange(len(values))
    ys = np.array(values)
    plt.figure(figsize=(7, 4.5))
    plt.scatter(xs, ys, s=22)
    plt.xlabel("Layer index")
    plt.ylabel("Dimension")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True, help="PaLU attention-size JSON path")
    ap.add_argument("--out", type=Path, required=True, help="Output directory")
    ap.add_argument("--head-dim", type=int, default=128, help="Original per-head dimension (default: 128)")
    ap.add_argument(
        "--num-kv-heads",
        type=int,
        default=8,
        help="Number of KV heads for the model (Mistral-7B default: 8). Used for total count = (K+V)*layers.",
    )
    args = ap.parse_args()

    payload = _load_attention_size_json(args.json)
    model = payload.get("model", "unknown-model")
    retain = payload.get("retain_ratio", "unknown-retain")

    dims_4, dims_head = _estimate_dims_per_layer(payload, head_dim=args.head_dim)

    # Expand "per-layer" dims to "all K and V heads across all layers".
    # Each layer has num_kv_heads K heads and num_kv_heads V heads.
    per_layer_kv_count = int(args.num_kv_heads) * 2
    dims_head_all_kv = [d for d in dims_head for _ in range(per_layer_kv_count)]

    # Save a small machine-readable dump too.
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "dims.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_json": str(args.json),
                "model": model,
                "retain_ratio": retain,
                "head_dim": args.head_dim,
                "num_layers": len(dims_head),
                "num_kv_heads": int(args.num_kv_heads),
                "per_layer_kv_count": per_layer_kv_count,
                "total_count_layers": len(dims_head),
                "total_count_all_kv": len(dims_head_all_kv),
                "dims_4_heads": dims_4,
                "dims_per_head": dims_head,
                "dims_per_head_all_kv": dims_head_all_kv,
            },
            f,
            indent=2,
        )

    _plot_hist(
        dims_head,
        title=f"PaLU estimated per-head dim distribution ({model}, retain={retain})",
        out_path=args.out / "palu_dim_per_head_hist.png",
    )
    _plot_bar_counts(
        dims_head,
        title=f"PaLU estimated per-head dim counts ({model}, retain={retain})",
        out_path=args.out / "palu_dim_per_head_counts.png",
    )
    _plot_count_scatter(
        dims_head,
        title=f"PaLU per-layer per-head dim counts (scatter) ({model}, retain={retain})",
        out_path=args.out / "palu_dim_per_head_counts_scatter.png",
    )
    _plot_count_scatter(
        dims_head_all_kv,
        title=f"PaLU all-(K+V) heads dim counts (scatter) ({model}, retain={retain})",
        out_path=args.out / "palu_dim_per_head_counts_scatter_all_kv.png",
    )
    _plot_hist(
        dims_4,
        title=f"PaLU estimated 4-head dim distribution ({model}, retain={retain})",
        out_path=args.out / "palu_dim_4heads_hist.png",
    )
    _plot_scatter(
        dims_head,
        title=f"PaLU estimated per-head dim by layer ({model}, retain={retain})",
        out_path=args.out / "palu_dim_per_head_scatter.png",
    )
    _plot_scatter(
        dims_4,
        title=f"PaLU estimated 4-head dim by layer ({model}, retain={retain})",
        out_path=args.out / "palu_dim_4heads_scatter.png",
    )

    print(f"Saved plots to: {args.out}")


if __name__ == "__main__":
    main()

