"""
Latency benchmark for SVD-LLM compressed models with different alignment strategies.

Measures:
  1. GEMM microbenchmark: per-projection factorized GEMM latency
  2. End-to-end inference: prefill latency for full compressed model

Usage (via Slurm):
    sbatch slurm/run_svdllm_latency.sbatch
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Import from our experiment script
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.svdllm_gac_experiment import (
    MODEL_ID, NUM_LAYERS, HIDDEN, INTER, ALL_PROJS, PROJ_SHAPES,
    ATTN_PROJS, MLP_PROJS,
    load_model, find_layers,
    compute_all_svdllm_ranks, param_cost, compute_fisher_proxy,
    strategy_round_to_n, strategy_gac_dp,
    whitened_svd_compress, LowRankWrapper,
)


# ---------------------------------------------------------------------------
# CUDA timing utilities
# ---------------------------------------------------------------------------
def measure_latency(fn, warmup=50, repeats=200, device="cuda"):
    """Measure kernel latency using CUDA events."""
    torch.cuda.synchronize(device)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end))

    arr = np.array(times)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "count": len(times),
    }


# ---------------------------------------------------------------------------
# Part 1: GEMM Microbenchmark
# ---------------------------------------------------------------------------
def bench_gemm_shapes(strategies_ranks, dev, dtype=torch.float16,
                      warmup=50, repeats=200, batch_tokens=512):
    """
    Benchmark GEMM latency for the factorized layers at each strategy's rank values.

    For a factorized linear W ≈ U @ V:
      V: (rank, in_features)  — first GEMM: x @ V^T → (batch, rank)
      U: (out_features, rank) — second GEMM: h @ U^T → (batch, out_features)

    We benchmark: x @ V^T and h @ U^T for representative rank values.
    """
    results = {}

    for strat_name, ranks in strategies_ranks.items():
        print(f"\n  GEMM benchmark: {strat_name}")
        strat_results = {}

        # Get unique (proj_type, rank) pairs
        proj_ranks = {}
        for (layer, proj), r in ranks.items():
            key = proj
            if key not in proj_ranks:
                proj_ranks[key] = set()
            proj_ranks[key].add(r)

        for proj_name in ALL_PROJS:
            m, n = PROJ_SHAPES[proj_name]  # (out_features, in_features)
            unique_ranks = sorted(proj_ranks.get(proj_name, set()))

            for rank in unique_ranks:
                label = f"{proj_name}_r{rank}"

                # Benchmark V projection: (batch, in_features) @ (in_features, rank)
                x = torch.randn(batch_tokens, n, dtype=dtype, device=dev)
                V = torch.randn(rank, n, dtype=dtype, device=dev)
                fn_v = lambda x=x, V=V: torch.mm(x, V.t())
                res_v = measure_latency(fn_v, warmup, repeats, str(dev))

                # Benchmark U projection: (batch, rank) @ (rank, out_features)
                h = torch.randn(batch_tokens, rank, dtype=dtype, device=dev)
                U = torch.randn(m, rank, dtype=dtype, device=dev)
                fn_u = lambda h=h, U=U: torch.mm(h, U.t())
                res_u = measure_latency(fn_u, warmup, repeats, str(dev))

                # Combined (V + U)
                total_mean = res_v["mean_ms"] + res_u["mean_ms"]

                strat_results[label] = {
                    "proj": proj_name,
                    "rank": rank,
                    "rank_mod8": rank % 8,
                    "aligned": rank % 8 == 0,
                    "shape_V": f"({batch_tokens}, {n}) @ ({n}, {rank})",
                    "shape_U": f"({batch_tokens}, {rank}) @ ({rank}, {m})",
                    "V_proj": res_v,
                    "U_proj": res_u,
                    "total_mean_ms": total_mean,
                }
                aligned_str = "aligned" if rank % 8 == 0 else f"mod8={rank%8}"
                print(f"    {label} ({aligned_str}): V={res_v['mean_ms']:.3f}ms "
                      f"U={res_u['mean_ms']:.3f}ms total={total_mean:.3f}ms")

                del x, V, h, U
                torch.cuda.empty_cache()

        results[strat_name] = strat_results

    return results


# ---------------------------------------------------------------------------
# Part 2: Per-layer forward pass latency
# ---------------------------------------------------------------------------
def bench_layer_latency(model, dev, seq_len=512, warmup=10, repeats=50):
    """Measure per-layer forward pass latency."""
    model.to(dev)
    model.eval()

    dtype = next(model.parameters()).dtype
    x = torch.randn(1, seq_len, HIDDEN, dtype=dtype, device=dev)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=dev)
    position_ids = torch.arange(seq_len, device=dev).unsqueeze(0)

    layer_times = []
    for i, layer in enumerate(model.model.layers):
        layer.to(dev)

        def run_layer(layer=layer, x=x):
            return layer(x, attention_mask=attention_mask, position_ids=position_ids)

        res = measure_latency(run_layer, warmup, repeats, str(dev))
        layer_times.append({"layer": i, **res})

        layer.cpu()
        torch.cuda.empty_cache()

    return layer_times


# ---------------------------------------------------------------------------
# Part 3: End-to-end prefill latency
# ---------------------------------------------------------------------------
def bench_prefill_latency(model, tokenizer, dev, seq_lens=[128, 256, 512],
                          warmup=5, repeats=20):
    """Measure end-to-end prefill latency."""
    model.to(dev)
    model.eval()

    results = {}
    for seq_len in seq_lens:
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len), device=dev)

        def run_prefill(ids=input_ids):
            with torch.no_grad():
                return model(ids, use_cache=False)

        res = measure_latency(run_prefill, warmup, repeats, str(dev))
        results[seq_len] = res
        tps = seq_len / (res["mean_ms"] / 1000)
        print(f"    seq_len={seq_len}: {res['mean_ms']:.2f}ms "
              f"(p50={res['p50_ms']:.2f}ms, {tps:.0f} tok/s)")

    model.cpu()
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Part 4: Decode latency (token-by-token generation)
# ---------------------------------------------------------------------------
def bench_decode_latency(model, tokenizer, dev, prompt_len=128, gen_tokens=64,
                         warmup=3, repeats=10):
    """
    Measure decode latency: time to generate `gen_tokens` after prefill.
    Reports per-token decode latency.
    """
    model.to(dev)
    model.eval()

    input_ids = torch.randint(0, tokenizer.vocab_size, (1, prompt_len), device=dev)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                max_new_tokens=gen_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    torch.cuda.synchronize(dev)

    # Measure total generation time
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize(dev)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        with torch.no_grad():
            _ = model.generate(
                input_ids,
                max_new_tokens=gen_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        end.record()
        torch.cuda.synchronize(dev)
        times.append(start.elapsed_time(end))

    arr = np.array(times)
    total_ms = float(np.mean(arr))
    per_token_ms = total_ms / gen_tokens

    result = {
        "prompt_len": prompt_len,
        "gen_tokens": gen_tokens,
        "total_mean_ms": total_ms,
        "total_std_ms": float(np.std(arr)),
        "per_token_ms": per_token_ms,
        "tokens_per_sec": 1000.0 / per_token_ms,
        "count": len(times),
    }

    print(f"    prompt={prompt_len}, gen={gen_tokens}: {total_ms:.2f}ms total, "
          f"{per_token_ms:.2f}ms/tok ({result['tokens_per_sec']:.1f} tok/s)")

    model.cpu()
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def generate_latency_plots(gemm_results, prefill_results, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    colors = {"baseline": "#95a5a6", "svdllm": "#e74c3c", "aligned_8": "#3498db", "gac_dp": "#2ecc71"}

    # --- GEMM comparison: total latency per projection type ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, proj_group, title in [
        (axes[0], ATTN_PROJS, "Attention Projections"),
        (axes[1], MLP_PROJS, "MLP Projections"),
    ]:
        strat_names = list(gemm_results.keys())
        x = np.arange(len(proj_group))
        width = 0.8 / len(strat_names)

        for i, strat in enumerate(strat_names):
            totals = []
            for proj in proj_group:
                # Average across rank values for this strategy/projection
                vals = [v["total_mean_ms"] for k, v in gemm_results[strat].items()
                        if v["proj"] == proj]
                totals.append(np.mean(vals) if vals else 0)
            bars = ax.bar(x + i * width, totals, width,
                         label=strat, color=colors.get(strat, "#999"),
                         edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, totals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x + width * (len(strat_names) - 1) / 2)
        ax.set_xticklabels(proj_group, rotation=45, ha="right")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.suptitle("GEMM Latency: Factorized Layer (V + U projection)", fontsize=13)
    plt.tight_layout()
    plt.savefig(plot_dir / "gemm_latency_comparison.png", dpi=150)
    plt.close()

    # --- Prefill latency comparison ---
    if prefill_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        strat_names = list(prefill_results.keys())
        seq_lens = sorted(next(iter(prefill_results.values())).keys())

        for strat in strat_names:
            means = [prefill_results[strat][sl]["mean_ms"] for sl in seq_lens]
            ax.plot(seq_lens, means, "o-", label=strat,
                   color=colors.get(strat, "#999"), linewidth=2, markersize=6)

        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Prefill Latency (ms)")
        ax.set_title("End-to-End Prefill Latency by Strategy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / "prefill_latency_comparison.png", dpi=150)
        plt.close()

    # --- Speedup table ---
    if "svdllm" in prefill_results and len(prefill_results) > 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        baseline_times = prefill_results["svdllm"]
        rows = []
        for strat in strat_names:
            if strat == "svdllm":
                continue
            for sl in seq_lens:
                base = baseline_times[sl]["mean_ms"]
                curr = prefill_results[strat][sl]["mean_ms"]
                speedup = (base - curr) / base * 100
                rows.append([strat, sl, f"{curr:.2f}", f"{speedup:+.2f}%"])

        if rows:
            table = ax.table(cellText=rows,
                           colLabels=["Strategy", "Seq Len", "Latency (ms)", "vs SVD-LLM"],
                           loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            ax.axis("off")
            ax.set_title("Latency Improvement vs Unaligned SVD-LLM")
            plt.tight_layout()
            plt.savefig(plot_dir / "speedup_table.png", dpi=150)
            plt.close()

    print(f"  Plots saved to {plot_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=0.7)
    parser.add_argument("--output", type=str, default="results/svdllm_latency")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--profiling-path", type=str, default=None)
    parser.add_argument("--gemm-warmup", type=int, default=100)
    parser.add_argument("--gemm-repeats", type=int, default=500)
    parser.add_argument("--prefill-warmup", type=int, default=5)
    parser.add_argument("--prefill-repeats", type=int, default=30)
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024")
    parser.add_argument("--batch-tokens", type=int, default=512,
                        help="Batch size (tokens) for GEMM microbenchmark")
    parser.add_argument("--decode-prompt-len", type=int, default=128,
                        help="Prompt length for decode benchmark")
    parser.add_argument("--decode-gen-tokens", type=int, default=64,
                        help="Number of tokens to generate for decode benchmark")
    parser.add_argument("--decode-warmup", type=int, default=3)
    parser.add_argument("--decode-repeats", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 70)
    print("SVD-LLM Latency Benchmark")
    print(f"Model: {MODEL_ID}")
    print(f"Keep ratio: {args.ratio}")
    print(f"Device: {dev}")
    print(f"GPU: {torch.cuda.get_device_name(dev)}")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Step 1: Compute rank allocations (same as experiment)
    # ---------------------------------------------------------------
    print("\n[Step 1] Loading model and computing ranks...")
    model, tokenizer = load_model(MODEL_ID)
    model.eval()

    fisher = compute_fisher_proxy(model)
    svdllm_ranks = compute_all_svdllm_ranks(args.ratio)
    budget = param_cost(svdllm_ranks)

    strategies = {
        "svdllm": svdllm_ranks,
        "aligned_8": strategy_round_to_n(svdllm_ranks, fisher, budget, 8),
        "gac_dp": strategy_gac_dp(svdllm_ranks, fisher, budget, align=8, search_radius=3),
    }

    print(f"  SVD-LLM budget: {budget:,}")
    for name, ranks in strategies.items():
        n_aligned = sum(1 for r in ranks.values() if r % 8 == 0)
        print(f"  {name}: {n_aligned}/224 aligned, budget={param_cost(ranks):,}")

    # ---------------------------------------------------------------
    # Step 2: GEMM microbenchmark
    # ---------------------------------------------------------------
    print("\n[Step 2] GEMM microbenchmark...")
    print(f"  batch_tokens={args.batch_tokens}, warmup={args.gemm_warmup}, "
          f"repeats={args.gemm_repeats}")

    gemm_results = bench_gemm_shapes(
        strategies, dev,
        warmup=args.gemm_warmup,
        repeats=args.gemm_repeats,
        batch_tokens=args.batch_tokens,
    )

    with open(out_dir / "gemm_results.json", "w") as f:
        json.dump(gemm_results, f, indent=2)

    # ---------------------------------------------------------------
    # Step 3: End-to-end prefill latency
    # ---------------------------------------------------------------
    print("\n[Step 3] End-to-end prefill latency...")
    print(f"  seq_lens={seq_lens}, warmup={args.prefill_warmup}, "
          f"repeats={args.prefill_repeats}")

    # Load profiling matrices
    if args.profiling_path and os.path.exists(args.profiling_path):
        print(f"  Loading profiling from {args.profiling_path}")
        profiling_mat = torch.load(args.profiling_path, weights_only=False)
    else:
        print("  WARNING: No profiling path provided. Skipping end-to-end benchmark.")
        profiling_mat = None

    prefill_results = {}

    # Baseline: uncompressed model
    print("\n  Strategy: baseline (uncompressed)")
    m_base, _ = load_model(MODEL_ID)
    m_base.eval().half()
    prefill_results["baseline"] = bench_prefill_latency(
        m_base, tokenizer, dev, seq_lens,
        warmup=args.prefill_warmup,
        repeats=args.prefill_repeats,
    )
    del m_base
    torch.cuda.empty_cache()

    if profiling_mat is not None:
        for strat_name, ranks in strategies.items():
            print(f"\n  Strategy: {strat_name}")

            # Reload fresh model
            m, _ = load_model(MODEL_ID)
            m.eval()

            # Compress
            whitened_svd_compress(m, profiling_mat, ranks, dev)
            m = m.half()

            # Benchmark prefill
            prefill_results[strat_name] = bench_prefill_latency(
                m, tokenizer, dev, seq_lens,
                warmup=args.prefill_warmup,
                repeats=args.prefill_repeats,
            )

            del m
            torch.cuda.empty_cache()

    with open(out_dir / "prefill_results.json", "w") as f:
        json.dump(prefill_results, f, indent=2, default=str)

    # ---------------------------------------------------------------
    # Step 4: Decode latency (token generation)
    # ---------------------------------------------------------------
    print("\n[Step 4] Decode latency (token generation)...")
    print(f"  prompt_len={args.decode_prompt_len}, gen_tokens={args.decode_gen_tokens}")

    decode_results = {}

    # Baseline
    print("\n  Strategy: baseline (uncompressed)")
    m_base, _ = load_model(MODEL_ID)
    m_base.eval().half()
    decode_results["baseline"] = bench_decode_latency(
        m_base, tokenizer, dev,
        prompt_len=args.decode_prompt_len,
        gen_tokens=args.decode_gen_tokens,
        warmup=args.decode_warmup,
        repeats=args.decode_repeats,
    )
    del m_base
    torch.cuda.empty_cache()

    if profiling_mat is not None:
        for strat_name, ranks in strategies.items():
            print(f"\n  Strategy: {strat_name}")

            m, _ = load_model(MODEL_ID)
            m.eval()
            whitened_svd_compress(m, profiling_mat, ranks, dev)
            m = m.half()

            decode_results[strat_name] = bench_decode_latency(
                m, tokenizer, dev,
                prompt_len=args.decode_prompt_len,
                gen_tokens=args.decode_gen_tokens,
                warmup=args.decode_warmup,
                repeats=args.decode_repeats,
            )

            del m
            torch.cuda.empty_cache()

    with open(out_dir / "decode_results.json", "w") as f:
        json.dump(decode_results, f, indent=2, default=str)

    # ---------------------------------------------------------------
    # Step 5: Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GEMM LATENCY SUMMARY (factorized V+U, batch=512)")
    print("=" * 70)

    # Aggregate GEMM: average total latency per strategy
    for strat_name, strat_res in gemm_results.items():
        attn_total = np.mean([v["total_mean_ms"] for v in strat_res.values()
                             if v["proj"] in ATTN_PROJS])
        mlp_total = np.mean([v["total_mean_ms"] for v in strat_res.values()
                            if v["proj"] in MLP_PROJS])
        aligned = all(v["aligned"] for v in strat_res.values())
        print(f"  {strat_name:<12}: attn={attn_total:.3f}ms  mlp={mlp_total:.3f}ms  "
              f"aligned={'yes' if aligned else 'no'}")

    if prefill_results:
        print(f"\nPREFILL LATENCY SUMMARY")
        print(f"{'Strategy':<12}", end="")
        for sl in seq_lens:
            print(f"  {'sl='+str(sl):>10}", end="")
        print()
        for strat_name, res in prefill_results.items():
            print(f"{strat_name:<12}", end="")
            for sl in seq_lens:
                print(f"  {res[sl]['mean_ms']:>9.2f}ms", end="")
            print()

        # Speedup vs svdllm
        if "svdllm" in prefill_results:
            print(f"\nSPEEDUP vs SVD-LLM (unaligned)")
            for strat_name, res in prefill_results.items():
                if strat_name == "svdllm":
                    continue
                print(f"{strat_name:<12}", end="")
                for sl in seq_lens:
                    base = prefill_results["svdllm"][sl]["mean_ms"]
                    curr = res[sl]["mean_ms"]
                    pct = (base - curr) / base * 100
                    print(f"  {pct:>+9.2f}%", end="")
                print()

    # Decode latency summary
    if decode_results:
        print(f"\nDECODE LATENCY (prompt={args.decode_prompt_len}, gen={args.decode_gen_tokens})")
        print(f"{'Strategy':<12}  {'Total':>10}  {'Per-Token':>10}  {'Tok/s':>8}")
        for strat_name, res in decode_results.items():
            print(f"{strat_name:<12}  {res['total_mean_ms']:>9.2f}ms  "
                  f"{res['per_token_ms']:>9.2f}ms  {res['tokens_per_sec']:>8.1f}")

    # ---------------------------------------------------------------
    # Step 6: Plots
    # ---------------------------------------------------------------
    print("\n[Step 6] Generating plots...")
    generate_latency_plots(gemm_results, prefill_results, out_dir)

    # Save combined results
    all_results = {
        "config": {
            "model": MODEL_ID,
            "ratio": args.ratio,
            "batch_tokens": args.batch_tokens,
            "gemm_warmup": args.gemm_warmup,
            "gemm_repeats": args.gemm_repeats,
            "prefill_warmup": args.prefill_warmup,
            "prefill_repeats": args.prefill_repeats,
            "seq_lens": seq_lens,
            "gpu": torch.cuda.get_device_name(dev),
        },
        "gemm": gemm_results,
        "prefill": prefill_results,
        "decode": decode_results,
    }
    with open(out_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {out_dir}/")
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
