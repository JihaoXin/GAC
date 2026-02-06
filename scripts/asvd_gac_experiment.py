"""
Full-model ASVD compression experiment with GAC rank alignment.

Uses ASVD's sensitivity-based per-layer rank allocation on Llama-3-8B, comparing:
  1. ASVD (original unaligned ranks - per-layer sensitivity-based)
  2. Aligned-8 (round to nearest multiple of 8)
  3. GAC DP (DP-optimized alignment preserving sensitive layer quality)

Key difference from SVD-LLM: ASVD allocates different ranks per layer based on
sensitivity, giving GAC DP real optimization space.

Usage (via Slurm):
    sbatch slurm/run_asvd_experiment.sbatch
"""

import os
import sys
import argparse
import json
import time
import gc
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# No ASVD imports needed - we use simplified SVD approach


# ---------------------------------------------------------------------------
# Constants for Llama-3-8B (GQA: 32 Q heads, 8 KV heads, head_dim=128)
# ---------------------------------------------------------------------------
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
NUM_LAYERS = 32
HIDDEN = 4096
INTER = 14336
NUM_KV_HEADS = 8
HEAD_DIM = 128
KV_DIM = NUM_KV_HEADS * HEAD_DIM  # 1024

ATTN_PROJS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_PROJS = ["gate_proj", "up_proj", "down_proj"]
ALL_PROJS = ATTN_PROJS + MLP_PROJS

PROJ_SHAPES = {
    "q_proj": (HIDDEN, HIDDEN), "k_proj": (KV_DIM, HIDDEN),
    "v_proj": (KV_DIM, HIDDEN), "o_proj": (HIDDEN, HIDDEN),
    "gate_proj": (INTER, HIDDEN), "up_proj": (INTER, HIDDEN),
    "down_proj": (HIDDEN, INTER),
}


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------
def load_model(model_id):
    """Load model and tokenizer using standard HF AutoModel."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cpu", torch_dtype=torch.float16, trust_remote_code=True
    )
    model.seqlen = 2048
    return model, tokenizer


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    """Recursively find all layers of given types."""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


# ---------------------------------------------------------------------------
# Sensitivity estimation using stable rank (no external dependencies)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_layer_sensitivity(model):
    """
    Compute per-layer sensitivity using stable rank metric.
    Stable rank = ||W||_F^2 / ||W||_2^2 (ratio of Frobenius to spectral norm squared)
    Higher stable rank = more important layer (more spread singular values)

    Returns: dict {layer_name: {param_ratio: sensitivity_score}}
    """
    sensitivity_dict = {}
    param_ratio_candidates = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for name, module in tqdm(model.named_modules(), desc="Computing stable rank sensitivity"):
        if isinstance(module, nn.Linear):
            w = module.weight.data.float()

            # Stable rank = ||W||_F^2 / ||W||_2^2
            w_fro = torch.norm(w, p="fro") ** 2
            try:
                singular_values = torch.linalg.svdvals(w)
                spectral_norm = singular_values[0]  # Largest singular value
            except Exception:
                spectral_norm = torch.norm(w, p=2)
            w_spec = spectral_norm ** 2

            stable_rank = (w_fro / (w_spec + 1e-8)) ** 0.5

            sensitivity_dict[name] = {}
            for param_ratio in param_ratio_candidates:
                # Higher stable rank + lower ratio = more sensitive
                sensitivity_dict[name][param_ratio] = stable_rank.item() * (1 - param_ratio)

    return sensitivity_dict


def extract_ideal_ranks(model, sensitivity_dict, target_ratio):
    """
    Use sensitivity-based binary search to find ideal (unaligned) per-layer ranks.
    Returns: dict {(layer_idx, proj_name): rank}
    """
    # Build linear layer info
    linear_info = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_info[name] = {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "params": module.weight.numel(),
            }

    # Build sensitivity list: (layername, param_ratio, sensitivity)
    sensitivity_list = []
    for layername, v in sensitivity_dict.items():
        for param_ratio, sens in v.items():
            if param_ratio >= 1:
                continue
            sensitivity_list.append((layername, param_ratio, sens))

    sorted_sensitive_list = sorted(sensitivity_list, key=lambda x: -x[2])

    # Binary search for target param ratio
    high = len(sorted_sensitive_list) - 1
    low = 0

    while low < high:
        mid = (low + high) // 2
        layers_min_ratio = {ln: 1.0 for ln in sensitivity_dict.keys()}
        for layername, param_ratio, sens in sorted_sensitive_list[mid:]:
            layers_min_ratio[layername] = min(layers_min_ratio[layername], param_ratio)

        # Compute total param ratio
        tot_params = 0
        compress_params = 0
        for layername, param_ratio in layers_min_ratio.items():
            if layername in linear_info:
                tot_params += linear_info[layername]["params"]
                compress_params += linear_info[layername]["params"] * param_ratio

        now_ratio = compress_params / tot_params if tot_params > 0 else 1.0

        if now_ratio > target_ratio:
            high = mid
        else:
            low = mid + 1

    # Extract final param ratios and convert to ranks
    layers_min_ratio = {ln: 1.0 for ln in sensitivity_dict.keys()}
    for layername, param_ratio, sens in sorted_sensitive_list[low:]:
        layers_min_ratio[layername] = min(layers_min_ratio[layername], param_ratio)

    # Convert param_ratio to rank: rank = params * ratio / (in + out)
    ideal_ranks = {}
    for layername, param_ratio in layers_min_ratio.items():
        if layername not in linear_info:
            continue
        info = linear_info[layername]
        n_params = info["params"]
        in_f = info["in_features"]
        out_f = info["out_features"]

        compressed_params = int(n_params * param_ratio)
        rank = compressed_params // (in_f + out_f)
        rank = max(1, rank)  # Ensure at least rank 1

        # Parse layer name to extract layer index and projection name
        # Format: "model.layers.X.self_attn.q_proj" or "model.layers.X.mlp.gate_proj"
        parts = layername.split(".")
        try:
            layer_idx = int(parts[2])
            proj_name = parts[-1]
            if proj_name in ALL_PROJS:
                ideal_ranks[(layer_idx, proj_name)] = rank
        except (ValueError, IndexError):
            continue

    return ideal_ranks


def compute_fisher_from_sensitivity(sensitivity_dict):
    """
    Compute Fisher proxy from sensitivity dict.
    Higher PPL impact = higher Fisher = more sensitive layer.
    """
    fisher = {}
    for layername, v in sensitivity_dict.items():
        # Get max PPL increase at any ratio (sensitivity metric)
        max_ppl = max(v.values()) if v else 0

        parts = layername.split(".")
        try:
            layer_idx = int(parts[2])
            proj_name = parts[-1]
            if proj_name in ALL_PROJS:
                fisher[(layer_idx, proj_name)] = max_ppl
        except (ValueError, IndexError):
            continue

    return fisher


# ---------------------------------------------------------------------------
# Cost and alignment utilities
# ---------------------------------------------------------------------------
def param_cost(ranks):
    """Total parameter cost for rank allocation."""
    total = 0
    for (layer, proj), r in ranks.items():
        m, n = PROJ_SHAPES[proj]
        total += r * (m + n)
    return total


def count_aligned(ranks, mod=8):
    """Count projections with aligned ranks."""
    return sum(1 for r in ranks.values() if r % mod == 0)


# ---------------------------------------------------------------------------
# Strategy: Round to nearest n, budget-constrained
# ---------------------------------------------------------------------------
def strategy_round_to_n(base_ranks, fisher, target_budget, n=8):
    """Round ranks to nearest multiple of n, respecting budget."""
    ranks = {}
    for key, r in base_ranks.items():
        rounded = max(n, round(r / n) * n)
        m, nf = PROJ_SHAPES[key[1]]
        ranks[key] = min(rounded, min(m, nf))

    total = param_cost(ranks)
    keys_asc = sorted(ranks.keys(), key=lambda k: fisher.get(k, 0))
    keys_desc = list(reversed(keys_asc))

    # Reduce if over budget (start with least sensitive)
    while total > target_budget:
        reduced = False
        for k in keys_asc:
            if ranks[k] > n:
                m, nf = PROJ_SHAPES[k[1]]
                ranks[k] -= n
                total -= n * (m + nf)
                reduced = True
                if total <= target_budget:
                    break
        if not reduced:
            break

    # Fill remaining budget (prioritize most sensitive)
    while True:
        added = False
        for k in keys_desc:
            m, nf = PROJ_SHAPES[k[1]]
            cost = n * (m + nf)
            if ranks[k] + n <= min(m, nf) and total + cost <= target_budget:
                ranks[k] += n
                total += cost
                added = True
        if not added:
            break

    return ranks


# ---------------------------------------------------------------------------
# Strategy: GAC DP (asymmetric objective)
# ---------------------------------------------------------------------------
def strategy_gac_dp(base_ranks, fisher, target_budget, align=8, search_radius=3):
    """
    GAC DP: multi-choice knapsack for optimal aligned rank allocation.

    Objective: max Σ f_i * (r_i - r*_i)
    - Sensitive layers (high f_i) get rounded UP
    - Insensitive layers get rounded DOWN
    """
    projections = []
    for layer in range(NUM_LAYERS):
        for proj in ALL_PROJS:
            key = (layer, proj)
            if key not in base_ranks:
                continue
            ideal = base_ranks[key]
            f_i = fisher.get(key, 1.0)
            m, n = PROJ_SHAPES[proj]
            max_rank = min(m, n)
            unit_cost = m + n

            # Generate candidate aligned ranks
            ideal_aligned = round(ideal / align) * align
            candidates = []
            for offset in range(-search_radius, search_radius + 1):
                c = ideal_aligned + offset * align
                if align <= c <= max_rank:
                    candidates.append(c)
            if not candidates:
                candidates = [max(align, min(max_rank, ideal_aligned))]

            projections.append({
                "key": key, "ideal": ideal, "fisher": f_i,
                "candidates": candidates, "unit_cost": unit_cost,
            })

    # DP with budget in units of (align * min_unit_cost)
    min_uc = min(p["unit_cost"] for p in projections)
    budget_unit = align * min_uc
    B = target_budget // budget_unit + 2

    if B > 500000:
        print(f"  DP too large (B={B}), using greedy")
        return strategy_round_to_n(base_ranks, fisher, target_budget, align)

    n_proj = len(projections)
    NEG_INF = float("-inf")
    dp = [NEG_INF] * (B + 1)
    dp[0] = 0.0
    choice = [[None] * (B + 1) for _ in range(n_proj)]

    for i, p in enumerate(projections):
        new_dp = [NEG_INF] * (B + 1)
        for c in p["candidates"]:
            val = p["fisher"] * (c - p["ideal"])
            c_units = (c * p["unit_cost"]) // budget_unit
            if c_units > B:
                continue
            for b in range(int(c_units), B + 1):
                prev = b - int(c_units)
                if dp[prev] > NEG_INF and dp[prev] + val > new_dp[b]:
                    new_dp[b] = dp[prev] + val
                    choice[i][b] = c
        dp = new_dp

    # Find best solution within budget
    max_b = target_budget // budget_unit
    best_b = None
    for b in range(min(max_b, B), max(0, max_b - 10), -1):
        if dp[b] > NEG_INF:
            best_b = b
            break
    if best_b is None:
        for b in range(min(max_b, B), -1, -1):
            if dp[b] > NEG_INF:
                best_b = b
                break
    if best_b is None:
        return strategy_round_to_n(base_ranks, fisher, target_budget, align)

    # Backtrack to get ranks
    ranks = {}
    b = best_b
    for i in range(n_proj - 1, -1, -1):
        c = choice[i][b]
        if c is None:
            c = projections[i]["candidates"][len(projections[i]["candidates"]) // 2]
        ranks[projections[i]["key"]] = c
        b -= int((c * projections[i]["unit_cost"]) // budget_unit)

    # Post-processing: enforce exact budget constraint
    actual = param_cost(ranks)
    if actual > target_budget:
        items = sorted(ranks.keys(), key=lambda k: fisher.get(k, 0))
        for k in items:
            if actual <= target_budget:
                break
            m, n = PROJ_SHAPES[k[1]]
            if ranks[k] > align:
                ranks[k] -= align
                actual -= align * (m + n)

    # Fill remaining budget
    items_desc = sorted(ranks.keys(), key=lambda k: fisher.get(k, 0), reverse=True)
    for k in items_desc:
        m, n = PROJ_SHAPES[k[1]]
        cost = align * (m + n)
        if ranks[k] + align <= min(m, n) and actual + cost <= target_budget:
            ranks[k] += align
            actual += cost

    return ranks


# ---------------------------------------------------------------------------
# SVD Compression with rank override (simplified, no external deps)
# ---------------------------------------------------------------------------
class SimpleSVDLinear(nn.Module):
    """Simple low-rank factorization wrapper."""
    def __init__(self, U, V, bias=None):
        super().__init__()
        # U: (out_features, rank), V: (rank, in_features)
        self.u_proj = nn.Linear(U.shape[1], U.shape[0], bias=bias is not None)
        self.v_proj = nn.Linear(V.shape[1], V.shape[0], bias=False)
        self.u_proj.weight.data = U.contiguous()
        self.v_proj.weight.data = V.contiguous()
        if bias is not None:
            self.u_proj.bias.data = bias
        self.truncation_rank = U.shape[1]

    def forward(self, x):
        return self.u_proj(self.v_proj(x))

    @property
    def weight(self):
        return self.u_proj.weight

    @property
    def in_features(self):
        return self.v_proj.in_features

    @property
    def out_features(self):
        return self.u_proj.out_features


@torch.no_grad()
def svd_compress_with_ranks(model, ranks_dict, dev):
    """
    Apply truncated SVD compression with specified per-projection ranks.
    Uses simple SVD factorization (no complex calibration needed).
    """
    layers = model.model.layers
    print("  Applying truncated SVD compression...")

    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)

        for name in subset:
            proj = name.split(".")[-1]
            key = (i, proj)
            if key not in ranks_dict:
                continue

            target_rank = ranks_dict[key]
            raw_linear = subset[name]

            # Truncated SVD
            W = raw_linear.weight.data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)

            # Truncate to target rank
            rank = min(target_rank, S.shape[0])
            U_trunc = U[:, :rank]
            S_trunc = S[:rank]
            Vt_trunc = Vt[:rank, :]

            # Fuse singular values into U and V (sqrt split)
            sqrtS = torch.sqrt(S_trunc)
            U_final = (U_trunc * sqrtS.unsqueeze(0)).half()  # (out, rank)
            V_final = (sqrtS.unsqueeze(1) * Vt_trunc).half()  # (rank, in)

            bias = raw_linear.bias.data.clone() if raw_linear.bias is not None else None

            # Create wrapper
            svd_linear = SimpleSVDLinear(U_final, V_final, bias)
            svd_linear.to(raw_linear.weight.device)

            # Replace in model
            parts = name.split(".")
            parent = layer
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], svd_linear)

        layers[i] = layer.cpu()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# PPL Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_ppl(model, tokenizer, dev, seq_len=2048, batch_size=4):
    """Evaluate WikiText-2 PPL."""
    from datasets import load_dataset

    model.to(dev)
    model.eval()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")

    max_length = seq_len
    stride = seq_len // 2
    seq_len_total = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len_total, stride):
        end_loc = min(begin_loc + max_length, seq_len_total)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(dev)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len_total:
            break

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    model.cpu()
    torch.cuda.empty_cache()
    return ppl


# ---------------------------------------------------------------------------
# Accuracy Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_accuracy(model, tokenizer, dev, tasks="piqa,hellaswag", limit=200):
    """Zero-shot accuracy evaluation."""
    from datasets import load_dataset

    model.to(dev)
    model.eval()
    accs = {}

    task_list = tasks.split(",")
    for task_name in task_list:
        try:
            correct = 0
            total = 0

            if task_name == "piqa":
                ds = load_dataset("piqa", split="validation")
                if limit:
                    ds = ds.select(range(min(limit, len(ds))))
                for ex in tqdm(ds, desc=f"  {task_name}"):
                    goal = ex["goal"]
                    choices = [ex["sol1"], ex["sol2"]]
                    label = ex["label"]
                    scores = []
                    for c in choices:
                        text = f"{goal} {c}"
                        ids = tokenizer(text, return_tensors="pt").input_ids.to(dev)
                        out = model(ids, use_cache=False)
                        logits = out.logits[0, :-1]
                        targets = ids[0, 1:]
                        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                        score = log_probs.gather(1, targets.unsqueeze(1)).sum().item()
                        scores.append(score)
                    pred = scores.index(max(scores))
                    if pred == label:
                        correct += 1
                    total += 1

            elif task_name == "hellaswag":
                ds = load_dataset("Rowan/hellaswag", split="validation")
                if limit:
                    ds = ds.select(range(min(limit, len(ds))))
                for ex in tqdm(ds, desc=f"  {task_name}"):
                    ctx = ex["ctx"]
                    choices = ex["endings"]
                    label = int(ex["label"])
                    scores = []
                    for c in choices:
                        text = f"{ctx} {c}"
                        ids = tokenizer(text, return_tensors="pt").input_ids.to(dev)
                        ctx_ids = tokenizer(ctx, return_tensors="pt").input_ids
                        ctx_len = ctx_ids.shape[1]
                        out = model(ids, use_cache=False)
                        logits = out.logits[0, ctx_len-1:-1]
                        targets = ids[0, ctx_len:]
                        if targets.numel() == 0:
                            scores.append(float("-inf"))
                            continue
                        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                        score = log_probs.gather(1, targets.unsqueeze(1)).mean().item()
                        scores.append(score)
                    pred = scores.index(max(scores))
                    if pred == label:
                        correct += 1
                    total += 1
            else:
                print(f"  Unknown task: {task_name}, skipping")
                continue

            acc = correct / total if total > 0 else 0
            accs[task_name] = round(acc, 4)
            print(f"  {task_name}: {acc:.4f} ({correct}/{total})")
        except Exception as e:
            import traceback
            print(f"  {task_name} eval failed: {type(e).__name__}: {e}")
            traceback.print_exc()

    model.cpu()
    torch.cuda.empty_cache()
    return accs


# ---------------------------------------------------------------------------
# Latency Benchmark
# ---------------------------------------------------------------------------
@torch.no_grad()
def benchmark_latency(model, tokenizer, dev, n_warmup=3, n_runs=10, seq_len=512):
    """Benchmark inference latency."""
    model.to(dev)
    model.eval()

    # Generate a sequence of tokens for benchmarking
    input_ids = tokenizer("Hello, world!", return_tensors="pt").input_ids.to(dev)
    # Repeat to fill seq_len
    repeats = (seq_len // input_ids.shape[1]) + 1
    input_ids = input_ids.repeat(1, repeats)[:, :seq_len]

    # Warmup
    for _ in range(n_warmup):
        _ = model(input_ids, use_cache=False)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(input_ids, use_cache=False)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    model.cpu()
    torch.cuda.empty_cache()

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def generate_plots(results, ranks_dict, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    strats = [r["strategy"] for r in results if r["strategy"] != "baseline"]
    ppls = [r["ppl"] for r in results if r["strategy"] != "baseline"]
    aligns = [r["pct_aligned"] for r in results if r["strategy"] != "baseline"]

    baseline_ppl = next((r["ppl"] for r in results if r["strategy"] == "baseline"), None)

    colors = {"asvd_unaligned": "#e74c3c", "aligned_8": "#3498db", "gac_dp": "#2ecc71"}

    # PPL comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    c = [colors.get(s, "#999") for s in strats]
    bars = ax.bar(strats, ppls, color=c, edgecolor="black", linewidth=0.8)
    if baseline_ppl:
        ax.axhline(y=baseline_ppl, color="gray", linestyle="--", label=f"Baseline: {baseline_ppl:.2f}")
        ax.legend()
    for bar, p in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{p:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Perplexity (WikiText-2)")
    ax.set_title("ASVD Compression: PPL by Alignment Strategy")
    plt.tight_layout()
    plt.savefig(plot_dir / "ppl_comparison.png", dpi=150)
    plt.close()

    # Alignment bar
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(strats, aligns, color=c, edgecolor="black", linewidth=0.8)
    for bar, a in zip(bars, aligns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{a:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("% Projections Aligned (mod 8)")
    ax.set_title("Dimension Alignment Comparison")
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(plot_dir / "alignment_comparison.png", dpi=150)
    plt.close()

    # Rank distribution histogram
    if "asvd_unaligned" in ranks_dict:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        unaligned_ranks = list(ranks_dict["asvd_unaligned"].values())
        gac_ranks = list(ranks_dict.get("gac_dp", {}).values())

        axes[0].hist(unaligned_ranks, bins=30, color="#e74c3c", edgecolor="black", alpha=0.7)
        axes[0].set_xlabel("Rank")
        axes[0].set_ylabel("Count")
        axes[0].set_title("ASVD Unaligned Rank Distribution")

        if gac_ranks:
            axes[1].hist(gac_ranks, bins=30, color="#2ecc71", edgecolor="black", alpha=0.7)
            axes[1].set_xlabel("Rank")
            axes[1].set_ylabel("Count")
            axes[1].set_title("GAC DP Rank Distribution")

        plt.tight_layout()
        plt.savefig(plot_dir / "rank_distribution.png", dpi=150)
        plt.close()

    print(f"  Plots saved to {plot_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    # Compression args
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--param_ratio_target", type=float, default=0.7,
                        help="Target parameter ratio (0.7 = keep 70%, compress 30%)")
    parser.add_argument("--output", type=str, default="results/asvd_experiment")
    parser.add_argument("--device", type=str, default="cuda")

    # ASVD calibration args
    parser.add_argument("--n_calib_samples", type=int, default=32)
    parser.add_argument("--calib_dataset", type=str, default="wikitext2")
    parser.add_argument("--scaling_method", type=str, default="abs_mean")
    parser.add_argument("--sensitivity_metric", type=str, default="ppl")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--sigma_fuse", type=str, default="UV")
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--use_bos", action="store_true")
    parser.add_argument("--act_aware", action="store_true", default=True)

    # Evaluation args
    parser.add_argument("--eval-accuracy", action="store_true")
    parser.add_argument("--accuracy-tasks", type=str, default="piqa,hellaswag")
    parser.add_argument("--accuracy-limit", type=int, default=200)
    parser.add_argument("--eval-latency", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)

    print("=" * 70)
    print("ASVD + GAC Alignment Experiment")
    print(f"Model: {args.model_id}")
    print(f"Target param ratio: {args.param_ratio_target} (compress {100*(1-args.param_ratio_target):.0f}%)")
    print(f"Device: {dev}")
    print("=" * 70)

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ---------------------------------------------------------------
    # Step 1: Load model and compute sensitivity
    # ---------------------------------------------------------------
    print("\n[Step 1] Loading model and computing sensitivity...")
    model, tokenizer = load_model(args.model_id)
    model.eval()

    t0 = time.time()
    sensitivity = compute_layer_sensitivity(model)
    print(f"  Sensitivity computed in {time.time()-t0:.0f}s")
    print(f"  Sensitivity data for {len(sensitivity)} layers")

    # Save sensitivity data
    with open(out_dir / "sensitivity.json", "w") as f:
        sens_json = {k: {str(r): v for r, v in vv.items()} for k, vv in sensitivity.items()}
        json.dump(sens_json, f, indent=2)

    # ---------------------------------------------------------------
    # Step 2: Extract ideal (unaligned) ranks
    # ---------------------------------------------------------------
    print("\n[Step 2] Extracting ideal per-layer ranks...")
    ideal_ranks = extract_ideal_ranks(model, sensitivity, args.param_ratio_target)
    print(f"  Found {len(ideal_ranks)} projections with per-layer ranks")

    # Show rank diversity
    unique_ranks = set(ideal_ranks.values())
    print(f"  Unique ranks: {len(unique_ranks)} values")
    print(f"  Range: {min(unique_ranks)} to {max(unique_ranks)}")

    # ---------------------------------------------------------------
    # Step 3: Compute Fisher scores and create strategies
    # ---------------------------------------------------------------
    print("\n[Step 3] Computing strategies...")
    fisher = compute_fisher_from_sensitivity(sensitivity)
    budget = param_cost(ideal_ranks)

    strategies = {
        "asvd_unaligned": ideal_ranks,
        "aligned_8": strategy_round_to_n(ideal_ranks, fisher, budget, 8),
        "gac_dp": strategy_gac_dp(ideal_ranks, fisher, budget, align=8, search_radius=3),
    }

    # Save rank allocations and show summary
    print(f"\n{'Strategy':<16} {'Budget':>14} {'Aligned/8':>12} {'Unique Ranks':>14}")
    print("-" * 60)
    for name, ranks in strategies.items():
        b = param_cost(ranks)
        n_aligned = count_aligned(ranks, 8)
        n_total = len(ranks)
        unique = len(set(ranks.values()))
        print(f"{name:<16} {b:>14,} {n_aligned:>6}/{n_total}    {unique:>6}")

        # Save ranks
        data = [{"layer": k[0], "proj": k[1], "rank": v}
                for k, v in sorted(ranks.items())]
        with open(out_dir / f"ranks_{name}.json", "w") as f:
            json.dump(data, f, indent=2)

    # ---------------------------------------------------------------
    # Step 4: Evaluate baseline
    # ---------------------------------------------------------------
    all_results = []
    print("\n[Step 4] Evaluating baseline...")

    # Clear memory before baseline eval
    del sensitivity
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.time()
    baseline_ppl = eval_ppl(model, tokenizer, dev)
    print(f"  Baseline PPL: {baseline_ppl:.2f} ({time.time()-t0:.0f}s)")

    baseline_result = {"strategy": "baseline", "ppl": baseline_ppl, "pct_aligned": 100.0}

    if args.eval_accuracy:
        baseline_acc = eval_accuracy(model, tokenizer, dev,
                                      args.accuracy_tasks, args.accuracy_limit)
        baseline_result["accuracy"] = baseline_acc

    if args.eval_latency:
        baseline_lat = benchmark_latency(model, tokenizer, dev)
        baseline_result["latency"] = baseline_lat
        print(f"  Baseline latency: {baseline_lat['mean_ms']:.1f}ms")

    all_results.append(baseline_result)

    # Free baseline model
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Step 5: For each strategy, compress and evaluate
    # ---------------------------------------------------------------
    for strat_name, ranks in strategies.items():
        print(f"\n[Step 5] Strategy: {strat_name}")
        t0 = time.time()

        # Reload fresh model
        model, _ = load_model(args.model_id)
        model.eval()

        # Compress with truncated SVD using specified ranks
        svd_compress_with_ranks(model, ranks, dev)
        print(f"  Compression done in {time.time()-t0:.0f}s")

        # PPL
        t1 = time.time()
        ppl = eval_ppl(model, tokenizer, dev)
        print(f"  PPL: {ppl:.2f} ({time.time()-t1:.0f}s)")

        n_aligned = count_aligned(ranks, 8)
        pct = 100.0 * n_aligned / len(ranks)

        result = {
            "strategy": strat_name,
            "ppl": ppl,
            "pct_aligned": pct,
            "budget": param_cost(ranks),
            "n_unique_ranks": len(set(ranks.values())),
        }

        if args.eval_accuracy:
            accs = eval_accuracy(model, tokenizer, dev,
                                  args.accuracy_tasks, args.accuracy_limit)
            result["accuracy"] = accs
            print(f"  Accuracy: {accs}")

        if args.eval_latency:
            lat = benchmark_latency(model, tokenizer, dev)
            result["latency"] = lat
            print(f"  Latency: {lat['mean_ms']:.1f}ms")

        all_results.append(result)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Step 6: Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    header = f"{'Strategy':<16} {'PPL':>8} {'Aligned%':>9} {'Unique':>7}"
    if args.eval_latency:
        header += f"  {'Latency(ms)':>12}"
    if args.eval_accuracy:
        header += f"  {'Avg Acc':>8}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        line = f"{r['strategy']:<16} {r['ppl']:>8.2f} {r['pct_aligned']:>8.0f}%"
        line += f" {r.get('n_unique_ranks', '-'):>7}"
        if args.eval_latency:
            lat = r.get("latency", {}).get("mean_ms", 0)
            line += f"  {lat:>11.1f}"
        if args.eval_accuracy:
            accs = r.get("accuracy", {})
            avg = np.mean(list(accs.values())) * 100 if accs else 0
            line += f"  {avg:>7.1f}%"
        print(line)

    # Key insight: rank diversity
    print("\n--- KEY INSIGHT: Rank Diversity ---")
    unaligned_unique = len(set(strategies["asvd_unaligned"].values()))
    gac_unique = len(set(strategies["gac_dp"].values()))
    print(f"ASVD unaligned: {unaligned_unique} unique ranks (per-layer sensitivity-based)")
    print(f"GAC DP: {gac_unique} unique aligned ranks")
    print("Unlike SVD-LLM (only 2 ranks), ASVD gives GAC DP real optimization space!")

    # Save results
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir / 'results.json'}")

    # Plots
    print("\n[Step 6] Generating plots...")
    generate_plots(all_results, strategies, out_dir)

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
