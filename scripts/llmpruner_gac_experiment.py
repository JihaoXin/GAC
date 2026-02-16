"""
LLM-Pruner + GAC Alignment Experiment on Llama-3-8B.

Demonstrates dimensional collapse from structured pruning:
  1. Global pruning produces non-uniform MLP widths per layer
  2. Without alignment, MLP dimensions can be non-multiples-of-8
  3. round_to=8 restores Tensor Core alignment
  4. round_to=16 additionally aligns to L2 cache sectors (32 bytes = 16 fp16)

Strategies:
  - baseline: unpruned Llama-3-8B
  - pruned: block-wise global Taylor pruning (no rounding)
  - pruned_r8: same pruning with round_to=8
  - pruned_r16: same pruning with round_to=16 (L2 sector aligned)

Metrics: alignment%, PPL (WikiText-2), accuracy (piqa, hellaswag), prefill latency.

Usage (via Slurm):
    sbatch slurm/run_llmpruner_llama3.sbatch
"""

import os
import sys
import gc
import copy
import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm

# Add LLM-Pruner to path
PRUNER_DIR = Path(__file__).parent.parent / "third_party" / "LLM-Pruner"
sys.path.insert(0, str(PRUNER_DIR))
import LLMPruner.torch_pruning as tp
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.datasets.example_samples import get_examples

# Add SVD-LLM to path for data utils
SVDLLM_DIR = Path(__file__).parent.parent / "third_party" / "SVD-LLM"
sys.path.insert(0, str(SVDLLM_DIR))
from utils.data_utils import get_test_data

# ---------------------------------------------------------------------------
# Constants for Llama-3-8B
# ---------------------------------------------------------------------------
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
NUM_LAYERS = 32


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = LlamaForCausalLM.from_pretrained(
        MODEL_ID, device_map="cpu", torch_dtype=torch.float16, trust_remote_code=True
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------
def prune_model(model, tokenizer, pruning_ratio, device, round_to=None,
                global_pruning=True, num_examples=10, seed=42):
    """
    Run LLM-Pruner block-wise pruning with Taylor importance.

    - Attention layers 3-31: pruned via k_proj root (consecutive_groups=head_dim)
    - MLP layers 3-31: pruned via gate_proj root
    - Global pruning: importance ranked across all layers → non-uniform widths
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model.to(device)

    for param in model.parameters():
        param.requires_grad_(True)

    before_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Dummy forward input for dependency graph (any valid tokens work)
    forward_prompts = torch.tensor([
        [128000, 791, 4320, 374, 220, 16, 13, 220, 17],
    ]).to(device)

    imp = llama_pruner.TaylorImportance(
        group_reduction='sum', taylor='param_first'
    )

    kwargs = {
        "importance": imp,
        "global_pruning": global_pruning,
        "iterative_steps": 1,
        "ch_sparsity": pruning_ratio,
        "ignored_layers": [],
        "channel_groups": {},
        "consecutive_groups": {
            layer.self_attn.k_proj: layer.self_attn.head_dim
            for layer in model.model.layers
        },
        "customized_pruners": {
            LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
        },
        "root_module_types": None,
        "root_instances": (
            [model.model.layers[i].self_attn.k_proj for i in range(3, 31)] +
            [model.model.layers[i].mlp.gate_proj for i in range(3, 31)]
        ),
    }
    if round_to is not None:
        kwargs["round_to"] = round_to

    pruner = tp.pruner.MetaPruner(
        model, forward_prompts, **kwargs,
        output_transform=lambda out: out.logits if hasattr(out, 'logits') else out[0],
    )
    model.zero_grad()

    # Compute Taylor importance via backward pass
    print("    Computing Taylor importance...")
    example_prompts = get_examples('c4', tokenizer, num_examples, seq_len=64)
    example_prompts = example_prompts.to(device)
    loss = model(example_prompts, labels=example_prompts).loss
    print(f"    Loss = {loss.item():.4f}")
    loss.backward()

    print("    Pruning step...")
    pruner.step()

    after_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Params: {before_params:,} → {after_params:,} "
          f"({100.0 * after_params / before_params:.1f}%)")

    # Update inference-related attributes per layer
    for layer in model.model.layers:
        layer.self_attn.num_heads = (
            layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
        )
        layer.self_attn.num_key_value_heads = (
            layer.self_attn.k_proj.weight.data.shape[0] // layer.self_attn.head_dim
        )

    # Clean gradients
    model.zero_grad()
    for name, module in model.named_parameters():
        if 'weight' in name:
            module.grad = None
    del pruner

    gc.collect()
    torch.cuda.empty_cache()
    return model


# ---------------------------------------------------------------------------
# Dimension analysis
# ---------------------------------------------------------------------------
def analyze_dimensions(model):
    """Inspect per-layer projection dimensions and alignment."""
    layer_dims = []
    all_dims = []

    for i, layer in enumerate(model.model.layers):
        info = {"layer": i}
        # Attention
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            proj = getattr(layer.self_attn, name)
            info[f"{name}_out"] = proj.out_features
            info[f"{name}_in"] = proj.in_features
            all_dims.append(proj.out_features)
            all_dims.append(proj.in_features)
        # MLP
        for name in ["gate_proj", "up_proj", "down_proj"]:
            proj = getattr(layer.mlp, name)
            info[f"{name}_out"] = proj.out_features
            info[f"{name}_in"] = proj.in_features
            all_dims.append(proj.out_features)
            all_dims.append(proj.in_features)
        layer_dims.append(info)

    n_total = len(all_dims)
    n_aligned_8 = sum(1 for d in all_dims if d % 8 == 0)
    n_aligned_16 = sum(1 for d in all_dims if d % 16 == 0)
    pct_aligned_8 = 100.0 * n_aligned_8 / n_total if n_total > 0 else 0
    pct_aligned_16 = 100.0 * n_aligned_16 / n_total if n_total > 0 else 0

    # Unique MLP intermediate sizes (most likely source of misalignment)
    mlp_sizes = set()
    for info in layer_dims:
        mlp_sizes.add(info["gate_proj_out"])

    return layer_dims, pct_aligned_8, pct_aligned_16, n_aligned_8, n_aligned_16, n_total, sorted(mlp_sizes)


# ---------------------------------------------------------------------------
# PPL Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_ppl(model, tokenizer, dev, seq_len=2048, batch_size=4):
    """Evaluate WikiText-2 PPL."""
    model.to(dev)
    model.eval()
    test_loader = get_test_data("wikitext2", tokenizer, seq_len=seq_len, batch_size=batch_size)
    nlls = []
    for batch in tqdm(test_loader, desc="PPL eval"):
        batch = batch.to(dev)
        output = model(batch, use_cache=False)
        lm_logits = output.logits
        if torch.isfinite(lm_logits).all():
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            nlls.append(loss)
        else:
            print("  Warning: non-finite logits detected, skipping batch")
    if not nlls:
        return float("nan")
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    model.cpu()
    torch.cuda.empty_cache()
    return ppl


# ---------------------------------------------------------------------------
# Accuracy Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_accuracy(model, tokenizer, dev, tasks="piqa,hellaswag", limit=200):
    """Zero-shot accuracy evaluation using log-likelihood scoring."""
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
# Latency measurement
# ---------------------------------------------------------------------------
def measure_latency(fn, warmup=10, repeats=30, device="cuda"):
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


def bench_prefill_latency(model, tokenizer, dev, seq_lens=[128, 256, 512, 1024],
                           warmup=5, repeats=20):
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
def generate_plots(results, dim_data, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    colors = {
        "baseline": "#95a5a6", "pruned": "#e74c3c",
        "pruned_r8": "#3498db", "pruned_r16": "#2ecc71",
    }

    # --- PPL comparison ---
    strats = [r["strategy"] for r in results]
    ppls = [r["ppl"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    c = [colors.get(s, "#999") for s in strats]
    bars = ax.bar(strats, ppls, color=c, edgecolor="black", linewidth=0.8)
    for bar, p in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{p:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Perplexity (WikiText-2)")
    ax.set_title("LLM-Pruner: PPL by Alignment Strategy")
    plt.tight_layout()
    plt.savefig(plot_dir / "ppl_comparison.png", dpi=150)
    plt.close()

    # --- Alignment comparison ---
    aligns = [r["pct_aligned"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(strats, aligns, color=c, edgecolor="black", linewidth=0.8)
    for bar, a in zip(bars, aligns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{a:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("% Dimensions Aligned (mod 8)")
    ax.set_title("Dimension Alignment: LLM-Pruner Strategies")
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(plot_dir / "alignment_comparison.png", dpi=150)
    plt.close()

    # --- MLP dimension distribution per layer ---
    for strat_name, dims in dim_data.items():
        if not dims:
            continue
        fig, ax = plt.subplots(figsize=(14, 5))
        layers_idx = [d["layer"] for d in dims]
        gate_outs = [d["gate_proj_out"] for d in dims]
        q_outs = [d["q_proj_out"] for d in dims]

        ax.bar(np.array(layers_idx) - 0.2, gate_outs, 0.4,
               label="MLP (gate_proj out)", color="#e74c3c", alpha=0.8)
        ax.bar(np.array(layers_idx) + 0.2, q_outs, 0.4,
               label="Attn (q_proj out)", color="#3498db", alpha=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Dimension")
        ax.set_title(f"Per-Layer Dimensions: {strat_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / f"dims_{strat_name}.png", dpi=150)
        plt.close()

    # --- Prefill latency ---
    latency_data = {r["strategy"]: r.get("prefill_latency", {}) for r in results}
    latency_data = {k: v for k, v in latency_data.items() if v}
    if latency_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        for strat, lat in latency_data.items():
            sls = sorted(int(k) for k in lat.keys())
            means = [lat[k]["mean_ms"] if isinstance(lat[k], dict) else lat[k]
                     for k in [str(s) if str(s) in lat else s for s in sls]]
            # Handle both int and str keys
            means_clean = []
            for sl in sls:
                entry = lat.get(sl, lat.get(str(sl), {}))
                if isinstance(entry, dict):
                    means_clean.append(entry["mean_ms"])
                else:
                    means_clean.append(entry)
            ax.plot(sls, means_clean, "o-", label=strat,
                    color=colors.get(strat, "#999"), linewidth=2, markersize=6)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Prefill Latency (ms)")
        ax.set_title("LLM-Pruner: Prefill Latency by Strategy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / "prefill_latency.png", dpi=150)
        plt.close()

    print(f"  Plots saved to {plot_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pruning-ratio", type=float, default=0.25,
                        help="Channel pruning ratio (0.25 = remove 25%%)")
    parser.add_argument("--output", type=str, default="results/llmpruner_experiment")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-accuracy", action="store_true")
    parser.add_argument("--accuracy-tasks", type=str, default="piqa,hellaswag")
    parser.add_argument("--accuracy-limit", type=int, default=200)
    parser.add_argument("--eval-latency", action="store_true")
    parser.add_argument("--seq-lens", type=str, default="128,256,512,1024")
    parser.add_argument("--prefill-warmup", type=int, default=5)
    parser.add_argument("--prefill-repeats", type=int, default=30)
    parser.add_argument("--decode-prompt-len", type=int, default=128)
    parser.add_argument("--decode-gen-tokens", type=int, default=64)
    parser.add_argument("--decode-warmup", type=int, default=3)
    parser.add_argument("--decode-repeats", type=int, default=10)
    parser.add_argument("--skip-decode", action="store_true",
                        help="Skip decode latency benchmark (prefill-only runs)")
    parser.add_argument("--skip-ppl", action="store_true",
                        help="Skip perplexity evaluation (latency-only runs)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 70)
    print("LLM-Pruner + GAC Alignment Experiment")
    print(f"Model: {MODEL_ID}")
    print(f"Pruning ratio: {args.pruning_ratio}")
    print(f"Device: {dev}")
    print("=" * 70)

    all_results = []
    dim_data = {}

    # ---------------------------------------------------------------
    # Step 1: Baseline evaluation
    # ---------------------------------------------------------------
    print("\n[Step 1] Evaluating baseline (unpruned)...")
    model, tokenizer = load_model()
    model.eval()

    # Dimension analysis for baseline
    dims_bl, pct8_bl, pct16_bl, na8_bl, na16_bl, nt_bl, mlp_sizes_bl = analyze_dimensions(model)
    dim_data["baseline"] = dims_bl
    print(f"  Alignment mod8: {na8_bl}/{nt_bl} ({pct8_bl:.1f}%)")
    print(f"  Alignment mod16: {na16_bl}/{nt_bl} ({pct16_bl:.1f}%)")
    print(f"  MLP sizes: {mlp_sizes_bl}")

    # PPL
    if args.skip_ppl:
        baseline_ppl = float("nan")
        print("  Baseline PPL: skipped")
    else:
        model = model.float()
        t0 = time.time()
        baseline_ppl = eval_ppl(model, tokenizer, dev)
        print(f"  Baseline PPL: {baseline_ppl:.2f} ({time.time()-t0:.0f}s)")

    result_bl = {
        "strategy": "baseline",
        "ppl": baseline_ppl,
        "pct_aligned": pct8_bl,
        "pct_aligned_16": pct16_bl,
        "n_params": sum(p.numel() for p in model.parameters()),
    }

    if args.eval_accuracy:
        model = model.half()
        accs = eval_accuracy(model, tokenizer, dev,
                              args.accuracy_tasks, args.accuracy_limit)
        result_bl["accuracy"] = accs

    if args.eval_latency:
        model = model.half()
        print("  Prefill latency (baseline):")
        lat = bench_prefill_latency(model, tokenizer, dev, seq_lens,
                                     args.prefill_warmup, args.prefill_repeats)
        result_bl["prefill_latency"] = lat

        if not args.skip_decode:
            print("  Decode latency (baseline):")
            dec = bench_decode_latency(model, tokenizer, dev,
                                        args.decode_prompt_len, args.decode_gen_tokens,
                                        args.decode_warmup, args.decode_repeats)
            result_bl["decode_latency"] = dec

    all_results.append(result_bl)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Step 2: Pruned (no rounding)
    # ---------------------------------------------------------------
    print("\n[Step 2] Pruning WITHOUT round_to...")
    model, tokenizer = load_model()
    model = prune_model(model, tokenizer, args.pruning_ratio, dev, round_to=None)

    dims_p, pct8_p, pct16_p, na8_p, na16_p, nt_p, mlp_sizes_p = analyze_dimensions(model)
    dim_data["pruned"] = dims_p
    print(f"  Alignment mod8: {na8_p}/{nt_p} ({pct8_p:.1f}%)")
    print(f"  Alignment mod16: {na16_p}/{nt_p} ({pct16_p:.1f}%)")
    print(f"  Unique MLP sizes: {mlp_sizes_p}")
    n_params_p = sum(p.numel() for p in model.parameters())

    # PPL
    if args.skip_ppl:
        ppl_p = float("nan")
        print("  PPL: skipped")
    else:
        model = model.float()
        t0 = time.time()
        ppl_p = eval_ppl(model, tokenizer, dev)
        print(f"  PPL: {ppl_p:.2f} ({time.time()-t0:.0f}s)")

    result_p = {
        "strategy": "pruned",
        "ppl": ppl_p,
        "pct_aligned": pct8_p,
        "pct_aligned_16": pct16_p,
        "n_params": n_params_p,
        "mlp_sizes": mlp_sizes_p,
    }

    if args.eval_accuracy:
        model = model.half()
        accs = eval_accuracy(model, tokenizer, dev,
                              args.accuracy_tasks, args.accuracy_limit)
        result_p["accuracy"] = accs

    if args.eval_latency:
        model = model.half()
        print("  Prefill latency (pruned):")
        lat = bench_prefill_latency(model, tokenizer, dev, seq_lens,
                                     args.prefill_warmup, args.prefill_repeats)
        result_p["prefill_latency"] = lat

        if not args.skip_decode:
            print("  Decode latency (pruned):")
            dec = bench_decode_latency(model, tokenizer, dev,
                                        args.decode_prompt_len, args.decode_gen_tokens,
                                        args.decode_warmup, args.decode_repeats)
            result_p["decode_latency"] = dec

    all_results.append(result_p)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Step 3: Pruned with round_to=8
    # ---------------------------------------------------------------
    print("\n[Step 3] Pruning WITH round_to=8...")
    model, tokenizer = load_model()
    model = prune_model(model, tokenizer, args.pruning_ratio, dev, round_to=8)

    dims_r8, pct8_r8, pct16_r8, na8_r8, na16_r8, nt_r8, mlp_sizes_r8 = analyze_dimensions(model)
    dim_data["pruned_r8"] = dims_r8
    print(f"  Alignment mod8: {na8_r8}/{nt_r8} ({pct8_r8:.1f}%)")
    print(f"  Alignment mod16: {na16_r8}/{nt_r8} ({pct16_r8:.1f}%)")
    print(f"  Unique MLP sizes: {mlp_sizes_r8}")
    n_params_r8 = sum(p.numel() for p in model.parameters())

    # PPL
    if args.skip_ppl:
        ppl_r8 = float("nan")
        print("  PPL: skipped")
    else:
        model = model.float()
        t0 = time.time()
        ppl_r8 = eval_ppl(model, tokenizer, dev)
        print(f"  PPL: {ppl_r8:.2f} ({time.time()-t0:.0f}s)")

    result_r8 = {
        "strategy": "pruned_r8",
        "ppl": ppl_r8,
        "pct_aligned": pct8_r8,
        "pct_aligned_16": pct16_r8,
        "n_params": n_params_r8,
        "mlp_sizes": mlp_sizes_r8,
    }

    if args.eval_accuracy:
        model = model.half()
        accs = eval_accuracy(model, tokenizer, dev,
                              args.accuracy_tasks, args.accuracy_limit)
        result_r8["accuracy"] = accs

    if args.eval_latency:
        model = model.half()
        print("  Prefill latency (pruned_r8):")
        lat = bench_prefill_latency(model, tokenizer, dev, seq_lens,
                                     args.prefill_warmup, args.prefill_repeats)
        result_r8["prefill_latency"] = lat

        if not args.skip_decode:
            print("  Decode latency (pruned_r8):")
            dec = bench_decode_latency(model, tokenizer, dev,
                                        args.decode_prompt_len, args.decode_gen_tokens,
                                        args.decode_warmup, args.decode_repeats)
            result_r8["decode_latency"] = dec

    all_results.append(result_r8)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Step 4: Pruned with round_to=16 (L2 sector alignment)
    # ---------------------------------------------------------------
    print("\n[Step 4] Pruning WITH round_to=16 (L2 sector aligned)...")
    model, tokenizer = load_model()
    model = prune_model(model, tokenizer, args.pruning_ratio, dev, round_to=16)

    dims_r16, pct8_r16, pct16_r16, na8_r16, na16_r16, nt_r16, mlp_sizes_r16 = analyze_dimensions(model)
    dim_data["pruned_r16"] = dims_r16
    print(f"  Alignment mod8: {na8_r16}/{nt_r16} ({pct8_r16:.1f}%)")
    print(f"  Alignment mod16: {na16_r16}/{nt_r16} ({pct16_r16:.1f}%)")
    print(f"  Unique MLP sizes: {mlp_sizes_r16}")
    n_params_r16 = sum(p.numel() for p in model.parameters())

    # PPL
    if args.skip_ppl:
        ppl_r16 = float("nan")
        print("  PPL: skipped")
    else:
        model = model.float()
        t0 = time.time()
        ppl_r16 = eval_ppl(model, tokenizer, dev)
        print(f"  PPL: {ppl_r16:.2f} ({time.time()-t0:.0f}s)")

    result_r16 = {
        "strategy": "pruned_r16",
        "ppl": ppl_r16,
        "pct_aligned": pct8_r16,
        "pct_aligned_16": pct16_r16,
        "n_params": n_params_r16,
        "mlp_sizes": mlp_sizes_r16,
    }

    if args.eval_accuracy:
        model = model.half()
        accs = eval_accuracy(model, tokenizer, dev,
                              args.accuracy_tasks, args.accuracy_limit)
        result_r16["accuracy"] = accs

    if args.eval_latency:
        model = model.half()
        print("  Prefill latency (pruned_r16):")
        lat = bench_prefill_latency(model, tokenizer, dev, seq_lens,
                                     args.prefill_warmup, args.prefill_repeats)
        result_r16["prefill_latency"] = lat

        if not args.skip_decode:
            print("  Decode latency (pruned_r16):")
            dec = bench_decode_latency(model, tokenizer, dev,
                                        args.decode_prompt_len, args.decode_gen_tokens,
                                        args.decode_warmup, args.decode_repeats)
            result_r16["decode_latency"] = dec

    all_results.append(result_r16)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    header = f"{'Strategy':<14} {'PPL':>8} {'Mod8%':>7} {'Mod16%':>7} {'#Params':>14}"
    if args.eval_accuracy:
        header += f"  {'Avg Acc':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        line = (f"{r['strategy']:<14} {r['ppl']:>8.2f} {r['pct_aligned']:>6.1f}% "
                f"{r.get('pct_aligned_16', 0):>6.1f}% "
                f"{r.get('n_params', 0):>14,}")
        if args.eval_accuracy:
            accs = r.get("accuracy", {})
            avg = np.mean(list(accs.values())) * 100 if accs else 0
            line += f"  {avg:>7.1f}%"
        print(line)

    if args.eval_latency:
        print(f"\nPREFILL LATENCY")
        print(f"{'Strategy':<14}", end="")
        for sl in seq_lens:
            print(f"  {'sl='+str(sl):>10}", end="")
        print()
        for r in all_results:
            lat = r.get("prefill_latency", {})
            if not lat:
                continue
            print(f"{r['strategy']:<14}", end="")
            for sl in seq_lens:
                entry = lat.get(sl, lat.get(str(sl), {}))
                ms = entry["mean_ms"] if isinstance(entry, dict) else 0
                print(f"  {ms:>9.2f}ms", end="")
            print()

        print(f"\nDECODE LATENCY (prompt={args.decode_prompt_len}, gen={args.decode_gen_tokens})")
        print(f"{'Strategy':<14}  {'Total':>10}  {'Per-Token':>10}  {'Tok/s':>8}")
        for r in all_results:
            dec = r.get("decode_latency", {})
            if not dec:
                continue
            print(f"{r['strategy']:<14}  {dec['total_mean_ms']:>9.2f}ms  "
                  f"{dec['per_token_ms']:>9.2f}ms  {dec['tokens_per_sec']:>8.1f}")

    # Save
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save dimension data
    with open(out_dir / "dimensions.json", "w") as f:
        json.dump(dim_data, f, indent=2, default=str)

    print(f"\nResults saved to {out_dir}/")

    # Plots
    print("\nGenerating plots...")
    generate_plots(all_results, dim_data, out_dir)
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
