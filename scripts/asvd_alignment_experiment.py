#!/usr/bin/env python3
"""
ASVD + GAC Alignment Experiment

Uses ASVD's original activation-aware SVD with proper calibration,
comparing unaligned (rank_align=1) vs aligned (rank_align=8) performance.

Inlines necessary ASVD functions to avoid import conflicts.
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset


# ============================================================================
# Inlined ASVD functions (to avoid import conflicts)
# ============================================================================

class SVDLinear(nn.Module):
    """SVD-decomposed linear layer from ASVD."""
    def __init__(self, U, S, V, bias=None, sigma_fuse="UV") -> None:
        super().__init__()
        self.ALinear = nn.Linear(U.size(1), U.size(0), bias=bias is not None)
        if bias is not None:
            self.ALinear.bias.data = bias
        self.BLinear = nn.Linear(V.size(1), V.size(0), bias=False)
        self.truncation_rank = S.size(0)
        if sigma_fuse == "UV":
            self.ALinear.weight.data = U.mul(S.sqrt()).contiguous()
            self.BLinear.weight.data = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
        elif sigma_fuse == "U":
            self.ALinear.weight.data = U.mul(S).contiguous()
            self.BLinear.weight.data = V.t().contiguous()
        elif sigma_fuse == "V":
            self.ALinear.weight.data = U.contiguous()
            self.BLinear.weight.data = V.t().mul(S.view(-1, 1)).contiguous()

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        param_ratio: float,
        act_aware=False,
        alpha=1,
        sigma_fuse="UV",
        rank_align=1,
    ):
        n_params = linear.weight.numel()
        compressed_params = int(n_params * param_ratio)
        rank = compressed_params // (linear.in_features + linear.out_features)
        # rank align
        rank = int(np.ceil(rank / rank_align) * rank_align)

        w = linear.weight.data.float()
        if act_aware:
            scaling_diag_matrix = 1
            if hasattr(linear, "scaling_diag_matrix"):
                scaling_diag_matrix = linear.scaling_diag_matrix ** alpha
            scaling_diag_matrix = scaling_diag_matrix + 1e-6
            w = w * scaling_diag_matrix.view(1, -1)

        try:
            U, S, V = torch.svd_lowrank(w, q=rank)
        except:
            print(f"svd failed for {linear}")
            return linear

        if act_aware:
            V = V / scaling_diag_matrix.view(-1, 1)

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        # nan check
        if (S != S).any() or (U != U).any() or (V != V).any():
            print("nan in SVD")
            return linear

        new_linear = SVDLinear(U, S, V, bias, sigma_fuse)
        new_linear.to(linear.weight.dtype)
        return new_linear

    def forward(self, inp):
        y = self.BLinear(inp)
        y = self.ALinear(y)
        return y


def get_calib_data(dataset_name, tokenizer, model_id, n_samples, seed=42):
    """Get calibration data."""
    if dataset_name == "wikitext2":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(data["text"])
    elif dataset_name == "c4":
        data = load_dataset("allenai/c4", data_files="en/c4-train.00000-of-01024.json.gz", split="train")
        text = "\n\n".join(data["text"][:1000])
    else:
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(data["text"])

    enc = tokenizer(text, return_tensors="pt")
    seq_len = 2048
    dataset = []
    np.random.seed(seed)

    for _ in range(n_samples):
        i = np.random.randint(0, enc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = enc.input_ids[:, i:j]
        dataset.append({"input_ids": inp})

    return dataset


@torch.no_grad()
def calib_input_distribution(model, calib_loader, method="abs_mean", use_cache=True):
    """Calibrate activation distribution for act-aware SVD."""
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/', '_')}_calib_input_distribution_{method}.pt"

    if os.path.exists(cache_file) and use_cache:
        all_scaling_diag_matrix = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.scaling_diag_matrix = all_scaling_diag_matrix[name].to(module.weight.device)
        return

    model.eval()

    def hook(module, input, output):
        if "abs_mean" in method:
            abs_mean = input[0].abs().mean(dim=-2).detach().view(-1)
            module.scaling_diag_matrix += abs_mean
        elif "abs_max" in method:
            abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
            module.scaling_diag_matrix = torch.where(
                abs_max > module.scaling_diag_matrix,
                abs_max,
                module.scaling_diag_matrix,
            )

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.scaling_diag_matrix = 0
            module.register_forward_hook(hook)

    for batch in tqdm(calib_loader, desc="Calibrating activation"):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        model(**batch)

    all_scaling_diag_matrix = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            all_scaling_diag_matrix[name] = module.scaling_diag_matrix

    os.makedirs("cache", exist_ok=True)
    torch.save(all_scaling_diag_matrix, cache_file)


@torch.no_grad()
def calib_sensitivity_ppl(model, calib_loader, args, use_cache=True):
    """Compute PPL-based sensitivity for each layer (ASVD's recommended method)."""
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/', '_')}_sensitivity_ppl_{args.scaling_method}_{args.alpha}_{args.n_calib_samples}_{args.calib_dataset}.pt"

    if os.path.exists(cache_file) and use_cache:
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict

    model.eval()

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_dict = {}
    # Use ASVD's recommended ratios
    param_ratio_candidates = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)

    print(f"Computing PPL-based sensitivity for {len(linear_info)} layers...")
    print(f"This will take a while ({len(linear_info) * len(param_ratio_candidates)} evaluations)...")
    pbar = tqdm(total=len(linear_info) * len(param_ratio_candidates))

    for raw_linear, info in linear_info.items():
        sensitivity_dict[info["full_name"]] = {}

        for param_ratio in param_ratio_candidates:
            # Apply SVD compression to this layer only
            svd_linear = SVDLinear.from_linear(
                raw_linear,
                param_ratio=param_ratio,
                alpha=args.alpha,
                act_aware=True,
                rank_align=1,
            )
            setattr(info["father"], info["name"], svd_linear)

            # Measure PPL with this layer compressed
            ppl = evaluate_perplexity(model, input_ids, min(args.n_calib_samples, 8))
            sensitivity_dict[info["full_name"]][param_ratio] = ppl

            pbar.update(1)
            pbar.set_postfix({"layer": info["full_name"][-25:], "ratio": param_ratio, "ppl": f"{ppl:.1f}"})

        # Restore original layer
        setattr(info["father"], info["name"], raw_linear)

    pbar.close()
    os.makedirs("cache", exist_ok=True)
    torch.save(sensitivity_dict, cache_file)
    return sensitivity_dict


@torch.no_grad()
def calib_sensitivity_stable_rank(model, calib_loader, args, use_cache=True):
    """Compute stable rank sensitivity for each layer (fast but less accurate)."""
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/', '_')}_sensitivity_stable_rank_{args.scaling_method}_{args.alpha}_{args.n_calib_samples}_{args.calib_dataset}.pt"

    if os.path.exists(cache_file) and use_cache:
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict

    model.eval()

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_dict = {}
    param_ratio_candidates = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for raw_linear, info in tqdm(linear_info.items(), desc="Computing sensitivity"):
        sensitivity_dict[info["full_name"]] = {}

        w = raw_linear.weight
        w_fro = torch.norm(w, p="fro") ** 2
        _, singular_values, _ = torch.svd(w.float(), compute_uv=False)
        spectral_norm = torch.max(singular_values)
        w_spec = spectral_norm ** 2
        sr = (w_fro / w_spec) ** 0.5

        for param_ratio in param_ratio_candidates:
            sensitivity_dict[info["full_name"]][param_ratio] = -sr * param_ratio ** 0.1

    os.makedirs("cache", exist_ok=True)
    torch.save(sensitivity_dict, cache_file)
    return sensitivity_dict


@torch.no_grad()
def evaluate_perplexity(model, input_ids, n_samples=32):
    """Evaluate perplexity on calibration data."""
    model.eval()
    device = next(model.parameters()).device

    nlls = []
    for i in range(min(n_samples, input_ids.shape[0])):
        inp = input_ids[i:i+1].to(device)
        out = model(inp, labels=inp)
        nlls.append(out.loss.item())

    return np.exp(np.mean(nlls))


# ============================================================================
# Experiment functions
# ============================================================================

def measure_latency(model, tokenizer, seq_len=512, n_warmup=5, n_measure=20):
    """Measure prefill latency with CUDA events."""
    device = next(model.parameters()).device
    input_ids = torch.randint(1, 1000, (1, seq_len), device=device)

    for _ in range(n_warmup):
        with torch.no_grad():
            model(input_ids)

    torch.cuda.synchronize()

    latencies = []
    for _ in range(n_measure):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            model(input_ids)
        end.record()

        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))

    return {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
    }


def evaluate_lm_tasks(model, tokenizer, tasks=["piqa", "hellaswag"], limit=200):
    """Evaluate on lm-eval tasks."""
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM

        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            limit=limit,
            batch_size=1,
        )

        accuracy = {}
        for task in tasks:
            if task in results["results"]:
                acc_key = "acc,none" if "acc,none" in results["results"][task] else "acc"
                accuracy[task] = results["results"][task].get(acc_key, 0)
        return accuracy
    except Exception as e:
        print(f"lm-eval failed: {e}")
        return {}


def apply_asvd_compression(model, sensitivity, args, rank_align=1):
    """Apply ASVD compression with specified rank alignment."""
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    # Build sensitivity list
    sensitivity_list = []
    for layername, v in sensitivity.items():
        for param_ratio, ppl in v.items():
            if param_ratio >= 1:
                continue
            sensitivity_list.append((layername, param_ratio, ppl))
    sorted_sensitive_list = sorted(sensitivity_list, key=lambda x: -x[2])

    # Binary search for target param ratio
    target_ratio = args.param_ratio_target
    high = len(sorted_sensitive_list) - 1
    low = 0

    while low < high:
        mid = (low + high) // 2
        layers_min_ratio = {layername: 1 for layername in sensitivity.keys()}
        for layername, param_ratio, ppl in sorted_sensitive_list[mid:]:
            layers_min_ratio[layername] = min(layers_min_ratio[layername], param_ratio)

        tot_params = 0
        compress_params = 0
        for layername, param_ratio in layers_min_ratio.items():
            raw_linear = module_dict[layername]
            tot_params += raw_linear.weight.numel()
            compress_params += raw_linear.weight.numel() * param_ratio

        now_ratio = compress_params / tot_params
        if now_ratio > target_ratio:
            high = mid
        else:
            low = mid + 1

    # Apply compression with specified rank_align
    layers_min_ratio = {layername: 1 for layername in sensitivity.keys()}
    for layername, param_ratio, ppl in sorted_sensitive_list[mid:]:
        layers_min_ratio[layername] = min(layers_min_ratio[layername], param_ratio)

    n_aligned = 0
    n_total = 0
    unique_ranks = set()

    for layername, param_ratio in tqdm(layers_min_ratio.items(), desc=f"Applying ASVD (align={rank_align})"):
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]

        if param_ratio < 1:
            svd_linear = SVDLinear.from_linear(
                raw_linear,
                param_ratio=param_ratio,
                alpha=args.alpha,
                act_aware=args.act_aware,
                sigma_fuse=args.sigma_fuse,
                rank_align=rank_align,
            )
            setattr(info["father"], info["name"], svd_linear)

            if hasattr(svd_linear, 'truncation_rank'):
                rank = svd_linear.truncation_rank
                unique_ranks.add(rank)
                n_total += 1
                if rank % 8 == 0:
                    n_aligned += 1

            raw_linear.to("cpu")

    pct_aligned = 100.0 * n_aligned / max(n_total, 1)
    return model, pct_aligned, len(unique_ranks), unique_ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--param_ratio_target", type=float, default=0.85)
    parser.add_argument("--n_calib_samples", type=int, default=32)
    parser.add_argument("--calib_dataset", type=str, default="wikitext2")
    parser.add_argument("--scaling_method", type=str, default="abs_mean")
    parser.add_argument("--act_aware", action="store_true", default=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--sigma_fuse", type=str, default="UV")
    parser.add_argument("--sensitivity_metric", type=str, default="ppl", choices=["ppl", "stable_rank"],
                        help="Sensitivity metric: 'ppl' (accurate but slow) or 'stable_rank' (fast but less accurate)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/asvd_alignment")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--eval_limit", type=int, default=200)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output, exist_ok=True)
    os.makedirs("cache", exist_ok=True)

    print("=" * 70)
    print("ASVD + GAC Alignment Experiment (Original ASVD Code)")
    print(f"Model: {args.model_id}")
    print(f"Target param ratio: {args.param_ratio_target} (compress {100*(1-args.param_ratio_target):.0f}%)")
    print(f"Activation-aware: {args.act_aware}, alpha={args.alpha}")
    print(f"Sensitivity metric: {args.sensitivity_metric}")
    print("=" * 70)

    results = []

    # ========== Baseline ==========
    print("\n[Step 1] Evaluating baseline model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    calib_loader = get_calib_data(
        args.calib_dataset, tokenizer, args.model_id,
        args.n_calib_samples, seed=args.seed
    )
    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)

    baseline_ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    print("  Evaluating baseline accuracy...")
    baseline_acc = evaluate_lm_tasks(model, tokenizer, limit=args.eval_limit)
    for task, acc in baseline_acc.items():
        print(f"  {task}: {acc:.4f}")

    baseline_latency = measure_latency(model, tokenizer, args.seq_len)
    print(f"  Baseline latency: {baseline_latency['mean_ms']:.1f}ms")

    results.append({
        "strategy": "baseline",
        "ppl": float(baseline_ppl),
        "pct_aligned": 100.0,
        "accuracy": baseline_acc,
        "latency": baseline_latency,
    })

    del model
    torch.cuda.empty_cache()

    # ========== ASVD Unaligned (rank_align=1) ==========
    print("\n[Step 2] ASVD with rank_align=1 (unaligned)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    calib_loader = get_calib_data(
        args.calib_dataset, tokenizer, args.model_id,
        args.n_calib_samples, seed=args.seed
    )
    calib_input_distribution(model, calib_loader, args.scaling_method, use_cache=True)
    if args.sensitivity_metric == "ppl":
        sensitivity = calib_sensitivity_ppl(model, calib_loader, args, use_cache=True)
    else:
        sensitivity = calib_sensitivity_stable_rank(model, calib_loader, args, use_cache=True)

    model, pct_aligned, n_unique, unique_ranks = apply_asvd_compression(
        model, sensitivity, args, rank_align=1
    )
    print(f"  Aligned: {pct_aligned:.1f}%, Unique ranks: {n_unique}")
    print(f"  Sample ranks: {sorted(unique_ranks)[:10]}...")

    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
    print(f"  PPL: {ppl:.2f}")

    print("  Evaluating accuracy...")
    accuracy = evaluate_lm_tasks(model, tokenizer, limit=args.eval_limit)
    for task, acc in accuracy.items():
        print(f"  {task}: {acc:.4f}")

    latency = measure_latency(model, tokenizer, args.seq_len)
    print(f"  Latency: {latency['mean_ms']:.1f}ms")

    results.append({
        "strategy": "asvd_unaligned",
        "ppl": float(ppl),
        "pct_aligned": pct_aligned,
        "n_unique_ranks": n_unique,
        "accuracy": accuracy,
        "latency": latency,
    })

    del model
    torch.cuda.empty_cache()

    # ========== ASVD Aligned (rank_align=8) ==========
    print("\n[Step 3] ASVD with rank_align=8 (aligned)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    calib_loader = get_calib_data(
        args.calib_dataset, tokenizer, args.model_id,
        args.n_calib_samples, seed=args.seed
    )
    calib_input_distribution(model, calib_loader, args.scaling_method, use_cache=True)
    if args.sensitivity_metric == "ppl":
        sensitivity = calib_sensitivity_ppl(model, calib_loader, args, use_cache=True)
    else:
        sensitivity = calib_sensitivity_stable_rank(model, calib_loader, args, use_cache=True)

    model, pct_aligned, n_unique, unique_ranks = apply_asvd_compression(
        model, sensitivity, args, rank_align=8
    )
    print(f"  Aligned: {pct_aligned:.1f}%, Unique ranks: {n_unique}")
    print(f"  Ranks: {sorted(unique_ranks)}")

    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
    print(f"  PPL: {ppl:.2f}")

    print("  Evaluating accuracy...")
    accuracy = evaluate_lm_tasks(model, tokenizer, limit=args.eval_limit)
    for task, acc in accuracy.items():
        print(f"  {task}: {acc:.4f}")

    latency = measure_latency(model, tokenizer, args.seq_len)
    print(f"  Latency: {latency['mean_ms']:.1f}ms")

    results.append({
        "strategy": "asvd_aligned_8",
        "ppl": float(ppl),
        "pct_aligned": pct_aligned,
        "n_unique_ranks": n_unique,
        "accuracy": accuracy,
        "latency": latency,
    })

    # ========== Save Results ==========
    output_file = os.path.join(args.output, "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<20} {'PPL':>10} {'Aligned':>10} {'Latency':>12} {'piqa':>8} {'hellaswag':>10}")
    print("-" * 70)
    for r in results:
        ppl_val = r["ppl"]
        aligned = r["pct_aligned"]
        lat = r["latency"]["mean_ms"]
        piqa = r["accuracy"].get("piqa", 0)
        hella = r["accuracy"].get("hellaswag", 0)
        print(f"{r['strategy']:<20} {ppl_val:>10.2f} {aligned:>9.1f}% {lat:>10.1f}ms {piqa:>8.2%} {hella:>10.2%}")

    if len(results) >= 3:
        unaligned_lat = results[1]["latency"]["mean_ms"]
        aligned_lat = results[2]["latency"]["mean_ms"]
        speedup = unaligned_lat / aligned_lat
        print(f"\nAlignment speedup: {speedup:.2f}x ({unaligned_lat:.1f}ms → {aligned_lat:.1f}ms)")

    print(f"\nExperiment Complete: {datetime.now()}")


if __name__ == "__main__":
    main()
