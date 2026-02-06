"""
Unified evaluation script for baseline, ASVD, and LLM-Pruner.
Measures: Accuracy (PIQA, HellaSwag) and Decode Latency.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add third_party to path
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "third_party" / "ASVD4LLM"))
sys.path.insert(0, str(SCRIPT_DIR / "third_party" / "LLM-Pruner"))


def measure_decode_latency(model, tokenizer, prompt_len=128, gen_tokens=64,
                           n_warmup=3, n_measure=10):
    """Measure decode (autoregressive generation) latency."""
    device = next(model.parameters()).device
    input_ids = torch.randint(1, 1000, (1, prompt_len), device=device)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model.generate(
                input_ids, max_new_tokens=gen_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    torch.cuda.synchronize()

    # Measure
    latencies = []
    for _ in range(n_measure):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(
                input_ids, max_new_tokens=gen_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "prompt_len": prompt_len,
        "gen_tokens": gen_tokens,
        "total_mean_ms": float(np.mean(latencies)),
        "total_std_ms": float(np.std(latencies)),
        "per_token_ms": float(np.mean(latencies) / gen_tokens),
        "tokens_per_sec": float(gen_tokens / (np.mean(latencies) / 1000)),
    }


def evaluate_accuracy(model, tokenizer, tasks=["piqa", "hellaswag"], limit=200):
    """Evaluate accuracy using log-likelihood scoring (no lm_eval dependency)."""
    from datasets import load_dataset
    from tqdm import tqdm

    device = next(model.parameters()).device
    model.eval()
    accuracy = {}

    for task_name in tasks:
        correct = 0
        total = 0

        if task_name == "piqa":
            ds = load_dataset("piqa", split="validation", trust_remote_code=True)
            if limit:
                ds = ds.select(range(min(limit, len(ds))))
            for ex in tqdm(ds, desc=f"  {task_name}"):
                goal = ex["goal"]
                choices = [ex["sol1"], ex["sol2"]]
                label = ex["label"]
                scores = []
                for c in choices:
                    text = f"{goal} {c}"
                    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
                    with torch.no_grad():
                        out = model(ids)
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
            ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
            if limit:
                ds = ds.select(range(min(limit, len(ds))))
            for ex in tqdm(ds, desc=f"  {task_name}"):
                ctx = ex["ctx"]
                choices = ex["endings"]
                label = int(ex["label"])
                scores = []
                for c in choices:
                    text = f"{ctx} {c}"
                    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
                    with torch.no_grad():
                        out = model(ids)
                    logits = out.logits[0, :-1]
                    targets = ids[0, 1:]
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    score = log_probs.gather(1, targets.unsqueeze(1)).sum().item()
                    scores.append(score)
                pred = scores.index(max(scores))
                if pred == label:
                    correct += 1
                total += 1

        if total > 0:
            accuracy[task_name] = correct / total

    return accuracy


def evaluate_baseline(model_id, output_dir):
    """Evaluate baseline model."""
    print("\n" + "=" * 60)
    print("Evaluating: BASELINE")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    print("Measuring accuracy...")
    accuracy = evaluate_accuracy(model, tokenizer)
    print(f"  Accuracy: {accuracy}")

    print("Measuring decode latency...")
    decode = measure_decode_latency(model, tokenizer)
    print(f"  Decode: {decode['per_token_ms']:.2f} ms/token, {decode['tokens_per_sec']:.1f} tok/s")

    results = {"method": "baseline", "accuracy": accuracy, "decode_latency": decode}

    with open(output_dir / "baseline_eval.json", 'w') as f:
        json.dump(results, f, indent=2)

    del model
    torch.cuda.empty_cache()
    return results


def evaluate_asvd(model_id, rank_align, output_dir):
    """Evaluate ASVD compressed model."""
    variant = "aligned" if rank_align == 8 else "unaligned"
    print("\n" + "=" * 60)
    print(f"Evaluating: ASVD {variant} (rank_align={rank_align})")
    print("=" * 60)

    from datautils import get_calib_data
    from act_aware_utils import calib_input_distribution
    from sensitivity_simple import calib_sensitivity_ppl
    from binary_search_simple import binary_search_truncation_rank

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    # Calibrate and compress
    print("Calibrating activation distribution...")
    calib_loader = get_calib_data("wikitext2", tokenizer, model_id, 32, seed=42)
    calib_input_distribution(model, calib_loader, "abs_mean", use_cache=True)

    # Load cached sensitivity if available
    sensitivity_file = output_dir / "sensitivity_ppl.json"
    if sensitivity_file.exists():
        print(f"Loading cached sensitivity from {sensitivity_file}")
        with open(sensitivity_file) as f:
            raw_sensitivity = json.load(f)
        # Convert string keys to float (JSON stores dict keys as strings)
        sensitivity = {
            layer: {float(k): v for k, v in ratios.items()}
            for layer, ratios in raw_sensitivity.items()
        }
    else:
        print("Computing PPL sensitivity (this is slow)...")

        class Args:
            param_ratio_target = 0.85
            n_calib_samples = 32
            calib_dataset = "wikitext2"
            scaling_method = "abs_mean"
            alpha = 0.5
            compress_kv_cache = False

        sensitivity = calib_sensitivity_ppl(model, calib_loader, Args(), use_cache=True)
        with open(sensitivity_file, 'w') as f:
            json.dump(sensitivity, f, indent=2)

    print(f"Applying ASVD compression (rank_align={rank_align})...")

    class CompressionArgs:
        param_ratio_target = 0.85
        ppl_target = -1  # Use ratio-based, not PPL-based
        n_calib_samples = 32
        alpha = 0.5
        act_aware = True
        sigma_fuse = "UV"
        compress_kv_cache = False

    CompressionArgs.rank_align = rank_align
    binary_search_truncation_rank(model, sensitivity, calib_loader, CompressionArgs())

    print("Measuring accuracy...")
    accuracy = evaluate_accuracy(model, tokenizer)
    print(f"  Accuracy: {accuracy}")

    print("Measuring decode latency...")
    decode = measure_decode_latency(model, tokenizer)
    print(f"  Decode: {decode['per_token_ms']:.2f} ms/token, {decode['tokens_per_sec']:.1f} tok/s")

    results = {
        "method": f"asvd_{variant}",
        "rank_align": rank_align,
        "accuracy": accuracy,
        "decode_latency": decode,
    }

    with open(output_dir / f"asvd_{variant}_eval.json", 'w') as f:
        json.dump(results, f, indent=2)

    del model
    torch.cuda.empty_cache()
    return results


def evaluate_llmpruner(model_id, round_to, output_dir):
    """Evaluate LLM-Pruner compressed model."""
    import LLMPruner.torch_pruning as tp
    from LLMPruner.pruner import hf_llama_pruner as llama_pruner
    from LLMPruner.datasets.example_samples import get_examples
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    variant = "pruned_r8" if round_to == 8 else "pruned"
    print("\n" + "=" * 60)
    print(f"Evaluating: LLM-Pruner {variant} (round_to={round_to})")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    device = next(model.parameters()).device
    forward_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len=64).to(device)

    imp = llama_pruner.TaylorImportance(group_reduction='sum', taylor='param_first')

    kwargs = {
        "importance": imp,
        "global_pruning": True,
        "iterative_steps": 1,
        "ch_sparsity": 0.15,
        "ignored_layers": [],
        "channel_groups": {},
        "consecutive_groups": {
            layer.self_attn.k_proj: layer.self_attn.head_dim
            for layer in model.model.layers
        },
        "customized_pruners": {LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner},
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

    print("Computing Taylor importance...")
    example_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len=64).to(device)
    loss = model(example_prompts, labels=example_prompts).loss
    loss.backward()

    print("Pruning...")
    pruner.step()

    # Update attention heads
    for layer in model.model.layers:
        layer.self_attn.num_heads = layer.self_attn.q_proj.weight.shape[0] // layer.self_attn.head_dim
        layer.self_attn.num_key_value_heads = layer.self_attn.k_proj.weight.shape[0] // layer.self_attn.head_dim

    model.zero_grad()
    for p in model.parameters():
        p.requires_grad = False

    print("Measuring accuracy...")
    accuracy = evaluate_accuracy(model, tokenizer)
    print(f"  Accuracy: {accuracy}")

    print("Measuring decode latency...")
    decode = measure_decode_latency(model, tokenizer)
    print(f"  Decode: {decode['per_token_ms']:.2f} ms/token, {decode['tokens_per_sec']:.1f} tok/s")

    results = {
        "method": f"llmpruner_{variant}",
        "round_to": round_to,
        "accuracy": accuracy,
        "decode_latency": decode,
    }

    with open(output_dir / f"llmpruner_{variant}_eval.json", 'w') as f:
        json.dump(results, f, indent=2)

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--output", type=str, default="results/unified_eval")
    parser.add_argument("--eval", type=str, nargs="+",
                        default=["baseline", "asvd_unaligned", "asvd_aligned",
                                 "llmpruner_pruned", "llmpruner_pruned_r8"],
                        help="What to evaluate")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for eval_target in args.eval:
        if eval_target == "baseline":
            r = evaluate_baseline(args.model_id, output_dir)
        elif eval_target == "asvd_unaligned":
            r = evaluate_asvd(args.model_id, rank_align=1, output_dir=output_dir)
        elif eval_target == "asvd_aligned":
            r = evaluate_asvd(args.model_id, rank_align=8, output_dir=output_dir)
        elif eval_target == "llmpruner_pruned":
            r = evaluate_llmpruner(args.model_id, round_to=None, output_dir=output_dir)
        elif eval_target == "llmpruner_pruned_r8":
            r = evaluate_llmpruner(args.model_id, round_to=8, output_dir=output_dir)
        else:
            print(f"Unknown eval target: {eval_target}")
            continue
        all_results.append(r)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<25} {'PIQA':>8} {'Hella':>8} {'Decode':>12}")
    print("-" * 60)
    for r in all_results:
        piqa = r.get("accuracy", {}).get("piqa", 0)
        hella = r.get("accuracy", {}).get("hellaswag", 0)
        decode = r.get("decode_latency", {}).get("per_token_ms", 0)
        print(f"{r['method']:<25} {piqa:>8.3f} {hella:>8.3f} {decode:>10.2f} ms")

    # Save combined results
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
