"""
Unified evaluation script for ASVD and LLM-Pruner models.
Measures: Accuracy (PIQA, HellaSwag) and Decode Latency.

Usage:
    python scripts/unified_eval.py --method asvd --variant aligned
    python scripts/unified_eval.py --method llmpruner --variant pruned_r8
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

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

    # Create input prompt
    input_ids = torch.randint(1, 1000, (1, prompt_len), device=device)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                max_new_tokens=gen_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    torch.cuda.synchronize()

    # Measure
    latencies = []
    for _ in range(n_measure):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=gen_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    import numpy as np
    total_mean = np.mean(latencies)
    total_std = np.std(latencies)
    per_token = total_mean / gen_tokens
    tokens_per_sec = gen_tokens / (total_mean / 1000)

    return {
        "prompt_len": prompt_len,
        "gen_tokens": gen_tokens,
        "total_mean_ms": total_mean,
        "total_std_ms": total_std,
        "per_token_ms": per_token,
        "tokens_per_sec": tokens_per_sec,
        "count": n_measure,
    }


def evaluate_accuracy(model, tokenizer, tasks=["piqa", "hellaswag"], limit=200):
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
                # Try different accuracy key names
                task_results = results["results"][task]
                for key in ["acc,none", "acc", "accuracy"]:
                    if key in task_results:
                        accuracy[task] = task_results[key]
                        break
        return accuracy
    except Exception as e:
        print(f"lm-eval failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def load_asvd_model(model_id, variant, results_dir):
    """Load ASVD compressed model."""
    from datautils import get_calib_data
    from act_aware_utils import calib_input_distribution
    from sensitivity_simple import calib_sensitivity_ppl
    from binary_search_simple import binary_search_truncation_rank
    from modules.svd_linear import SVDLinear

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if variant == "baseline":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
        )
        return model, tokenizer

    # Load compressed model
    rank_align = 8 if variant == "aligned" else 1

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    # Load calibration data
    calib_loader = get_calib_data("wikitext2", tokenizer, model_id, 32, seed=42)

    # Calibrate activation distribution
    calib_input_distribution(model, calib_loader, "abs_mean", use_cache=True)

    # Load or compute sensitivity
    sensitivity_file = Path(results_dir) / "sensitivity_ppl.json"
    if sensitivity_file.exists():
        with open(sensitivity_file) as f:
            sensitivity = json.load(f)
    else:
        # This is slow, should be pre-computed
        print("Computing PPL sensitivity (this is slow)...")

        class Args:
            def __init__(self):
                self.param_ratio_target = 0.85
                self.n_calib_samples = 32
                self.calib_dataset = "wikitext2"

        sensitivity = calib_sensitivity_ppl(model, calib_loader, Args(), use_cache=True)

        with open(sensitivity_file, 'w') as f:
            json.dump(sensitivity, f, indent=2)

    # Apply compression
    binary_search_truncation_rank(
        model, sensitivity,
        target=-1,
        target_ratio=0.85,
        act_aware=True,
        alpha=0.5,
        rank_align=rank_align,
    )

    return model, tokenizer


def load_llmpruner_model(model_id, variant, results_dir):
    """Load LLM-Pruner compressed model."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if variant == "baseline":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
        )
        return model, tokenizer

    # Load pruned model checkpoint
    checkpoint_dir = Path(results_dir) / f"{variant}_checkpoint"

    if checkpoint_dir.exists():
        print(f"Loading checkpoint from {checkpoint_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
        )
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["asvd", "llmpruner"])
    parser.add_argument("--variant", type=str, required=True,
                        help="baseline, aligned, unaligned (ASVD) or baseline, pruned, pruned_r8 (LLM-Pruner)")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/unified_eval")
    parser.add_argument("--tasks", type=str, nargs="+", default=["piqa", "hellaswag"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--prompt_len", type=int, default=128)
    parser.add_argument("--gen_tokens", type=int, default=64)
    parser.add_argument("--skip_accuracy", action="store_true")
    parser.add_argument("--skip_decode", action="store_true")
    args = parser.parse_args()

    if args.results_dir is None:
        if args.method == "asvd":
            args.results_dir = "results/asvd_simple"
        else:
            args.results_dir = "results/llmpruner_llama3_v2"

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print(f"Unified Evaluation: {args.method.upper()} - {args.variant}")
    print(f"Model: {args.model_id}")
    print("=" * 60)

    # Load model
    print("\n[Step 1] Loading model...")
    if args.method == "asvd":
        model, tokenizer = load_asvd_model(args.model_id, args.variant, args.results_dir)
    else:
        model, tokenizer = load_llmpruner_model(args.model_id, args.variant, args.results_dir)

    results = {
        "method": args.method,
        "variant": args.variant,
        "model_id": args.model_id,
    }

    # Measure accuracy
    if not args.skip_accuracy:
        print(f"\n[Step 2] Evaluating accuracy on {args.tasks}...")
        accuracy = evaluate_accuracy(model, tokenizer, args.tasks, args.limit)
        results["accuracy"] = accuracy
        print(f"  Accuracy: {accuracy}")

    # Measure decode latency
    if not args.skip_decode:
        print(f"\n[Step 3] Measuring decode latency...")
        decode_latency = measure_decode_latency(
            model, tokenizer, args.prompt_len, args.gen_tokens
        )
        results["decode_latency"] = decode_latency
        print(f"  Decode: {decode_latency['per_token_ms']:.2f} ms/token, "
              f"{decode_latency['tokens_per_sec']:.1f} tok/s")

    # Save results
    output_file = Path(args.output) / f"{args.method}_{args.variant}_eval.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
