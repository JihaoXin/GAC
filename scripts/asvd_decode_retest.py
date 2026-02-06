#!/usr/bin/env python3
"""Re-test ASVD decode latency with more repeats for accuracy."""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add third_party to path (same as eval_all.py)
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "third_party" / "ASVD4LLM"))


def measure_decode_latency(model, tokenizer, prompt_len=128, gen_tokens=64,
                           n_warmup=3, n_measure=30):
    """Measure decode latency."""
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

    total = np.array(latencies)
    return {
        'total_mean_ms': float(np.mean(total)),
        'total_std_ms': float(np.std(total)),
        'per_token_ms': float(np.mean(total) / gen_tokens),
        'tokens_per_sec': float(gen_tokens / (np.mean(total) / 1000)),
        'count': n_measure,
    }


def load_asvd_model(model_id, rank_align):
    """Load ASVD model using same path as eval_all.py."""
    from datautils import get_calib_data
    from act_aware_utils import calib_input_distribution
    from sensitivity_simple import calib_sensitivity_ppl
    from binary_search_simple import binary_search_truncation_rank

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    # Calibrate
    calib_loader = get_calib_data("wikitext2", tokenizer, model_id, 32, seed=42)
    calib_input_distribution(model, calib_loader, "abs_mean", use_cache=True)

    # Load cached sensitivity
    sensitivity_file = SCRIPT_DIR / "results" / "unified_eval" / "sensitivity_ppl.json"
    if sensitivity_file.exists():
        print(f"Loading cached sensitivity from {sensitivity_file}")
        with open(sensitivity_file) as f:
            raw_sensitivity = json.load(f)
        sensitivity = {
            layer: {float(k): v for k, v in ratios.items()}
            for layer, ratios in raw_sensitivity.items()
        }
    else:
        print("Computing PPL sensitivity...")
        class Args:
            param_ratio_target = 0.85
            n_calib_samples = 32
            calib_dataset = "wikitext2"
            scaling_method = "abs_mean"
            alpha = 0.5
            compress_kv_cache = False
        sensitivity = calib_sensitivity_ppl(model, calib_loader, Args(), use_cache=True)

    # Compress
    class CompressionArgs:
        param_ratio_target = 0.85
        ppl_target = -1
        n_calib_samples = 32
        alpha = 0.5
        act_aware = True
        sigma_fuse = "UV"
        compress_kv_cache = False

    CompressionArgs.rank_align = rank_align
    binary_search_truncation_rank(model, sensitivity, calib_loader, CompressionArgs())

    return model, tokenizer


def main():
    model_id = "meta-llama/Meta-Llama-3-8B"

    print("=" * 60)
    print("ASVD Decode Latency Re-test (30 repeats)")
    print("=" * 60)

    results = {}

    # 1. Baseline
    print("\n[1/3] Baseline...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    results['baseline'] = measure_decode_latency(model, tokenizer)
    print(f"  Decode: {results['baseline']['per_token_ms']:.2f} ms/token")
    del model; torch.cuda.empty_cache()

    # 2. ASVD Unaligned
    print("\n[2/3] ASVD Unaligned (rank_align=1)...")
    model, tokenizer = load_asvd_model(model_id, rank_align=1)
    results['asvd_unaligned'] = measure_decode_latency(model, tokenizer)
    print(f"  Decode: {results['asvd_unaligned']['per_token_ms']:.2f} ms/token")
    del model; torch.cuda.empty_cache()

    # 3. ASVD Aligned
    print("\n[3/3] ASVD Aligned (rank_align=8)...")
    model, tokenizer = load_asvd_model(model_id, rank_align=8)
    results['asvd_aligned'] = measure_decode_latency(model, tokenizer)
    print(f"  Decode: {results['asvd_aligned']['per_token_ms']:.2f} ms/token")

    # Summary
    base = results['baseline']['per_token_ms']
    unalign = results['asvd_unaligned']['per_token_ms']
    align = results['asvd_aligned']['per_token_ms']

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Baseline:       {base:.2f} ms/token")
    print(f"ASVD Unaligned: {unalign:.2f} ms/token ({(unalign/base-1)*100:+.1f}%)")
    print(f"ASVD Aligned:   {align:.2f} ms/token ({(align/base-1)*100:+.1f}%)")
    print(f"GAC Speedup:    {(unalign/align-1)*100:+.1f}%")

    # Save
    Path('results/asvd_decode_retest').mkdir(parents=True, exist_ok=True)
    with open('results/asvd_decode_retest/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/asvd_decode_retest/results.json")


if __name__ == "__main__":
    main()
