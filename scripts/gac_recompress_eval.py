"""
GAC Recompression & Perplexity Evaluation.

Takes rank allocation configs (from gac_rank_allocation.py), re-compresses
Llama-3-8B using SVD decomposition with each config, and evaluates
perplexity on WikiText-2.

Uses a self-contained SVD decomposition (no PaLU dependency needed).

Usage:
    python scripts/gac_recompress_eval.py \
        --rank-dir results/gac_allocation/ \
        --output results/gac_eval/ \
        --strategies unaligned,round8,gac_dp \
        --device cuda:0

Requires: ~40GB GPU memory (Llama-3-8B base model + decomposition overhead)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn


# -----------------------------------------------------------------------
# Self-contained headwise low-rank module (no PaLU dependency)
# -----------------------------------------------------------------------
class HeadwiseLowRankModule(nn.Module):
    """Headwise low-rank linear layer via SVD decomposition."""

    def __init__(self, ranks, in_features, out_features, bias=False):
        super().__init__()
        self.ranks = ranks
        self.num_groups = len(ranks)
        self.in_features = in_features
        self.out_features = out_features
        self.group_dim = out_features // self.num_groups

        self.VT = nn.Linear(in_features, sum(ranks), bias=False)
        self.U = nn.ModuleList([
            nn.Linear(r, self.group_dim, bias=bias) for r in ranks
        ])

    def forward(self, hidden_states):
        latents = self.VT(hidden_states)
        outputs = []
        offset = 0
        for i in range(self.num_groups):
            outputs.append(self.U[i](latents[..., offset:offset + self.ranks[i]]))
            offset += self.ranks[i]
        return torch.cat(outputs, dim=-1)

    @staticmethod
    def from_linear(old_module: nn.Linear, ranks: list):
        """Create from existing Linear via per-group SVD decomposition."""
        has_bias = old_module.bias is not None
        new_module = HeadwiseLowRankModule(
            ranks, old_module.in_features, old_module.out_features, bias=has_bias
        )

        w = old_module.weight.data.reshape(len(ranks), -1, old_module.in_features)
        if has_bias:
            b = old_module.bias.data.reshape(len(ranks), -1)

        wl_list = []
        wr_list = []
        for i in range(len(ranks)):
            U, S, Vt = torch.linalg.svd(w[i].float(), full_matrices=False)
            U = U[:, :ranks[i]]
            S = S[:ranks[i]]
            Vt = Vt[:ranks[i], :]
            sqrtS = torch.sqrt(torch.diag(S))
            L = (U @ sqrtS).to(w.dtype)   # (group_dim, rank)
            R = (sqrtS @ Vt).to(w.dtype)   # (rank, in_features)
            wl_list.append(L)
            wr_list.append(R)

        # Load into U modules
        for i in range(len(ranks)):
            new_module.U[i].weight.data = wl_list[i].contiguous()
            if has_bias:
                new_module.U[i].bias.data = b[i]

        # Load into VT
        new_module.VT.weight.data = torch.cat(wr_list, dim=0).contiguous()

        return new_module


# -----------------------------------------------------------------------
# Evaluation utilities
# -----------------------------------------------------------------------
def compute_perplexity(model, tokenizer, device: str, block_size: int = 512) -> dict:
    """Compute perplexity on WikiText-2 validation set."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        text = "\n".join(ds["text"])
    except Exception as e:
        print(f"WARNING: Could not load wikitext-2 ({e}), using fallback")
        text = "Artificial intelligence is transforming systems. " * 1000

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    nlls = []
    total_tokens = 0
    model.eval()
    with torch.inference_mode():
        for i in range(0, input_ids.size(1), block_size):
            chunk = input_ids[:, i:i + block_size]
            if chunk.size(1) < 2:
                continue
            out = model(chunk, labels=chunk)
            nll = out.loss * chunk.size(1)
            nlls.append(nll)
            total_tokens += chunk.size(1)

    if not nlls:
        return {"ppl": None, "nll": None, "tokens": 0}

    nll_sum = torch.stack(nlls).sum()
    ppl = torch.exp(nll_sum / total_tokens).item()
    return {"ppl": ppl, "nll": nll_sum.item(), "tokens": total_tokens}


def load_rank_config(rank_file: str) -> dict:
    with open(rank_file) as f:
        return json.load(f)


def compress_and_eval(
    rank_config: dict,
    strategy_name: str,
    device: str,
    dtype: torch.dtype = torch.float16,
) -> dict:
    """Load base model, compress with given ranks, evaluate perplexity."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"Loading base model: {model_id}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto",
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Compress: replace k_proj and v_proj with HeadwiseLowRankModule
    print(f"  Compressing {len(rank_config)} projections via SVD...")
    t1 = time.time()

    # Build module lookup
    module_dict = {name: module for name, module in model.named_modules()}
    parent_map = {}
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            parent_map[full_name] = (module, child_name)

    n_compressed = 0
    for layer_name, ranks_list in rank_config.items():
        if layer_name not in module_dict:
            print(f"  WARNING: {layer_name} not found, skipping")
            continue

        raw_linear = module_dict[layer_name]
        if not isinstance(raw_linear, nn.Linear):
            print(f"  WARNING: {layer_name} is not nn.Linear, skipping")
            continue

        if layer_name not in parent_map:
            print(f"  WARNING: {layer_name} parent not found, skipping")
            continue

        parent, attr = parent_map[layer_name]
        low_rank = HeadwiseLowRankModule.from_linear(raw_linear, ranks_list)
        setattr(parent, attr, low_rank)
        n_compressed += 1

    print(f"  Compressed {n_compressed} projections in {time.time() - t1:.1f}s")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    lr_params = 0
    for name, module in model.named_modules():
        if isinstance(module, HeadwiseLowRankModule):
            lr_params += sum(p.numel() for p in module.parameters())

    print(f"  Total params: {total_params:,}")
    print(f"  Low-rank projection params: {lr_params:,}")

    # Evaluate perplexity
    print(f"  Computing perplexity on WikiText-2...")
    t2 = time.time()
    ppl_result = compute_perplexity(model, tokenizer, device)
    print(f"  Perplexity: {ppl_result['ppl']:.2f} ({ppl_result['tokens']} tokens)")
    print(f"  Eval done in {time.time() - t2:.1f}s")

    del model
    torch.cuda.empty_cache()

    return {
        "strategy": strategy_name,
        "perplexity": ppl_result["ppl"],
        "nll": ppl_result["nll"],
        "tokens": ppl_result["tokens"],
        "total_params": total_params,
        "lr_params": lr_params,
        "n_compressed": n_compressed,
    }


def eval_palu_checkpoint(device: str, dtype: torch.dtype = torch.float16) -> dict:
    """Evaluate existing PaLU checkpoint perplexity."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "palu"))
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from gcompress_bench.palu_loader import load_palu_model

    print(f"\n{'='*60}")
    print("Strategy: palu_actual (existing checkpoint)")
    t0 = time.time()
    model, tokenizer, palu_dir = load_palu_model(device=device, torch_dtype=dtype)
    print(f"  PaLU model loaded from {palu_dir} in {time.time() - t0:.1f}s")

    total_params = sum(p.numel() for p in model.parameters())

    print(f"  Computing perplexity on WikiText-2...")
    t2 = time.time()
    ppl_result = compute_perplexity(model, tokenizer, device)
    print(f"  Perplexity: {ppl_result['ppl']:.2f} ({ppl_result['tokens']} tokens)")
    print(f"  Eval done in {time.time() - t2:.1f}s")

    del model
    torch.cuda.empty_cache()

    return {
        "strategy": "palu_actual",
        "perplexity": ppl_result["ppl"],
        "nll": ppl_result["nll"],
        "tokens": ppl_result["tokens"],
        "total_params": total_params,
    }


def eval_baseline(device: str, dtype: torch.dtype = torch.float16) -> dict:
    """Evaluate baseline (uncompressed) model perplexity."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    print(f"\n{'='*60}")
    print("Strategy: baseline (uncompressed)")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto",
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    total_params = sum(p.numel() for p in model.parameters())

    print(f"  Computing perplexity on WikiText-2...")
    t2 = time.time()
    ppl_result = compute_perplexity(model, tokenizer, device)
    print(f"  Perplexity: {ppl_result['ppl']:.2f} ({ppl_result['tokens']} tokens)")
    print(f"  Eval done in {time.time() - t2:.1f}s")

    del model
    torch.cuda.empty_cache()

    return {
        "strategy": "baseline",
        "perplexity": ppl_result["ppl"],
        "nll": ppl_result["nll"],
        "tokens": ppl_result["tokens"],
        "total_params": total_params,
    }


def main():
    parser = argparse.ArgumentParser(description="GAC Recompression + Eval")
    parser.add_argument("--rank-dir", default="results/gac_allocation")
    parser.add_argument("--output", default="results/gac_eval")
    parser.add_argument("--strategies", default="unaligned,round8,gac_dp")
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--include-palu", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rank_dir = Path(args.rank_dir)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    strategy_names = [s.strip() for s in args.strategies.split(",")]

    all_results = []

    # Evaluate baseline if requested
    if args.include_baseline:
        result = eval_baseline(args.device, dtype)
        all_results.append(result)

    # Evaluate existing PaLU checkpoint if requested
    if args.include_palu:
        try:
            result = eval_palu_checkpoint(args.device, dtype)
            all_results.append(result)
        except Exception as e:
            print(f"WARNING: PaLU eval failed ({e}), skipping")

    # Evaluate each strategy
    for strategy in strategy_names:
        rank_file = rank_dir / f"ranks_{strategy}.json"
        if not rank_file.exists():
            print(f"\nWARNING: {rank_file} not found, skipping {strategy}")
            continue

        rank_config = load_rank_config(rank_file)
        result = compress_and_eval(rank_config, strategy, args.device, dtype)
        all_results.append(result)

    # Print summary
    print(f"\n\n{'='*70}")
    print("PERPLEXITY COMPARISON")
    print(f"{'='*70}")
    print(f"{'Strategy':<20} {'Perplexity':>12} {'Total Params':>15}")
    print("-" * 50)
    for r in all_results:
        ppl_str = f"{r['perplexity']:.2f}" if r.get('perplexity') else "N/A"
        print(f"{r['strategy']:<20} {ppl_str:>12} {r.get('total_params', 0):>15,}")

    # Save results
    results_file = out_dir / "perplexity_comparison.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
