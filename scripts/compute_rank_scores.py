"""
Compute 4 types of per-layer importance scores for rank allocation,
for all projection matrices (Q, K, V, O).

Methods:
  1. Fisher Information: E[grad^2] on projection weights
  2. Magnitude: RoPE pair energy (Q, K) / weight Frobenius norm (V, O)
  3. Activation: Input activation norm to projection layers
  4. Gradient: ||grad||_1 on projection weights

Usage:
  python scripts/compute_rank_scores.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --output results/rank_scores/llama3_8b.json \
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj"]


# ── Utilities (inlined from RAP/src/ops.py to avoid import issues) ────────

def load_model_and_tokenizer(model_path: str):
    print(f"[load] Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except (ValueError, Exception):
        # Fallback for Mistral-v0.3 etc.: try slow tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        except Exception:
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    return model, tokenizer


def get_rotary_emb(model, layer_id=0):
    """Get the rotary embedding from the model (works across transformers versions)."""
    attn = model.model.layers[layer_id].self_attn
    rotary = getattr(attn, "rotary_emb", None)
    if rotary is not None:
        return rotary
    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    if rotary is not None:
        return rotary
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    try:
        return LlamaRotaryEmbedding(config=attn.config)
    except TypeError:
        return LlamaRotaryEmbedding(head_dim)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def sample_batch(dataset, tokenizer, *, batch_size=8, seq_len=128, device="cuda"):
    texts = []
    while len(texts) < batch_size:
        t = dataset[np.random.randint(len(dataset))]["text"].strip()
        if t:
            texts.append(t)
    tok = tokenizer(texts, max_length=seq_len, truncation=True, padding="max_length", return_tensors="pt")
    return tok.input_ids.to(device)


@torch.no_grad()
def capture_layer_input(model, input_ids, layer_id, device):
    captured = {}

    class _Stop(Exception):
        pass

    def _hook(module, args):
        captured[0] = args[0].detach()
        raise _Stop()

    handle = model.model.layers[layer_id].register_forward_pre_hook(_hook)
    model.eval()
    try:
        model(input_ids.to(device))
    except _Stop:
        pass
    finally:
        handle.remove()
    return captured[0]


# ── Calibration data ──────────────────────────────────────────────────────

def get_calib_loader(tokenizer, nsamples: int = 32, seqlen: int = 1024, seed: int = 3):
    random.seed(seed)
    np.random.seed(seed)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")
    full_ids = enc.input_ids[0]
    samples = []
    for _ in range(nsamples):
        start = random.randint(0, len(full_ids) - seqlen - 1)
        ids = full_ids[start : start + seqlen].unsqueeze(0)
        samples.append({"input_ids": ids})
    return samples


# ── Score: Fisher Information ─────────────────────────────────────────────

def compute_fisher_scores(model, calib_loader, device) -> dict[str, list[float]]:
    """E[grad^2] on each projection's weights. All 4 projections scored in one backward pass."""
    num_layers = model.config.num_hidden_layers
    accum = {p: [0.0] * num_layers for p in PROJECTIONS}
    model.train()

    for batch in tqdm(calib_loader, desc="Fisher"):
        ids = batch["input_ids"][:, :-1].to(device)
        labels = batch["input_ids"][:, 1:].to(device)
        out = model(input_ids=ids, labels=labels)
        out[0].backward()
        for i in range(num_layers):
            attn = model.model.layers[i].self_attn
            for p in PROJECTIONS:
                g = getattr(attn, p).weight.grad
                if g is not None:
                    accum[p][i] += g.detach().float().pow(2).sum().item()
        model.zero_grad()

    n = len(calib_loader)
    model.eval()
    return {p: [(a / n) ** 0.5 for a in accum[p]] for p in PROJECTIONS}


# ── Score: Gradient L1 ────────────────────────────────────────────────────

def compute_gradient_scores(model, calib_loader, device) -> dict[str, list[float]]:
    """||grad||_1 on each projection's weights. All 4 projections scored in one backward pass."""
    num_layers = model.config.num_hidden_layers
    accum = {p: [0.0] * num_layers for p in PROJECTIONS}
    model.train()

    for batch in tqdm(calib_loader, desc="Gradient"):
        ids = batch["input_ids"][:, :-1].to(device)
        labels = batch["input_ids"][:, 1:].to(device)
        out = model(input_ids=ids, labels=labels)
        out[0].backward()
        for i in range(num_layers):
            attn = model.model.layers[i].self_attn
            for p in PROJECTIONS:
                g = getattr(attn, p).weight.grad
                if g is not None:
                    accum[p][i] += g.detach().float().abs().sum().item()
        model.zero_grad()

    n = len(calib_loader)
    model.eval()
    return {p: [a / n for a in accum[p]] for p in PROJECTIONS}


# ── Score: Activation Norm ────────────────────────────────────────────────

@torch.no_grad()
def compute_activation_scores(model, calib_loader, device) -> dict[str, list[float]]:
    """Input activation norm to each projection (pre-hook).
    Note: q/k/v share the same input (hidden_states), so their scores are identical.
    o_proj receives attention output, so its scores differ.
    """
    num_layers = model.config.num_hidden_layers
    accum = {p: [0.0] * num_layers for p in PROJECTIONS}
    count = {p: [0] * num_layers for p in PROJECTIONS}
    model.eval()

    hooks = []
    for i in range(num_layers):
        for p in PROJECTIONS:
            def hook_fn(module, args, layer_id=i, proj=p):
                hs = args[0].detach().float()
                accum[proj][layer_id] += hs.norm(dim=-1).mean().item()
                count[proj][layer_id] += 1
            h = getattr(model.model.layers[i].self_attn, p).register_forward_pre_hook(hook_fn)
            hooks.append(h)

    for batch in tqdm(calib_loader, desc="Activation"):
        model(input_ids=batch["input_ids"].to(device))

    for h in hooks:
        h.remove()

    return {p: [accum[p][i] / max(count[p][i], 1) for i in range(num_layers)] for p in PROJECTIONS}


# ── Score: Magnitude (RoPE energy for Q/K, weight norm for V/O) ──────────

@torch.no_grad()
def compute_magnitude_scores(model, tokenizer, device, batches=8, batch_size=4, seq_len=1024) -> dict[str, list[float]]:
    """RoPE pair energy for Q and K projections (computed together per layer).
    Weight Frobenius norm for V and O projections.
    """
    model.eval()
    model.config.use_cache = False

    train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    num_layers = model.config.num_hidden_layers
    qk_scores = {"q_proj": [], "k_proj": []}

    for layer_id in tqdm(range(num_layers), desc="Magnitude (Q,K RoPE)"):
        attn = model.model.layers[layer_id].self_attn
        head_dim = attn.head_dim
        rotary = get_rotary_emb(model, layer_id)

        # Per-projection accumulators
        proj_info = {}
        for pname in ["q_proj", "k_proj"]:
            proj = getattr(attn, pname)
            num_heads = proj.weight.shape[0] // head_dim
            proj_info[pname] = {
                "accum": torch.zeros(num_heads, head_dim // 2, device=device, dtype=torch.float32),
                "num_heads": num_heads,
            }
        total = 0

        for _ in range(batches):
            input_ids = sample_batch(train_dataset, tokenizer, batch_size=batch_size, seq_len=seq_len, device=device)
            hs = capture_layer_input(model, input_ids, layer_id, device)
            hs = model.model.layers[layer_id].input_layernorm(hs)
            B, T, _ = hs.shape

            # Get cos/sin from rotary embedding once (handle both Llama and Mistral APIs)
            num_kv = attn.k_proj.weight.shape[0] // head_dim
            dummy = torch.empty(B, num_kv, T, head_dim, device=device, dtype=hs.dtype)
            try:
                pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
                cos, sin = rotary(dummy, pos_ids)
            except (RuntimeError, TypeError):
                cos, sin = rotary(dummy, seq_len=T)

            while cos.dim() < 4:
                cos = cos.unsqueeze(0)
            while sin.dim() < 4:
                sin = sin.unsqueeze(0)
            # Ensure head dim is 1 so it broadcasts with both Q (32 heads) and K (8 heads)
            if cos.shape[1] > 1:
                cos = cos[:, :1, :, :]
                sin = sin[:, :1, :, :]

            # Compute RoPE energy for both Q and K using shared hidden states
            for pname in ["q_proj", "k_proj"]:
                proj = getattr(attn, pname)
                num_heads = proj_info[pname]["num_heads"]
                weight = proj.weight.to(device=device, dtype=hs.dtype)
                bias = proj.bias
                bias = bias.to(device=device, dtype=hs.dtype) if bias is not None else None
                out = F.linear(hs, weight, bias)
                out = out.view(B, T, num_heads, head_dim).transpose(1, 2)
                # out: [B, num_heads, T, head_dim]

                # Apply RoPE (cos/sin [1,1,T,D] broadcasts across head dimension)
                out_rope = (out * cos) + (rotate_half(out) * sin)

                out_fp32 = out_rope.to(torch.float32)
                first_half, second_half = torch.chunk(out_fp32, 2, dim=-1)
                energy = (first_half.pow(2) + second_half.pow(2)).mean(dim=(0, 2))
                proj_info[pname]["accum"] += energy

            total += 1

        for pname in ["q_proj", "k_proj"]:
            proj_info[pname]["accum"] /= max(total, 1)
            qk_scores[pname].append(proj_info[pname]["accum"].sum().item())

    # V and O: weight Frobenius norm per layer
    vo_scores = {"v_proj": [], "o_proj": []}
    print("  Computing weight Frobenius norms for V, O projections...")
    for layer_id in range(num_layers):
        attn = model.model.layers[layer_id].self_attn
        for pname in ["v_proj", "o_proj"]:
            w = getattr(attn, pname).weight.detach().float()
            vo_scores[pname].append(w.norm().item())

    return {**qk_scores, **vo_scores}


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--nsamples", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    device = torch.device(args.device)
    model, tokenizer = load_model_and_tokenizer(args.model)

    cfg = model.config
    model_short = args.model.split("/")[-1]
    num_layers = cfg.num_hidden_layers
    num_attention_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    print(f"Model: {model_short}, layers={num_layers}, heads={num_attention_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}")

    calib_loader = get_calib_loader(tokenizer, args.nsamples, args.seqlen, args.seed)

    print("\n=== Fisher Information (all projections) ===")
    fisher = compute_fisher_scores(model, calib_loader, device)

    print("\n=== Gradient L1 (all projections) ===")
    gradient = compute_gradient_scores(model, calib_loader, device)

    print("\n=== Activation Norm (all projections) ===")
    activation = compute_activation_scores(model, calib_loader, device)

    print("\n=== Magnitude (Q/K: RoPE energy, V/O: weight norm) ===")
    magnitude = compute_magnitude_scores(model, tokenizer, device, seq_len=args.seqlen)

    result = {
        "model": model_short,
        "model_path": args.model,
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_size": cfg.hidden_size,
        "calib": {
            "dataset": "wikitext-2-raw-v1",
            "nsamples": args.nsamples,
            "seqlen": args.seqlen,
            "seed": args.seed,
        },
        "scores": {
            "fisher": fisher,
            "magnitude": magnitude,
            "activation": activation,
            "gradient": gradient,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved scores to {args.output}")
    for method, proj_scores in result["scores"].items():
        print(f"\n  {method}:")
        for proj, vals in proj_scores.items():
            print(f"    {proj:8s}: min={min(vals):.4f}  max={max(vals):.4f}  ratio={max(vals)/max(min(vals),1e-12):.2f}")


if __name__ == "__main__":
    main()
