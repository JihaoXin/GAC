"""Simple script to evaluate LLM-Pruner with round_to=8."""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "third_party" / "LLM-Pruner"))


def measure_decode_latency(model, tokenizer, prompt_len=128, gen_tokens=64,
                           n_warmup=3, n_measure=10):
    device = next(model.parameters()).device
    input_ids = torch.randint(1, 1000, (1, prompt_len), device=device)

    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model.generate(
                input_ids, max_new_tokens=gen_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    torch.cuda.synchronize()

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


def main():
    model_id = "meta-llama/Meta-Llama-3-8B"
    output_dir = Path("results/unified_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    import LLMPruner.torch_pruning as tp
    from LLMPruner.pruner import hf_llama_pruner as llama_pruner
    from LLMPruner.datasets.example_samples import get_examples
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    print("=" * 60)
    print("Evaluating: LLM-Pruner pruned_r8 (round_to=8)")
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
        "round_to": 8,
    }

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
        "method": "llmpruner_pruned_r8",
        "round_to": 8,
        "accuracy": accuracy,
        "decode_latency": decode,
    }

    with open(output_dir / "llmpruner_pruned_r8_eval.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'llmpruner_pruned_r8_eval.json'}")


if __name__ == "__main__":
    main()
