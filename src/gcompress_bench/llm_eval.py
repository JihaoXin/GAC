"""
Evaluation: perplexity and lm-eval harness for baseline/palu variants.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from environment import collect_environment
from .metrics import compute_stats
from .palu_loader import load_palu_model


def load_model(variant: str, device: str, dtype_str: str = "float16"):
    torch_dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16
    if variant == "baseline":
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if device.startswith("cuda") else None,
        )
        palu_dir = None
    elif variant == "palu":
        model, tokenizer, palu_dir = load_palu_model(device=device, torch_dtype=torch_dtype)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer, palu_dir


def load_text_corpus() -> str:
    tiny_path = Path("data/tiny_corpus.txt")
    if tiny_path.exists():
        return tiny_path.read_text()
    tiny_path.parent.mkdir(parents=True, exist_ok=True)
    tiny_text = "Artificial intelligence is transforming systems. Performance depends on alignment."
    tiny_path.write_text(tiny_text)
    return tiny_text


def get_wikitext():
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        return "\n".join(ds["text"])
    except Exception:
        return load_text_corpus()


def compute_ppl(model, tokenizer, text: str, device: str, block_size: int = 512):
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    nlls = []
    for i in range(0, input_ids.size(1), block_size):
        chunk = input_ids[:, i : i + block_size]
        if chunk.size(1) < 2:
            continue
        with torch.inference_mode():
            out = model(chunk, labels=chunk)
            nll = out.loss * chunk.size(1)
            nlls.append(nll)
    if not nlls:
        return {"ppl": None, "nll": None, "tokens": 0}
    nll_sum = torch.stack(nlls).sum()
    tok_count = input_ids.size(1)
    ppl = torch.exp(nll_sum / tok_count).item()
    return {"ppl": ppl, "nll": nll_sum.item(), "tokens": tok_count}


def run_ppl(model, tokenizer, device):
    text = get_wikitext()
    return compute_ppl(model, tokenizer, text, device)


def run_lmeval(variant, model, tokenizer, tasks: str, limit: int, device: str, dtype_str: str):
    try:
        from lm_eval import evaluator
    except Exception as e:
        return {"error": f"lm-eval not available: {e}"}
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    model_args = f"pretrained={tokenizer.name_or_path},dtype={dtype_str}"
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=task_list,
        device=device,
        batch_size=1,
        limit=limit,
        no_cache=True,
    )
    # Extract accuracy if available
    scores = {}
    if "results" in results:
        for t, vals in results["results"].items():
            if "acc" in vals:
                scores[t] = vals["acc"]
            elif "acc,none" in vals:
                scores[t] = vals["acc,none"]
    return {"raw": results, "scores": scores}


def save_results(run_dir: Path, config: dict, raw: dict, summary: dict, run_summary: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    env = collect_environment()
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))
    (run_dir / "raw.json").write_text(json.dumps(raw, indent=2))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "run_summary.md").write_text(run_summary)


def main():
    parser = argparse.ArgumentParser(description="LLM eval (ppl / lm-eval)")
    parser.add_argument("--variant", choices=["baseline", "palu"], required=True)
    parser.add_argument("--suite", choices=["ppl", "lmeval"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--tasks", default="piqa,hellaswag")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.variant}_{args.suite}"
    run_dir = args.out / run_id

    model, tokenizer, palu_dir = load_model(args.variant, args.device, args.dtype)
    config = {
        "variant": args.variant,
        "suite": args.suite,
        "tasks": args.tasks,
        "limit": args.limit,
        "device": args.device,
        "dtype": args.dtype,
        "palu_dir": str(palu_dir) if palu_dir else None,
    }

    raw = {}
    summary = {}
    if args.suite == "ppl":
        res = run_ppl(model, tokenizer, args.device)
        raw["ppl"] = res
        summary["ppl"] = res["ppl"]
        summary["tokens"] = res["tokens"]
    else:
        res = run_lmeval(args.variant, model, tokenizer, args.tasks, args.limit, args.device, args.dtype)
        raw["lmeval"] = res
        summary["scores"] = res.get("scores", {})
        if "error" in res:
            summary["error"] = res["error"]

    run_summary = f"# Run summary\\n\\nVariant: {args.variant}\\nSuite: {args.suite}\\nRun ID: {run_id}\\n"
    save_results(run_dir, config, raw, summary, run_summary)
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
