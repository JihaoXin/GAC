"""
PaLU helper: locate compressed checkpoint directory and load model/tokenizer.
Assumes compressed HF-style folder exists under /home/xinj/rap/submodules/palu/.
"""
import glob
from pathlib import Path
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def find_palu_dir(base: str = "/home/xinj/rap/submodules/palu", pattern: str = "Meta-Llama-3-8B-Instruct_ratio-0.7_gs-4*") -> Path:
    candidates = sorted(glob.glob(str(Path(base) / pattern)))
    if not candidates:
        raise FileNotFoundError(f"No PaLU ratio directory matching {pattern} under {base}")
    return Path(candidates[0])


def load_palu_model(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
) -> Tuple:
    palu_dir = find_palu_dir()
    tokenizer = AutoTokenizer.from_pretrained(palu_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        palu_dir,
        torch_dtype=torch_dtype,
        device_map="auto" if device.startswith("cuda") else None,
    )
    return model, tokenizer, palu_dir
