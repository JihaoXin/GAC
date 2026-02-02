#!/bin/bash
#SBATCH --job-name=palu_ppl
#SBATCH --output=slurm_logs/palu_ppl_%j.out
#SBATCH --error=slurm_logs/palu_ppl_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=01:00:00
#SBATCH --qos=spot

set -eo pipefail
mkdir -p slurm_logs

source ~/.bashrc
mamba activate gac
cd $SLURM_SUBMIT_DIR

echo "Installing fast_hadamard_transform..."
pip install fast_hadamard_transform 2>&1 || echo "Install failed, trying without"

echo "Evaluating PaLU checkpoint perplexity..."
python3 -c "
import sys, json, torch
sys.path.insert(0, 'third_party/palu')
sys.path.insert(0, 'src')

# Try to import PaLU; if hadamard fails, mock it
try:
    from gcompress_bench.palu_loader import load_palu_model
except ImportError:
    # Mock hadamard_utils
    import types
    mock_mod = types.ModuleType('fast_hadamard_transform')
    sys.modules['fast_hadamard_transform'] = mock_mod
    from gcompress_bench.palu_loader import load_palu_model

from datasets import load_dataset

print('Loading PaLU model...')
model, tokenizer, palu_dir = load_palu_model(device='cuda:0', torch_dtype=torch.float16)
total_params = sum(p.numel() for p in model.parameters())
print(f'PaLU model loaded from {palu_dir}')
print(f'Total params: {total_params:,}')

# Compute perplexity on WikiText-2
print('Computing perplexity on WikiText-2...')
ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
text = '\n'.join(ds['text'])
enc = tokenizer(text, return_tensors='pt')
input_ids = enc.input_ids.to('cuda:0')

nlls = []
total_tokens = 0
model.eval()
block_size = 512
with torch.inference_mode():
    for i in range(0, input_ids.size(1), block_size):
        chunk = input_ids[:, i:i + block_size]
        if chunk.size(1) < 2:
            continue
        out = model(chunk, labels=chunk)
        nll = out.loss * chunk.size(1)
        nlls.append(nll)
        total_tokens += chunk.size(1)

nll_sum = torch.stack(nlls).sum()
ppl = torch.exp(nll_sum / total_tokens).item()
print(f'Perplexity: {ppl:.4f} ({total_tokens} tokens)')
print(f'NLL sum: {nll_sum.item():.2f}')

# Save result
result = {
    'strategy': 'palu_actual',
    'perplexity': ppl,
    'nll': nll_sum.item(),
    'tokens': total_tokens,
    'total_params': total_params,
    'palu_dir': str(palu_dir),
}
import os
os.makedirs('results/gac_eval', exist_ok=True)
with open('results/gac_eval/palu_perplexity.json', 'w') as f:
    json.dump(result, f, indent=2)
print('Saved to results/gac_eval/palu_perplexity.json')
"
