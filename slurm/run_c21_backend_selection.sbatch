#!/bin/bash
#SBATCH --job-name=C21_backend
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%j_C21_backend_selection.out
#SBATCH --error=slurm_logs/%j_C21_backend_selection.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jihao.xin@kaust.edu.sa

# =============================================================================
# C2.1 Experiment: PyTorch Backend Selection Verification
# =============================================================================
# Purpose: Verify which SDPA backend is selected for different head_dims
# Focus:   PaLU compression typical dimensions (114-125) and 8-alignment boundary
# Expected: head_dim % 8 != 0 => cannot use FlashAttention
# =============================================================================

set -e
source ~/.bashrc
mamba activate gc
cd $SLURM_SUBMIT_DIR

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "C2.1: PyTorch Backend Selection Experiment"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Run the experiment
OUTPUT_DIR=$(python -m scripts.run_experiment run \
    --spec experiments/night_sweep.yaml \
    --name C21_backend_selection \
    --results-root results \
    --device cuda:0 2>&1 | tee /dev/stderr | tail -1)

echo ""
echo "Results directory: $OUTPUT_DIR"

# Generate summary analysis
if [[ -n "$OUTPUT_DIR" && -d "$OUTPUT_DIR" ]]; then
    echo ""
    echo "=========================================="
    echo "Backend Selection Summary"
    echo "=========================================="
    python -c "
import json
from pathlib import Path

results_dir = Path('$OUTPUT_DIR')
raw_path = results_dir / 'raw.json'

if raw_path.exists():
    with open(raw_path) as f:
        data = json.load(f)

    summary = data.get('backend_summary', {})

    # Group by head_dim
    by_dim = {}
    for k, v in summary.items():
        dim = v['head_dim']
        if dim not in by_dim:
            by_dim[dim] = v

    print(f'Head_dim | 8-aligned | Flash | MemEff | Detected Backend')
    print('-' * 60)
    for dim in sorted(by_dim.keys()):
        v = by_dim[dim]
        aligned = 'Yes' if v['is_8_aligned'] else 'No'
        flash = 'Yes' if v['flash_available'] else 'No'
        memeff = 'Yes' if v['mem_efficient_available'] else 'No'
        detected = v['detected_backend']
        print(f'{dim:8} | {aligned:9} | {flash:5} | {memeff:6} | {detected}')

    # Key findings
    print('')
    print('Key Findings:')
    misaligned_use_flash = [d for d, v in by_dim.items() if not v['is_8_aligned'] and v['flash_available']]
    aligned_no_flash = [d for d, v in by_dim.items() if v['is_8_aligned'] and not v['flash_available']]

    if not misaligned_use_flash:
        print('  - Confirmed: Non-8-aligned dims cannot use FlashAttention')
    else:
        print(f'  - UNEXPECTED: Non-8-aligned dims using Flash: {misaligned_use_flash}')

    if not aligned_no_flash:
        print('  - Confirmed: All 8-aligned dims can use FlashAttention')
    else:
        print(f'  - UNEXPECTED: 8-aligned dims without Flash: {aligned_no_flash}')
else:
    print('Results file not found')
"
fi

echo ""
echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="
