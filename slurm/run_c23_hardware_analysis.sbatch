#!/bin/bash
#SBATCH --job-name=C23_hardware
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%j_C23_hardware_analysis.out
#SBATCH --error=slurm_logs/%j_C23_hardware_analysis.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jihao.xin@kaust.edu.sa
#SBATCH --qos=spot

# =============================================================================
# C2.3 Experiment: Hardware Layer Analysis
# =============================================================================
# Purpose: Analyze hardware-level factors affecting dimension alignment performance
# Hypotheses:
#   H1: Tensor Core requires K % 16 == 0 for optimal FP16 performance
#   H2: L2 cache 32-byte sector causes over-fetch for non-aligned strides
#   H3: Non-aligned dims cause bandwidth efficiency loss
#   H4: Vectorized loads (float4) require head_dim % 8 == 0
# =============================================================================

set -e
source ~/.bashrc
mamba activate gac
cd $SLURM_SUBMIT_DIR

export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "C2.3: Hardware Layer Analysis Experiment"
echo "=========================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "CUDA version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
echo ""

# Create timestamp for results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/C23/${TIMESTAMP}_C23_hardware_layer"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run the hardware analysis
python scripts/run_c23_hardware_analysis.py \
    --output-dir "$OUTPUT_DIR" \
    --device cuda:0 \
    --seed 42

echo ""
echo "=========================================="
echo "Generating Summary Report"
echo "=========================================="

# Parse and display key findings
python -c "
import json
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
raw_path = output_dir / 'raw.json'

if raw_path.exists():
    with open(raw_path) as f:
        data = json.load(f)

    print('='*60)
    print('KEY FINDINGS')
    print('='*60)

    # H1 Findings
    h1 = data['analyses']['H1_tensor_core']
    print('\nH1: Tensor Core Utilization')
    print('-' * 40)
    s = h1['summary']
    print(f'  16-aligned K: {s[\"aligned_16_avg_tflops\"]:.1f} TFLOPS')
    print(f'  8-aligned K:  {s[\"aligned_8_only_avg_tflops\"]:.1f} TFLOPS')
    print(f'  Non-aligned:  {s[\"non_aligned_avg_tflops\"]:.1f} TFLOPS')
    if 'non_aligned_slowdown_vs_16' in s:
        print(f'  Slowdown: {(s[\"non_aligned_slowdown_vs_16\"]-1)*100:.1f}% (non-aligned vs 16-aligned)')

    # H2 Findings
    h2 = data['analyses']['H2_l2_cache']
    print('\nH2: L2 Cache Efficiency')
    print('-' * 40)
    s = h2['summary']
    print(f'  16-aligned bandwidth: {s[\"aligned_16_avg_bw_gbs\"]:.1f} GB/s')
    print(f'  Non-aligned bandwidth: {s[\"non_aligned_avg_bw_gbs\"]:.1f} GB/s')
    print(f'  Avg sector waste (non-aligned): {s[\"avg_sector_waste_non_aligned_pct\"]:.1f}%')

    # Conclusion
    print('\n' + '='*60)
    print('CONCLUSION')
    print('='*60)

    tc_slowdown = s.get('non_aligned_slowdown_vs_16', 1)
    if tc_slowdown > 1.1:
        print('H1 CONFIRMED: Tensor Core has significant slowdown for non-aligned K')
    elif tc_slowdown > 1.02:
        print('H1 PARTIAL: Tensor Core has small slowdown for non-aligned K')
    else:
        print('H1 NOT CONFIRMED: Tensor Core performance similar for all alignments')

    bw_ratio = s['aligned_16_avg_bw_gbs'] / max(s['non_aligned_avg_bw_gbs'], 0.001)
    if bw_ratio > 1.05:
        print(f'H2 CONFIRMED: L2 cache inefficiency causes {(bw_ratio-1)*100:.1f}% bandwidth loss')
    else:
        print('H2 NOT CONFIRMED: L2 cache efficiency similar for all alignments')

else:
    print('Results file not found')
"

echo ""
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo "Job completed at $(date)"
echo "=========================================="
