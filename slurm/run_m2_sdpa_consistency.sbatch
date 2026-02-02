#!/bin/bash
#SBATCH --job-name=GAC_M2_sdpa_consistency
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%j_M2_sdpa_consistency.out
#SBATCH --error=slurm_logs/%j_M2_sdpa_consistency.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jihao.xin@kaust.edu.sa
#SBATCH --qos=spot

# ==============================================================================
# M2: SDPA Benchmark Consistency Test
# ==============================================================================
# Purpose: Re-run SDPA benchmark to verify measurement consistency
# Issue: Table 3 vs Table 5 inconsistency - D=107 baseline latency differs by 6%
#        (2.192ms vs 2.064ms)
#
# Config:
#   - head_dims: 107, 112, 120, 128 (matching paper tables)
#   - iterations: 200 (consistent with paper claim of 200 measurements)
#   - warmup: 20 (extra warmup for stability)
#   - trials: Multiple runs captured in single measurement session
#   - batch_sizes: 1, 4, 8
#   - seq_lengths: 1024, 4096
#   - dtype: float16
#
# Expected: std/mean < 2% for CUDA event timing with 200 measurements
# ==============================================================================

set -e

# Activate environment
source ~/.bashrc
mamba activate gac
cd $SLURM_SUBMIT_DIR

# Set CUDA memory allocation settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print environment info
echo "========================================"
echo "M2: SDPA Benchmark Consistency Test"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "Working directory: $(pwd)"
echo "========================================"

# Check GPU
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Run SDPA benchmark with 200 iterations
# Note: --iterations maps to measurement_iterations in the code
OUTPUT_DIR="results/M2_benchmark_consistency"

python -m scripts.run_benchmarks \
    --experiment sdpa_backend \
    --output-dir "$OUTPUT_DIR" \
    --dtype float16 \
    --warmup 20 \
    --iterations 200 \
    --sdpa-head-dims 107 112 120 128 \
    --sdpa-batch-sizes 1 4 8 \
    --sdpa-seq-lengths 1024 4096 \
    --device cuda:0

echo ""
echo "========================================"
echo "Experiment completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"

# List output files
echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR"/sdpa_backend/

# Print summary statistics for D=107
echo ""
echo "========================================"
echo "Quick Summary for D=107 (key dimension):"
echo "========================================"
RESULT_FILE=$(ls -t "$OUTPUT_DIR"/sdpa_backend/*/sdpa_results.json | head -1)
if [ -f "$RESULT_FILE" ]; then
    python3 -c "
import json
with open('$RESULT_FILE') as f:
    data = json.load(f)

print('D=107 Results (batch=1, seq_len=4096):')
for exp in data['experiments']:
    if exp.get('head_dim') == 107 and exp.get('batch_size') == 1 and exp.get('seq_len') == 4096:
        timing = exp.get('timing', {})
        mean = timing.get('mean', 'N/A')
        std = timing.get('std', 'N/A')
        if isinstance(mean, (int, float)) and isinstance(std, (int, float)):
            cv = std / mean * 100 if mean > 0 else 0
            print(f'  Mean: {mean:.4f} ms')
            print(f'  Std:  {std:.4f} ms')
            print(f'  CV:   {cv:.2f}%')
        break
"
fi

echo ""
echo "Job completed successfully!"
