#!/bin/bash
#SBATCH --job-name=dim_fix_test
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%j_dim_fix_test.out
#SBATCH --error=slurm_logs/%j_dim_fix_test.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jihao.xin@kaust.edu.sa
#SBATCH --qos=spot

set -e
source ~/.bashrc
mamba activate gac
cd $SLURM_SUBMIT_DIR

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "Testing analyze_palu_dimensions() fix"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Run the test script
python scripts/test_dimension_analysis_fix.py

echo ""
echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="
