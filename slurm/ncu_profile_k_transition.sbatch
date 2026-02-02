#!/bin/bash
#SBATCH --job-name=ncu_prof_k
#SBATCH --output=slurm_logs/ncu_prof_k_%j.out
#SBATCH --error=slurm_logs/ncu_prof_k_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --qos=spot

set -eo pipefail
mkdir -p slurm_logs

source ~/.bashrc
mamba activate gac
cd $SLURM_SUBMIT_DIR

METRICS="launch__grid_size,launch__block_size,launch__registers_per_thread,launch__shared_mem_per_block_driver"

# K sweep: aligned (mod8=0), even (mod2=0, mod8!=0), odd
for K in 64 65 66 72 73 74 80 96 104 105 106 112 113 114 120 127 128; do
    echo "========================================"
    echo "  M=2048  N=2048  K=$K"
    echo "========================================"
    ncu --launch-skip 3 --launch-count 1 \
        --metrics "$METRICS" \
        --kernel-name regex:'gemm|cutlass|xmma' \
        python scripts/ncu_profile_kernels.py --M 2048 --N 2048 --K $K 2>&1
    echo ""
done
