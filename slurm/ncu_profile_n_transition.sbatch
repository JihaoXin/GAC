#!/bin/bash
#SBATCH --job-name=ncu_prof_n
#SBATCH --output=slurm_logs/ncu_prof_n_%j.out
#SBATCH --error=slurm_logs/ncu_prof_n_%j.err
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

# N sweep: aligned (mod8=0), even (mod2=0, mod8!=0), odd
for N in 1024 1025 1026 1032 1033 1034 1040 1280 1536 2048; do
    echo "========================================"
    echo "  M=2048  N=$N  K=128"
    echo "========================================"
    ncu --launch-skip 3 --launch-count 1 \
        --metrics "$METRICS" \
        --kernel-name regex:'gemm|cutlass|xmma' \
        python scripts/ncu_profile_kernels.py --M 2048 --N $N --K 128 2>&1
    echo ""
done
