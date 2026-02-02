#!/bin/bash
#SBATCH --job-name=ncu_prof_sdpa
#SBATCH --output=slurm_logs/ncu_prof_sdpa_%j.out
#SBATCH --error=slurm_logs/ncu_prof_sdpa_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --qos=spot

set -eo pipefail
mkdir -p slurm_logs

source ~/.bashrc
mamba activate gac
cd $SLURM_SUBMIT_DIR

METRICS="launch__grid_size,launch__block_size,launch__registers_per_thread,launch__shared_mem_per_block_driver"

# SDPA head_dim sweep: 32-multiples, 8-multiples, and neighbors
# Goal: understand the staircase pattern within aligned (mod8=0) dimensions
# Specifically: why dips at d=96, 128, 160 (32-multiples)?
for D in 64 72 80 88 96 104 112 120 128 136 144 152 160; do
    echo "========================================"
    echo "  SDPA  batch=4  seq=2048  heads=32  head_dim=$D"
    echo "========================================"
    # Flash Attention kernels have various naming conventions
    ncu --launch-skip 3 --launch-count 1 \
        --metrics "$METRICS" \
        --kernel-name regex:'flash|fmha|attention|softmax' \
        python scripts/ncu_profile_sdpa.py --head-dim $D --batch 4 --seq-len 2048 --n-heads 32 2>&1
    echo ""
done

echo ""
echo "============================================="
echo "  Also test a few misaligned dims for comparison"
echo "============================================="

for D in 65 97 107 129; do
    echo "========================================"
    echo "  SDPA  batch=4  seq=2048  heads=32  head_dim=$D (misaligned)"
    echo "========================================"
    ncu --launch-skip 3 --launch-count 1 \
        --metrics "$METRICS" \
        python scripts/ncu_profile_sdpa.py --head-dim $D --batch 4 --seq-len 2048 --n-heads 32 2>&1
    echo ""
done
