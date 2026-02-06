#!/bin/bash
#SBATCH --job-name=l2_verify
#SBATCH --output=slurm_logs/l2_verify_%j.out
#SBATCH --error=slurm_logs/l2_verify_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=spot

set -e

echo "================================================"
echo "L2 Cache Sector Alignment Verification"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Date: $(date)"
echo ""

cd /home/xinj/GAC

# Activate environment
source ~/.bashrc
conda activate gac

# Part 1: Run timing experiments
echo ""
echo "================================================"
echo "Part 1: Timing Experiments"
echo "================================================"

python scripts/l2_sector_verify.py --n-rows 1024 --output results/l2_sector_verify.json

# Part 2: NCU profiling for L2 metrics
echo ""
echo "================================================"
echo "Part 2: NCU L2 Cache Profiling"
echo "================================================"

mkdir -p results/ncu_profiles

# Define L2-related metrics
L2_METRICS="lts__t_sectors_srcunit_tex_op_read.sum"
L2_METRICS+=",lts__t_requests_srcunit_tex_op_read.sum"
L2_METRICS+=",lts__average_t_sectors_per_request_srcunit_tex_op_read.ratio"
L2_METRICS+=",lts__t_sector_hit_rate.pct"
L2_METRICS+=",dram__bytes_read.sum"
L2_METRICS+=",gpu__time_duration.sum"

echo ""
echo "Profiling K=4096 (aligned, K%16=0)..."
ncu --metrics $L2_METRICS --target-processes all \
    python scripts/ncu_l2_profile.py --K 4096 --ncu-mode 2>&1 | tee results/ncu_profiles/k4096.txt

echo ""
echo "Profiling K=4088 (K%8=0 but K%16=8)..."
ncu --metrics $L2_METRICS --target-processes all \
    python scripts/ncu_l2_profile.py --K 4088 --ncu-mode 2>&1 | tee results/ncu_profiles/k4088.txt

echo ""
echo "Profiling K=4095 (misaligned)..."
ncu --metrics $L2_METRICS --target-processes all \
    python scripts/ncu_l2_profile.py --K 4095 --ncu-mode 2>&1 | tee results/ncu_profiles/k4095.txt

echo ""
echo "Profiling K=4080 (aligned, K%16=0)..."
ncu --metrics $L2_METRICS --target-processes all \
    python scripts/ncu_l2_profile.py --K 4080 --ncu-mode 2>&1 | tee results/ncu_profiles/k4080.txt

# Part 3: Summary
echo ""
echo "================================================"
echo "Part 3: Summary"
echo "================================================"

echo ""
echo "Key L2 metrics comparison:"
echo ""
echo "K=4096 (aligned):"
grep -E "sectors_srcunit|sector_hit_rate|dram__bytes|time_duration" results/ncu_profiles/k4096.txt | head -10 || true

echo ""
echo "K=4088 (K%8=0 only):"
grep -E "sectors_srcunit|sector_hit_rate|dram__bytes|time_duration" results/ncu_profiles/k4088.txt | head -10 || true

echo ""
echo "K=4095 (misaligned):"
grep -E "sectors_srcunit|sector_hit_rate|dram__bytes|time_duration" results/ncu_profiles/k4095.txt | head -10 || true

echo ""
echo "================================================"
echo "Done!"
echo "================================================"
