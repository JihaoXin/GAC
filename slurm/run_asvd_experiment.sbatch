#!/bin/bash
#SBATCH --job-name=asvd_gac
#SBATCH --output=slurm_logs/asvd_gac_%j.out
#SBATCH --error=slurm_logs/asvd_gac_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=12:00:00
#SBATCH --qos=spot

set -eo pipefail
mkdir -p slurm_logs

source ~/.bashrc
mamba activate svdllm
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
cd $SLURM_SUBMIT_DIR

echo "============================================="
echo "ASVD + GAC Alignment Experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Python: $(python --version)"
echo "Transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "============================================="
echo ""

nvidia-smi
echo ""

# Run experiment: ASVD (per-layer sensitivity-based) + 3 strategies
# Model: Llama-3-8B
# Target: 85% params (15% compression) - within ASVD's recommended range
# Key: ASVD produces diverse per-layer ranks, giving GAC DP optimization space
python scripts/asvd_gac_experiment.py \
    --model_id meta-llama/Meta-Llama-3-8B \
    --param_ratio_target 0.85 \
    --output results/asvd_experiment_15pct \
    --device cuda \
    --n_calib_samples 32 \
    --calib_dataset wikitext2 \
    --scaling_method abs_mean \
    --sensitivity_metric ppl \
    --alpha 0.5 \
    --act_aware \
    --eval-accuracy \
    --accuracy-tasks piqa,hellaswag \
    --accuracy-limit 200 \
    --eval-latency \
    2>&1

echo ""
echo "============================================="
echo "Experiment Complete: $(date)"
echo "============================================="
