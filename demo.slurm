#!/bin/bash
#SBATCH --job-name=LongBench
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%j_longbench.out
#SBATCH --error=slurm_logs/%j_longbench.err

# Examples:
#   sbatch slurm_longbench.sh
#   sbatch slurm_longbench.sh --model=llama --attention=rap --retain-ratios="0.9 0.8"
#   sbatch slurm_longbench.sh --model=mistral --attention=rap --retain-ratios="0.9 0.8" --lora

set -e  # Exit on error

# Activate mamba environment
source ~/.bashrc
mamba activate rap
cd ~/rap

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Defaults:
RETAIN_RATIOS="0.9 0.8 0.7 0.6 0.5"
ATTENTION="rap"  # rap | svd | palu
FAMILY="llama"  # llama | mistral
LORA=""
DATASETS="triviaqa,qasper,trec,samsum,lcc,repobench-p,qmsum,multi_news"
MAX_LENGTH="31500"
MODEL_PATH_LLAMAS="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_PATH_MISTRAL="mistralai/Mistral-7B-v0.3"

# Arg parsing: flags only (no positional args).
while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --) shift 1 ;;  # Ignore -- separator (used by sbatch to pass args)
    --attention=*) ATTENTION="${1#*=}"; shift 1 ;;
    --model=*) FAMILY="${1#*=}"; shift 1 ;;
    --family=*) FAMILY="${1#*=}"; shift 1 ;;
    --retain-ratios=*) RETAIN_RATIOS="${1#*=}"; shift 1 ;;
    --datasets=*) DATASETS="${1#*=}"; shift 1 ;;
    --max-length=*) MAX_LENGTH="${1#*=}"; shift 1 ;;
    --lora) LORA="--lora"; shift 1 ;;
    --attention|--retain-ratios|--model|--family|--datasets|--max-length)
      echo "Error: only --flag=value form is supported. Example: --model=llama --attention=rap --retain-ratios=\"0.9 0.8\" --lora"
      exit 2
      ;;
    *) echo "Unknown arg: ${1:-}"; exit 1 ;;
  esac
done

case "${FAMILY}" in
  llama) MODEL_PATH="${MODEL_PATH_LLAMAS}" ;;
  mistral) MODEL_PATH="${MODEL_PATH_MISTRAL}" ;;
  *)
    echo "Error: --family must be one of: llama, mistral"
    exit 1
    ;;
esac

echo "=========================================="
echo "LongBench Evaluation"
echo "=========================================="
echo "Model: ${FAMILY}"
echo "Model Path: ${MODEL_PATH}"
echo "Attention: ${ATTENTION}"
echo "Retain Ratios: ${RETAIN_RATIOS}"
echo "Datasets: ${DATASETS}"
echo "Max Length: ${MAX_LENGTH}"
if [[ -n "${LORA}" ]]; then
  echo "LoRA: enabled"
fi
echo "=========================================="
echo ""

for RR in $RETAIN_RATIOS; do
  echo "=========================================="
  echo "Evaluating ${ATTENTION} (retain ratio ${RR})"
  echo "=========================================="
  python3 -u test_longbench.py \
    --attention="${ATTENTION}" \
    --model="${FAMILY}" \
    --model-path="${MODEL_PATH}" \
    --retain-ratio="${RR}" \
    --datasets="${DATASETS}" \
    --max-length="${MAX_LENGTH}" \
    $LORA
  echo "✅ ${ATTENTION} (retain ratio ${RR}) completed"
  echo ""
done

echo "=========================================="
echo "✅ All LongBench evaluations completed!"
echo "=========================================="

