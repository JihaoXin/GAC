#!/bin/bash
#SBATCH --job-name=LLM
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jihao.xin@kaust.edu.sa
#SBATCH --qos=spot

set -e

source ~/.bashrc
conda activate gac
cd $SLURM_SUBMIT_DIR

export PYTHONPATH="$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR/third_party/palu:$PYTHONPATH"
export TRANSFORMERS_OFFLINE=1
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

VARIANT=""
SUITE=""
RESULTS_ROOT="results"
RUN_ID=""
TASKS="piqa,hellaswag"
LIMIT=200

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2;;
    --suite) SUITE="$2"; shift 2;;
    --results_root) RESULTS_ROOT="$2"; shift 2;;
    --run_id) RUN_ID="$2"; shift 2;;
    --tasks) TASKS="$2"; shift 2;;
    --limit) LIMIT="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$VARIANT" || -z "$SUITE" ]]; then
  echo "Usage: sbatch run_llm.sbatch --variant baseline|palu --suite infer_sweep|ppl|lmeval [--results_root DIR] [--run_id ID] [--tasks TASKS] [--limit N]"
  exit 1
fi

echo "Running suite=$SUITE variant=$VARIANT run_id=${RUN_ID:-auto}"

if [[ "$SUITE" == "infer_sweep" ]]; then
  python -m gcompress_bench.llm_run --variant "$VARIANT" --suite infer_sweep --out "$RESULTS_ROOT" --run-id "$RUN_ID"
elif [[ "$SUITE" == "ppl" ]]; then
  python -m gcompress_bench.llm_eval --variant "$VARIANT" --suite ppl --out "$RESULTS_ROOT" --run-id "$RUN_ID"
elif [[ "$SUITE" == "lmeval" ]]; then
  python -m gcompress_bench.llm_eval --variant "$VARIANT" --suite lmeval --tasks "$TASKS" --limit "$LIMIT" --out "$RESULTS_ROOT" --run-id "$RUN_ID"
else
  echo "Unknown suite $SUITE"; exit 1
fi
