#!/bin/bash
# Submit all LLM benchmarks (baseline/palu × infer_sweep/ppl/lmeval)

set -e

cd "$(dirname "$0")/.."
mkdir -p slurm_logs

TS=$(date +%Y%m%d_%H%M%S)
RESULTS_ROOT="results"

declare -a jobs=(
  "baseline infer_sweep"
  "palu infer_sweep"
  "baseline ppl"
  "palu ppl"
  "baseline lmeval"
  "palu lmeval"
)

echo "Submitting jobs at $TS"
JOB_IDS=()

for entry in "${jobs[@]}"; do
  VARIANT=$(echo $entry | awk '{print $1}')
  SUITE=$(echo $entry | awk '{print $2}')
  RUN_ID="${TS}_${VARIANT}_${SUITE}"
  NAME="${SUITE}_${VARIANT}"
  echo "  -> $NAME (run_id=$RUN_ID)"
  JID=$(sbatch --job-name="$NAME" slurm/run_llm.sbatch \
    --variant "$VARIANT" \
    --suite "$SUITE" \
    --results_root "$RESULTS_ROOT" \
    --run_id "$RUN_ID" \
    2>&1 | grep -oP '\\d+')
  echo "     Job ID: $JID"
  JOB_IDS+=("$JID")
done

echo ""
echo "Submitted Job IDs:"
for j in "${JOB_IDS[@]}"; do
  echo "  $j"
done
echo ""
echo "Monitor with: squeue -u $USER"
