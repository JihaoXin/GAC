#!/bin/bash
# Launch all night sweep experiments

set -e

# Create slurm_logs directory if missing
mkdir -p slurm_logs

# Experiments to run
EXPERIMENTS=(
    "S1_sdpa_dense_sweep"
    "S2_sdpa_backend_forced"
    "G3_gemm_k_dense"
    "G4_gemm_n_dense_projectionlike"
    "P1_padding_rescue"
    "HET1_head_hetero_batching_penalty"
)

RESULTS_ROOT="results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "Launching Night Sweep Experiments"
echo "Timestamp: $TIMESTAMP"
echo "=========================================="
echo ""

JOB_IDS=()

for EXP in "${EXPERIMENTS[@]}"; do
    RUN_ID="${TIMESTAMP}_${EXP}"
    
    echo "Submitting: $EXP (run_id: $RUN_ID)"
    
    JOB_ID=$(sbatch \
        --job-name="${EXP}" \
        --export=ALL,EXP_NAME="$EXP",RESULTS_ROOT="$RESULTS_ROOT",RUN_ID="$RUN_ID" \
        slurm/run_experiment.sbatch \
        --exp "$EXP" \
        --results_root "$RESULTS_ROOT" \
        --run_id "$RUN_ID" \
        2>&1 | grep -oP '\d+')
    
    if [[ -n "$JOB_ID" ]]; then
        JOB_IDS+=("$JOB_ID")
        echo "  → Job ID: $JOB_ID"
    else
        echo "  → Failed to submit"
    fi
    echo ""
done

echo "=========================================="
echo "✅ All jobs submitted!"
echo "=========================================="
echo ""
echo "Submitted Job IDs:"
for JID in "${JOB_IDS[@]}"; do
    echo "  - $JID"
done
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs in: slurm_logs/"
