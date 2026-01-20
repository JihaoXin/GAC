#!/bin/bash
# Quick smoke test for night sweep experiments
# Runs a tiny subset of S1 to verify the pipeline works

set -e

echo "=========================================="
echo "Night Sweep Smoke Test"
echo "=========================================="
echo ""

# Check if CUDA is available
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'CUDA available: {torch.cuda.get_device_name(0)}')"

echo ""
echo "Running minimal S1 test (head_dim 96, 107, 128 only)..."
echo ""

# Create a minimal test spec
cat > /tmp/test_spec.yaml << 'EOF'
experiments:
  S1_test:
    type: sdpa_dense
    dtype: fp16
    shapes:
      - {batch: 1, seq_len: 1024, n_heads: 32}
    head_dim_range: [96, 128]
    head_dim_step_1: 31  # Only test 96, 107, 128
    head_dim_step_2: 1
    backend: AUTO
    warmup: 5
    measure: 10
    trials: 1
EOF

# Run test
python3 -m scripts.run_experiment run \
    --spec /tmp/test_spec.yaml \
    --name S1_test \
    --results-root results/smoke_test \
    --warmup 5 \
    --measure 10

echo ""
echo "=========================================="
echo "✅ Smoke test completed!"
echo "Results: results/smoke_test/S1_test/"
echo "=========================================="

# Cleanup
rm /tmp/test_spec.yaml
