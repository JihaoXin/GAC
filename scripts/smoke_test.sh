#!/bin/bash
# Smoke test script for quick validation
# This runs a minimal benchmark to verify the setup works

set -e

echo "=========================================="
echo "GPU Irregular Dimensions Benchmark Suite"
echo "Smoke Test"
echo "=========================================="
echo ""

# Check if CUDA is available
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'CUDA available: {torch.cuda.get_device_name(0)}')"

echo ""
echo "Running minimal GEMM benchmark..."
echo ""

# Run minimal benchmark
python3 -m scripts.run_benchmarks \
    --experiment gemm_projection \
    --gemm-m-values 1024 \
    --gemm-k-values 96 107 128 \
    --gemm-n-values 96 107 128 \
    --warmup 5 \
    --iterations 20 \
    --dtype float16 \
    --output-dir results/smoke_test

echo ""
echo "=========================================="
echo "✅ Smoke test completed!"
echo "Results saved to: results/smoke_test/"
echo "=========================================="
