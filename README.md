# GPU Irregular Dimensions Benchmark Suite

A reproducible microbenchmark suite to study GPU performance sensitivity to irregular dimensions (e.g., `head_dim=107` vs aligned dims like 96/112/128) on NVIDIA A100 GPUs, focusing on GEMM and SDPA backend selection.

## Motivation

Modern GPU architectures (A100/H100) are highly sensitive to dimension alignment. Irregular dimensions cause:

- **16-30% bandwidth waste** from non-aligned memory access (32-byte L2 cache sectors)
- **Tensor Core inefficiency** requiring padding (~16% wasted compute)
- **FlashAttention fallback** to Math path (O(N²) complexity, 3-10x slower)
- **Hardware acceleration failures** (TMA/WGMMA disabled on H100)

This suite validates these effects through systematic dimension sweeps. See `G-Compress_DeepResearch.md` for detailed analysis.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Local Run (Smoke Test)

**Option 1: Use smoke test script** (recommended):
```bash
bash scripts/smoke_test.sh
```

**Option 2: Manual command**:
```bash
# Quick test with minimal dimensions
python -m scripts.run_benchmarks \
    --experiment gemm_projection \
    --gemm-m-values 1024 \
    --gemm-k-values 96 107 128 \
    --gemm-n-values 96 107 128 \
    --warmup 5 \
    --iterations 20 \
    --dtype float16
```

### Slurm Run (Full Suite)

1. **Edit Slurm script**: Open `slurm/run_bench.sbatch` and paste your Slurm template header (partition, account, etc.) at the marked location.

2. **Submit job**:
```bash
sbatch slurm/run_bench.sbatch
```

3. **Generate plots** (after completion):
```bash
python -m scripts.plot_results results/<experiment_name>/<timestamp>/
```

## Repository Structure

```
G-Compress/
├── README.md                    # This file
├── PLAN.md                      # Detailed experiment plan
├── requirements.txt             # Python dependencies
├── src/                         # Core benchmark code
│   ├── config.py               # Configuration system
│   ├── benchmark_gemm.py       # GEMM benchmarks
│   ├── benchmark_sdpa.py       # SDPA benchmarks
│   ├── measurement.py          # CUDA event timing
│   ├── environment.py          # Environment collection
│   └── utils.py                # Utilities
├── scripts/                     # CLI scripts
│   ├── run_benchmarks.py       # Main entrypoint
│   ├── collect_env.py          # Environment collection
│   └── plot_results.py         # Plotting script
├── slurm/                       # Slurm integration
│   └── run_bench.sbatch         # Slurm script template
└── results/                     # Generated results (created at runtime)
    └── <experiment_name>/
        └── <timestamp>/
            ├── config.json
            ├── env.json
            ├── gemm_results.json
            ├── sdpa_results.json
            └── plots/
```

## Night Sweep Experiments

The night sweep suite extends the initial benchmarks with 6 focused experiments to provide deeper insights into dimension sensitivity, backend selection, and optimization strategies.

### Quick Start: Night Sweep

**Local run (single experiment)**:
```bash
python -m scripts.run_experiment run \
    --spec experiments/night_sweep.yaml \
    --name S1_sdpa_dense_sweep \
    --results-root results
```

**Slurm run (single experiment)**:
```bash
sbatch slurm/run_experiment.sbatch --exp S1_sdpa_dense_sweep
```

**Slurm run (all 6 experiments)**:
```bash
bash slurm/launch_all.sh
```

### Night Sweep Experiments

1. **S1_sdpa_dense_sweep**: Dense head_dim sweep (64-160) to identify performance cliffs
   - Outputs: `results/S1/<run_id>/plots/s1_latency_vs_head_dim.png`

2. **S2_sdpa_backend_forced**: Test forced backend selection for irregular dimensions
   - Outputs: `results/S2/<run_id>/plots/s2_backend_latency.png`

3. **G3_gemm_k_dense**: Dense K dimension sweep for Tensor Core sensitivity
   - Outputs: `results/G3/<run_id>/plots/g3_latency_vs_K.png`, `g3_tflops_vs_K.png`

4. **G4_gemm_n_dense_projectionlike**: Dense N dimension sweep (projection operations)
   - Outputs: `results/G4/<run_id>/plots/g4_latency_vs_N.png`

5. **P1_padding_rescue**: Compare padding strategies (107 vs padded 112/128)
   - Outputs: `results/P1/<run_id>/plots/p1_padding_rescue.png`

6. **HET1_head_hetero_batching_penalty**: Measure heterogeneous batching penalty
   - Outputs: `results/HET1/<run_id>/plots/het1_hetero_batching.png`

Each experiment writes results to `results/<exp_group>/<run_id>/` with:
- `config.json`: Experiment configuration
- `env.json`: Environment metadata
- `raw.json`: Complete raw results
- `summary.json`: Summary statistics
- `plots/*.png`: Visualization plots

See `PLAN_NIGHT_SWEEP.md` for detailed experiment rationale.

## Experiments

### Experiment A: GEMM Projection-Like Shapes

Tests QKV projection pattern `(M, K) @ (K, N)`:
- **M** (batch × tokens): [1024, 4096, 16384]
- **K** (d_model): 4096 (fixed)
- **N** (head_dim-like): [96, 104, 107, 112, 120, 128]

**Run**:
```bash
python -m scripts.run_benchmarks --experiment gemm_projection
```

### Experiment B: GEMM Reduction Dimension

Tests K as reduction dimension (critical for Tensor Core):
- **M**: 4096 (fixed)
- **N**: 4096 (fixed)
- **K**: [96, 104, 107, 112, 120, 128]

**Run**:
```bash
python -m scripts.run_benchmarks --experiment gemm_reduction --gemm-reduction-k
```

### Experiment C: SDPA Backend Selection

Tests PyTorch `scaled_dot_product_attention`:
- **Batch**: [1, 4, 8]
- **Seq length**: [1024, 4096]
- **Heads**: 32 (fixed)
- **Head dim**: [96, 104, 107, 112, 120, 128]

**Run**:
```bash
python -m scripts.run_benchmarks --experiment sdpa_backend
```

### Full Suite

Run all experiments:
```bash
python -m scripts.run_benchmarks --experiment all
```

## CLI Options

### Main Options

- `--experiment`: `gemm_projection`, `gemm_reduction`, `sdpa_backend`, or `all`
- `--output-dir`: Output directory (default: `results/`)
- `--dtype`: Data types to test: `float16`, `bfloat16`, `float32` (can specify multiple)
- `--warmup`: Warmup iterations (default: 10)
- `--iterations`: Measurement iterations (default: 100)
- `--device`: CUDA device (default: `cuda:0`)
- `--seed`: Random seed (default: 42)

### GEMM-Specific Options

- `--gemm-m-values`: M values (default: `1024 4096 16384`)
- `--gemm-k-values`: K values (default: `96 104 107 112 120 128`)
- `--gemm-n-values`: N values (default: `96 104 107 112 120 128`)
- `--gemm-reduction-k`: Enable reduction dimension test

### SDPA-Specific Options

- `--sdpa-batch-sizes`: Batch sizes (default: `1 4 8`)
- `--sdpa-seq-lengths`: Sequence lengths (default: `1024 4096`)
- `--sdpa-n-heads`: Number of heads (default: 32)
- `--sdpa-head-dims`: Head dimensions (default: `96 104 107 112 120 128`)

## Results Interpretation

### JSON Structure

Results are saved as JSON with the following structure:

```json
{
  "config": { ... },
  "experiments": [
    {
      "experiment_type": "projection_qkv",
      "m": 4096,
      "k": 4096,
      "n": 107,
      "dtype": "float16",
      "shape": {"m": 4096, "n": 107, "k": 4096},
      "timing": {
        "mean": 0.123,
        "std": 0.005,
        "p50": 0.122,
        "p90": 0.130,
        "p99": 0.135,
        "times_ms": [...],
        "times_s": [...]
      },
      "performance": {
        "tflops_mean": 278.5,
        "tflops_stats": {...},
        "bandwidth_gbs_mean": 1234.5,
        "bandwidth_gbs_stats": {...}
      }
    }
  ]
}
```

### Key Metrics

- **Latency**: Mean latency in milliseconds (lower is better)
- **TFLOPs**: Achieved compute throughput (higher is better, A100 peak ~312 TFLOPS FP16)
- **Bandwidth**: Memory bandwidth utilization (GB/s)
- **Backend**: For SDPA, which backend was used (Flash/MemEfficient/Math)

### Expected Patterns

1. **Aligned dimensions** (96, 112, 128) should show highest performance
2. **Irregular dimension** (107) should show significant degradation:
   - Lower TFLOPs (padding overhead)
   - Higher latency (memory access inefficiency)
   - SDPA fallback to Math backend
3. **Performance should correlate with alignment**, not just dimension size

## Plotting

Generate plots from results:

```bash
python -m scripts.plot_results results/<experiment_name>/<timestamp>/
```

This creates:
- `gemm_latency_vs_dimension.png`: Latency vs dimension with error bars
- `gemm_tflops_vs_dimension.png`: Achieved TFLOPs vs dimension
- `sdpa_latency_vs_head_dim.png`: SDPA latency vs head_dim
- `sdpa_backend_vs_head_dim.png`: Backend selection vs head_dim

## Environment Collection

Collect environment metadata separately:

```bash
python -m scripts.collect_env --output env.json
```

This records:
- PyTorch and CUDA versions
- GPU information (name, compute capability)
- Backend settings (TF32, cuDNN)
- System information

## Reproducibility

The suite uses:
- Fixed random seed (42)
- Deterministic cuDNN settings
- Environment metadata capture
- JSON output for post-analysis

Results should be reproducible across runs with the same hardware and software versions.

## Troubleshooting

### CUDA Out of Memory

Reduce dimensions or batch sizes:
```bash
python -m scripts.run_benchmarks \
    --experiment gemm_projection \
    --gemm-m-values 1024 \
    --gemm-k-values 96 107 128 \
    --gemm-n-values 96 107 128
```

### BF16 Not Supported

Skip bfloat16 if your GPU doesn't support it:
```bash
python -m scripts.run_benchmarks --dtype float16
```

### Slurm Script Issues

Ensure your Slurm template header includes:
- `--gres=gpu:a100:1` (or appropriate GPU)
- `--partition` and `--account` as needed
- Environment activation (conda/mamba/venv)

## References

- `G-Compress_DeepResearch.md`: Comprehensive analysis of dimension effects
- `PLAN.md`: Detailed experiment plan and rationale

## License

This benchmark suite is provided as-is for research purposes.
