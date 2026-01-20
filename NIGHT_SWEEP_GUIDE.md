# Night Sweep Experiments - Quick Guide

## File Tree

```
G-Compress/
├── experiments/
│   └── night_sweep.yaml          # Experiment specifications
├── src/
│   ├── experiment_runner.py      # Experiment implementations
│   └── ...                       # Existing benchmark code
├── scripts/
│   ├── run_experiment.py         # CLI entrypoint for experiments
│   ├── plot_night_sweep.py       # Plotting for night sweep
│   └── smoke_test_night_sweep.sh # Quick smoke test
├── slurm/
│   ├── run_experiment.sbatch     # Slurm script for single experiment
│   └── launch_all.sh             # Launch all 6 experiments
└── results/                       # Results directory
    ├── S1/                       # S1 experiment results
    ├── S2/                       # S2 experiment results
    ├── G3/                       # G3 experiment results
    ├── G4/                       # G4 experiment results
    ├── P1/                       # P1 experiment results
    └── HET1/                     # HET1 experiment results
```

## Commands

### Local Run (Single Experiment)

```bash
# Run S1 experiment locally
python -m scripts.run_experiment run \
    --spec experiments/night_sweep.yaml \
    --name S1_sdpa_dense_sweep \
    --results-root results

# Run with custom run_id
python -m scripts.run_experiment run \
    --spec experiments/night_sweep.yaml \
    --name S2_sdpa_backend_forced \
    --results-root results \
    --run-id my_custom_run_id
```

### Slurm Run (Single Experiment)

```bash
# Submit single experiment
sbatch slurm/run_experiment.sbatch --exp S1_sdpa_dense_sweep

# With custom run_id
sbatch slurm/run_experiment.sbatch \
    --exp G3_gemm_k_dense \
    --results_root results \
    --run_id test_run_001
```

### Slurm Run (All 6 Experiments)

```bash
# Launch all experiments
bash slurm/launch_all.sh
```

This will:
- Submit all 6 experiments as independent jobs
- Generate unique run_ids with timestamp
- Print all job IDs for monitoring

### Generate Plots (After Experiment Completes)

```bash
# Plot results from a run directory
python -m scripts.plot_night_sweep results/S1/<run_id>/
```

### Smoke Test

```bash
# Quick test with minimal parameters
bash scripts/smoke_test_night_sweep.sh
```

## Experiment Details

### S1: SDPA Dense Sweep
- **Purpose**: Identify performance cliffs
- **Range**: head_dim 64-160 (step 1 for 64-128, step 2 for 128-160)
- **Shapes**: (B=4,S=2048,H=32), (B=1,S=4096,H=32)
- **Output**: `plots/s1_latency_vs_head_dim.png`

### S2: SDPA Backend Forced
- **Purpose**: Test forced backend selection
- **head_dims**: [96, 104, 107, 112, 120, 128]
- **backends**: [AUTO, FLASH, MEM_EFFICIENT, MATH]
- **Output**: `plots/s2_backend_latency.png`

### G3: GEMM K Dense
- **Purpose**: Tensor Core K dimension sensitivity
- **Range**: K 64-160 (same stepping as S1)
- **dtypes**: fp16, bf16
- **Output**: `plots/g3_latency_vs_K.png`, `g3_tflops_vs_K.png`

### G4: GEMM N Dense
- **Purpose**: Projection operation N dimension sensitivity
- **Range**: N 64-160
- **M values**: [1024, 4096, 16384]
- **Output**: `plots/g4_latency_vs_N.png`

### P1: Padding Rescue
- **Purpose**: Compare padding strategies
- **logical_dim**: 107
- **pad_options**: [107, 112, 128]
- **Operations**: SDPA + GEMM (reduction + projection)
- **Output**: `plots/p1_padding_rescue.png`

### HET1: Heterogeneous Batching
- **Purpose**: Measure mixed-dimension penalty
- **Patterns**: uniform, mild, medium, severe
- **Output**: `plots/het1_hetero_batching.png`

## Result Structure

Each experiment creates:
```
results/<exp_group>/<run_id>/
├── config.json      # Experiment configuration
├── env.json         # Environment metadata
├── raw.json         # Complete raw results
├── summary.json     # Summary statistics
└── plots/           # Visualization plots
    └── *.png
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View output
tail -f slurm_logs/<job_id>_<exp_name>.out

# View errors
tail -f slurm_logs/<job_id>_<exp_name>.err
```

## Expected Runtime

- **S1**: ~15-20 minutes (dense sweep, 2 shapes)
- **S2**: ~5-10 minutes (6 head_dims × 4 backends)
- **G3**: ~10-15 minutes (dense K sweep, 2 dtypes)
- **G4**: ~20-30 minutes (dense N sweep, 3 M values, 2 dtypes)
- **P1**: ~5-10 minutes (3 pad options, 2 operations)
- **HET1**: ~5-10 minutes (4 patterns)

**Total**: ~60-90 minutes for all 6 experiments

## Troubleshooting

### Experiment fails to start
- Check YAML syntax: `python -c "import yaml; yaml.safe_load(open('experiments/night_sweep.yaml'))"`
- Verify experiment name matches spec file

### CUDA out of memory
- Reduce batch size or sequence length in spec
- Use smaller dimension ranges for testing

### Backend not available
- Check PyTorch version (needs 2.0+)
- Some backends may not be available on all GPUs
- Errors are recorded in results, experiment continues

### Plots not generated
- Check that experiment completed successfully
- Verify `raw.json` exists in run directory
- Run plotting manually: `python -m scripts.plot_night_sweep <run_dir>`
