# Quick Start: Night Sweep Experiments

## Prerequisites

```bash
# Activate conda environment
conda activate gc

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

## Validate Setup

```bash
python scripts/validate_setup.py
```

## Run Experiments

### Option 1: Submit All 6 Experiments Tonight

```bash
bash slurm/launch_all.sh
```

This will:
- Submit all 6 experiments as independent Slurm jobs
- Generate unique run_ids with timestamp
- Print all job IDs for monitoring

### Option 2: Run Single Experiment Locally (for testing)

```bash
python -m scripts.run_experiment run \
    --spec experiments/night_sweep.yaml \
    --name S1_sdpa_dense_sweep \
    --results-root results
```

### Option 3: Submit Single Experiment via Slurm

```bash
sbatch slurm/run_experiment.sbatch --exp S1_sdpa_dense_sweep
```

## Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View output
tail -f slurm_logs/*.out

# View errors  
tail -f slurm_logs/*.err
```

## Check Results

Results will be in:
```
results/
├── S1/<run_id>/
├── S2/<run_id>/
├── G3/<run_id>/
├── G4/<run_id>/
├── P1/<run_id>/
└── HET1/<run_id>/
```

Each directory contains:
- `config.json` - Experiment configuration
- `env.json` - Environment metadata
- `raw.json` - Complete results
- `summary.json` - Summary statistics
- `plots/*.png` - Visualization plots

## Generate Plots (if not auto-generated)

```bash
python -m scripts.plot_night_sweep results/S1/<run_id>/
```

## Expected Runtime

- **S1**: ~15-20 minutes
- **S2**: ~5-10 minutes
- **G3**: ~10-15 minutes
- **G4**: ~20-30 minutes
- **P1**: ~5-10 minutes
- **HET1**: ~5-10 minutes

**Total**: ~60-90 minutes for all experiments

## Troubleshooting

### PyYAML not found
```bash
pip install pyyaml
```

### Experiment not found
- Check experiment name matches `experiments/night_sweep.yaml`
- Run: `python scripts/validate_setup.py`

### CUDA out of memory
- Reduce batch size or sequence length in spec
- Use smaller dimension ranges for testing
