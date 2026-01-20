# Quick Start Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Verify Setup

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 3. Run Smoke Test

```bash
bash scripts/smoke_test.sh
```

Or manually:
```bash
python -m scripts.run_benchmarks \
    --experiment gemm_projection \
    --gemm-m-values 1024 \
    --gemm-k-values 96 107 128 \
    --gemm-n-values 96 107 128 \
    --warmup 5 \
    --iterations 20 \
    --dtype float16
```

## 4. Run Full Suite via Slurm

1. Edit `slurm/run_bench.sbatch` and add your Slurm header
2. Submit: `sbatch slurm/run_bench.sbatch`
3. Generate plots: `python -m scripts.plot_results results/<exp>/<timestamp>/`

## Key Files

- `scripts/run_benchmarks.py`: Main entrypoint
- `scripts/plot_results.py`: Generate plots from results
- `slurm/run_bench.sbatch`: Slurm script template
- `README.md`: Full documentation
- `PLAN.md`: Experiment plan and rationale
