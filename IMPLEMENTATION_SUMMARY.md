# Night Sweep Implementation Summary

## ✅ Implementation Complete

All 6 experiments (S1, S2, G3, G4, P1, HET1) have been implemented and are ready to run.

## File Tree

```
G-Compress/
├── experiments/
│   └── night_sweep.yaml              # Experiment specifications (YAML)
├── src/
│   ├── experiment_runner.py          # All 6 experiment implementations
│   ├── config.py                     # Existing config system
│   ├── measurement.py                # Enhanced with multi-trial support
│   └── utils.py                      # Enhanced dtype mapping (fp16/bf16)
├── scripts/
│   ├── run_experiment.py             # New CLI entrypoint
│   ├── plot_night_sweep.py           # Plotting for all 6 experiments
│   └── smoke_test_night_sweep.sh     # Quick smoke test
├── slurm/
│   ├── run_experiment.sbatch         # Slurm script for single experiment
│   └── launch_all.sh                 # Launch all 6 experiments
├── PLAN_NIGHT_SWEEP.md               # Experiment rationale
├── NIGHT_SWEEP_GUIDE.md              # Usage guide
└── requirements.txt                   # Updated with pyyaml
```

## Commands

### Local Run (Single Experiment)

```bash
python -m scripts.run_experiment run \
    --spec experiments/night_sweep.yaml \
    --name S1_sdpa_dense_sweep \
    --results-root results
```

### Slurm Run (Single Experiment)

```bash
sbatch slurm/run_experiment.sbatch --exp S1_sdpa_dense_sweep
```

### Slurm Run (All 6 Experiments Tonight)

```bash
bash slurm/launch_all.sh
```

### Smoke Test (Quick Validation)

```bash
bash scripts/smoke_test_night_sweep.sh
```

## Experiments Implemented

1. ✅ **S1_sdpa_dense_sweep**: Dense head_dim sweep (64-160)
2. ✅ **S2_sdpa_backend_forced**: Forced backend testing
3. ✅ **G3_gemm_k_dense**: GEMM K dimension sweep
4. ✅ **G4_gemm_n_dense_projectionlike**: GEMM N dimension sweep
5. ✅ **P1_padding_rescue**: Padding comparison
6. ✅ **HET1_head_hetero_batching_penalty**: Heterogeneous batching

## Key Features

- ✅ High-quality timing: CUDA events, warmup≥50, measure≥200, 3 trials
- ✅ Statistics: mean, std, p50, p90, p99
- ✅ Stable JSON schema: metadata, config, measurements, derived
- ✅ Automatic plotting: plots saved to results/<group>/<run_id>/plots/
- ✅ Slurm integration: ready to submit all jobs tonight
- ✅ Environment recording: config.json, env.json saved
- ✅ Error handling: continues on backend failures, records errors

## Result Structure

Each experiment writes to:
```
results/<exp_group>/<run_id>/
├── config.json      # Experiment configuration
├── env.json         # Environment metadata  
├── raw.json         # Complete raw results
├── summary.json     # Summary statistics
└── plots/           # Visualization plots
    └── *.png
```

## Next Steps

1. **Install dependencies** (if not already):
   ```bash
   pip install -r requirements.txt
   ```

2. **Run smoke test** to verify setup:
   ```bash
   bash scripts/smoke_test_night_sweep.sh
   ```

3. **Submit all jobs tonight**:
   ```bash
   bash slurm/launch_all.sh
   ```

4. **Monitor jobs**:
   ```bash
   squeue -u $USER
   tail -f slurm_logs/*.out
   ```

5. **Review results tomorrow**:
   - Check `results/S1/`, `results/S2/`, etc.
   - Review plots in each `plots/` directory
   - Analyze `raw.json` files for detailed data

## Notes

- All experiments use single GPU A100
- No large models - only GEMM and SDPA microbenchmarks
- Heterogeneous batching uses synthetic workload
- Padding rescue includes both SDPA and GEMM
- Backend forcing handles unavailable backends gracefully

## Expected Runtime

- Total: ~60-90 minutes for all 6 experiments
- Individual: 5-30 minutes per experiment (see NIGHT_SWEEP_GUIDE.md)

Ready to launch! 🚀
