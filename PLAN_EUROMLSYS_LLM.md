# EuroMLSys LLM Benchmark Execution Plan

## Goals
- Benchmark Meta-Llama-3-8B-Instruct on a single A100 with two variants:
  - **baseline**: HF Transformers
  - **palu**: PaLU SVD compression (group size = 4 heads, ratio ≈ 0.7) using provided ratios
- Suites: `infer_sweep`, `ppl`, `lmeval`
- Slurm-based, reproducible, offline-friendly (HF cache only; no downloads during jobs)

## Deliverables
- CLIs:
  - `python -m gcompress_bench.llm_run --variant {baseline,palu} --suite infer_sweep --out <dir>`
  - `python -m gcompress_bench.llm_eval --variant {baseline,palu} --suite {ppl,lmeval} --tasks piqa,hellaswag --limit 200 --out <dir>`
- PaLU integration (gs=4, ratio=0.7) via provided ratio dir under `/home/xinj/rap/submodules/palu/…`
- Result structure: `results/<exp_group>/<run_id>/{config.json, env.json, raw.json, summary.json, run_summary.md, plots/*.png}`
- Slurm:
  - `slurm/run_llm.sbatch` (single job)
  - `slurm/launch_llm_night.sh` (6 jobs: baseline/palu × infer_sweep/ppl/lmeval)
- Plotting: `scripts/plot_llm_results.py` comparing baseline vs palu

## Suites
- **infer_sweep**
  - Prefill-only: B ∈ {1,4,8}, S ∈ {256,512,1024,2048,4096? (drop if OOM)}
  - Decode: ctx ∈ {512,1024,2048}, gen ∈ {64,128}, B ∈ {1,4}
  - Greedy decoding (temperature=0, do_sample=False)
- **ppl**
  - Preferred: WikiText-2 (if cached); fallback: `data/tiny_corpus.txt`
- **lmeval**
  - Tasks: piqa, hellaswag; limit=200, 0/1-shot
  - Offline fallback: mini JSONL tasks if harness/datasets unavailable

## Timing & Metrics
- torch.inference_mode()
- CUDA events; warmup ≥10 (prefill) / ≥5 (decode); measure ≥30; trials=3
- Stats: mean, std, p50, p90, p99
- Peak GPU memory: allocated + reserved
- Env logging: torch / cuda / driver / GPU model

## PaLU Integration
- Locate ratio dir under `/home/xinj/rap/submodules/palu/` (resolve best match)
- Approach: load baseline, apply PaLU low-rank/linear replacement in-memory (gs=4, ratio≈0.7)
- Verify: dummy forward + short generation
- Record: ratio path used, gs, target ratio, logical vs physical dims (if padding)

## Slurm
- `run_llm.sbatch`: args `--variant`, `--suite`, `--results_root`, `--run_id`; `conda activate gc`
- `launch_llm_night.sh`: submit 6 jobs; unique run_id (timestamp + exp); logs → `slurm_logs/<exp>_%j.(out|err)`

## File Layout
- `src/gcompress_bench/{llm_run.py,llm_eval.py,palu_loader.py,metrics.py}`
- `scripts/plot_llm_results.py`
- `experiments/llm_suites.yaml`
- `data/tiny_corpus.txt` (fallback)
- `results/<exp_group>/<run_id>/...`

## Risks & Mitigations
- Model/cache path: use HF cache; set `HF_HOME` if needed; fail fast with clear error
- PaLU path mismatch: list and pick best match; record chosen path
- OOM: drop S=4096 or reduce batch; log decision
- Offline data: fallback corpus and mini tasks
- lm-eval missing: add dependency if absent

## Runtime Estimate (per job, A100)
- infer_sweep: ~20–40 min each (baseline/palu)
- ppl: ~10–20 min each (fallback <5 min)
- lmeval: ~20–30 min each
- Total overnight budget: ~2–3 hours (plus Slurm queue)
