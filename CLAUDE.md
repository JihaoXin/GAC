# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GAC (GPU-Aligned Compression)** is a research project studying dimensional collapse in compressed LLMs - the phenomenon where post-training compression produces irregular tensor dimensions that cause significant GPU performance degradation despite reducing FLOPs.

Paper title: "When Smaller Is Slower: Dimensional Collapse in Compressed LLMs"
Target venue: EuroMLSys (SIGPLAN format, 6 pages main content, references and appendix unlimited)

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run smoke test
bash scripts/smoke_test.sh

# Run GEMM benchmark
python -m scripts.run_benchmarks --experiment gemm_projection --dtype float16

# Run SDPA benchmark
python -m scripts.run_benchmarks --experiment sdpa_backend

# Run night sweep experiment (single)
python -m scripts.run_experiment run --spec experiments/night_sweep.yaml --name S1_sdpa_dense_sweep --results-root results

# Run LLM inference benchmark
python -m src.gcompress_bench.llm_run --variant baseline --suite infer_sweep --out results/llm
python -m src.gcompress_bench.llm_run --variant palu --suite infer_sweep --out results/llm

# Run LLM evaluation
python -m src.gcompress_bench.llm_eval --variant baseline --suite ppl --out results/llm
python -m src.gcompress_bench.llm_eval --variant palu --suite lmeval --tasks piqa,hellaswag --limit 200 --out results/llm

# Generate plots
python -m scripts.plot_results results/<experiment>/<timestamp>/
python -m scripts.plot_llm_results results/llm/

# Submit Slurm jobs
sbatch slurm/run_bench.sbatch
sbatch slurm/run_experiment.sbatch --exp S1_sdpa_dense_sweep
bash slurm/launch_all.sh           # all 6 night sweep experiments
bash slurm/launch_llm_night.sh     # all 6 LLM experiments (baseline/palu × 3 suites)
```

## Architecture

### Core Benchmarks (`src/`)
- `benchmark_gemm.py` / `benchmark_sdpa.py`: Low-level CUDA-timed microbenchmarks
- `measurement.py`: CUDA event timing utilities (warmup, measure, trials)
- `config.py`: Dataclass-based configuration (GEMMConfig, SDPAConfig)

### LLM Benchmarking (`src/gcompress_bench/`)
- `llm_run.py`: Prefill/decode latency benchmarks for Llama-3-8B
- `llm_eval.py`: Perplexity and lm-eval harness evaluation
- `palu_loader.py`: Loads PaLU-compressed models from `third_party/palu`
- `metrics.py`: Kernel measurement and stats utilities

### Third-Party (`third_party/`)
- `palu/`: PaLU SVD compression library (submodule)
- `RAP/`: RAP attention implementation (submodule)

### Experiment Specs (`experiments/`)
- `night_sweep.yaml`: Defines 6 experiment configurations (S1, S2, G3, G4, P1, HET1)

### Results Structure
```
results/<exp_group>/<run_id>/
├── config.json
├── env.json
├── raw.json
├── summary.json
└── plots/
```

## Key Concepts

- **Aligned dimensions**: 64, 96, 112, 128 (multiples of 8/16, optimal for Tensor Cores)
- **Irregular dimensions**: 107 (causes padding overhead, SDPA Math backend fallback)
- **PaLU**: Low-rank SVD compression with group_size=4, ratio≈0.7
- **Night sweep experiments**: S1/S2 (SDPA), G3/G4 (GEMM), P1 (padding), HET1 (heterogeneous heads)

## Automated Research System

本项目有全自动科研系统 (`auto_research/`)：

```bash
# 启动自动化系统（可连续运行数天）
python auto_research/orchestrator.py --max-days 3

# 后台运行
nohup python auto_research/orchestrator.py --max-days 3 > auto_research/logs/orchestrator.log 2>&1 &
```

关键文件：
- `auto_research/orchestrator.py` - 主流程
- `auto_research/memory.py` - 极简 Memory（分数追踪、停滞检测）
- `auto_research/agents/*.prompt` - 7 个 Agent prompts
- `auto_research/state/` - 状态文件
- `auto_research/logs/` - 运行日志

架构：Reviewer → Planner（自主决策）→ 执行 Actions → Validator

目标产出：
- `report.md` - 中文报告
- `Latex/main.tex` - EuroMLSys 论文 (SIGPLAN, 6页)

## Slurm Policy

**所有 GPU 任务必须通过 Slurm 运行**，不要在登录节点直接跑 GPU 代码。当需要创建或提交 Slurm 作业时，使用 `/slurm` skill。

- **默认机器**：`--constraint=gpu_a100`
- **大显存任务**（完整 LLM 实验）：`--constraint=gpu_a100_80gb`
- **快速验证**（< 3 分钟）：用 `srun --gres=gpu:a100:1 --constraint=gpu_a100 --qos=spot --pty bash`
- **其他任务**：创建 sbatch 脚本并提交

参考模板：`slurm/run_bench.sbatch`
日志目录：`slurm_logs/`

## LaTeX 编译规则

- **编译输出目录**：编译 LaTeX 时必须使用 `-output-directory=.out`，避免在源码目录生成 `.aux`、`.log` 等临时文件。例如：
  ```bash
  cd Latex && mkdir -p .out && pdflatex -output-directory=.out -interaction=nonstopmode main.tex
  ```
- **References 前必须加 `\clearpage`**：Conclusion 和 References 之间始终保留 `\clearpage`，不要删除
