# Slurm Job Management

当用户需要运行 GPU 任务时，使用此指南创建和提交 Slurm 作业。

## 规则

1. **所有 GPU 任务必须通过 Slurm 运行**，不要直接在登录节点运行 GPU 代码

2. **机器选择**：
   - 默认使用 A100：`--constraint=gpu_a100`
   - 完整 LLM 实验（需要大显存）：`--constraint=gpu_a100_80gb`

3. **提交方式**：
   - 快速验证任务（< 3 分钟）：使用 `srun --gres=gpu:a100:1 --constraint=gpu_a100 --pty bash` 交互式运行
   - 其他所有任务：创建 sbatch 脚本并用 `sbatch` 提交

4. **参考模板**：`slurm/run_bench.sbatch`

## Sbatch 脚本模板

```bash
#!/bin/bash
#SBATCH --job-name=<任务名>
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu_a100        # 或 gpu_a100_80gb 用于大显存任务
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=<预估时间，如 02:00:00>
#SBATCH --output=slurm_logs/%j_<名称>.out
#SBATCH --error=slurm_logs/%j_<名称>.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jihao.xin@kaust.edu.sa

set -e

source ~/.bashrc
mamba activate gc
cd $SLURM_SUBMIT_DIR

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 实际运行的命令
<用户的命令>

echo "Job completed!"
```

## 工作流程

1. 用户描述要运行的任务
2. 判断任务类型：
   - 快速验证 → 建议用 `srun`
   - 长时间任务 → 创建 sbatch 脚本
3. 如果需要 sbatch：
   - 在 `slurm/` 目录下创建脚本
   - 确保 `slurm_logs/` 目录存在
   - 提交作业并告知用户 job ID
4. 提供查看日志的命令：`tail -f slurm_logs/<job_id>_*.out`

## 常用命令

```bash
# 查看队列
squeue -u $USER

# 取消作业
scancel <job_id>

# 查看作业详情
scontrol show job <job_id>

# 交互式 GPU shell（快速测试用）
srun --gres=gpu:a100:1 --constraint=gpu_a100 --time=00:10:00 --pty bash
```
