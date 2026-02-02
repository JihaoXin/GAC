# GAC Paper Plan (EuroMLSys Workshop, 6 pages)

## Title
"When Smaller Is Slower: Dimensional Collapse in Compressed LLMs"

## Core Message
维度对齐是压缩算法设计中被忽视的系统约束，考虑它能获得 free speedup。

---

## 1. Introduction & Motivation

**Story:**
- LLM 的维度经过高度优化设计，通常有很好的 GPU 亲和性（8/32/64/128 的倍数）
- Post-training 压缩技术普及（SVD, pruning, token eviction），旨在减少参数/内存
- 然而，很多压缩算法会改变模型维度，导致**维度变小但延迟变高**的 paradox
- 我们称之为 **Dimensional Collapse**

**三种压缩改变维度的方式：**
1. **Token Eviction**: KV cache 的 seq_len 改变 → GEMM 的 M 维度
2. **SVD**: W ≈ A@B，middle dimension 改变 → GEMM 的 K 维度
3. **Pruning**: 权重维度 D 改变 → GEMM 的 N 维度

**关键观察：**
- 这些压缩算法的 budget allocation（跨层分配保留维度）基于 importance/sensitivity
- 忽视了维度与延迟的 mismatch
- 压缩后的维度分布很随机，大多数不落在 8 和 32 的倍数上

**Reviewer 预防：**
- "为什么不直接 padding？" → (1) padding 增加内存和计算开销；(2) 事后补救不如压缩时就对齐；(3) 某些场景（token eviction seq_len）不容易 pad

---

## 2. Background

- Transformer 的主要算子：GEMM, GeMV, SDPA
- Post-training compression: SVD (PaLU), Pruning, Token Eviction
- Budget allocation 的四种策略（文中已有）
- GPU 计算层次：Grid → CTA → Warp → MMA → Fragment

---

## 3. Analysis: Sources of Performance Degradation (从上到下)

### 3.1 PyTorch Level: SDPA Backend Selection
- SDPA 有三个 backend：FlashAttention, Efficient Attention, Math fallback
- 当 head_dim 不满足条件（head_dim%8!=0）时会 fallback → **cliff 效应**（2-5× 慢）
- NCU 验证：misaligned dims (65,97,107,129) 完全不调用 flash_fwd_kernel，fallback 到 Math backend
- **Flash Attention 2 内部 template tier（NCU 验证）:**
  - FA2 选择最小的 template ≥ head_dim: {64, 96, 128, 160, 192, 224, 256}
  - 区间左开右闭：d=64 → t=64, d∈(64,96] → t=96, d∈(96,128] → t=128, ...
  - 每个 template 对应不同的 tile 配置 (Br×Bc):
    - d=64: template=64, Br×Bc=128×128（Bc 最大，最快）
    - d∈(64,96]: template=96, Br×Bc=128×64（Bc 减半，seq_len 方向迭代翻倍）
    - d∈(96,128]: template=128, Br×Bc=128×64
    - d∈(128,160]: template=160, Br×Bc=128×32（Bc 再减半）
    - d∈(160,256]: template=192/224/256, Br×Bc=128×32
  - **Bc 控制 KV-tile 宽度**: Bc 越小 → 沿 seq_len 迭代次数越多 → 越慢（主要 staircase 驱动）
  - **is_even_K 优化**: 当 head_dim 恰好等于 template (64,96,128,160) 时，设置 is_even_K=true
    → 跳过边界检查 → 解释了 32 倍数处的 dip（96, 128 恰好是 template 边界）
  - **Padding waste**: d=72 在 template=96 中浪费 25% 计算; d=104 在 template=128 中浪费 19%
- 数据: slurm_logs/ncu_prof_sdpa_18731.out

### 3.2 Library Level: cuBLAS Kernel Selection
- cuBLAS 根据维度对齐性选不同 kernel（ncu 验证）
- **三层 kernel tier:**
  - align8 (dim%8==0): cuBLAS-native sm80, mma.m16n8k16
  - align2 (dim%2==0): CUTLASS sm80 align2, mma.m16n8k16
  - align1 (dim 奇数): CUTLASS sm75 align1, mma.m16n8k8 (指令降级!)
- **cuBLAS heuristic kernel 切换**: 不同 M/N 区间选不同 CTA tile，可能次优
  - M 维度 NCU 验证：三次 kernel 切换（M=1088/1089, 1152/1153, 1728/1729）
  - 两个 kernel 家族：ampere_fp16 手写 SASS kernel vs sm80_xmma CUTLASS 代码生成 kernel
  - ampere_fp16_s16816gemm: 手写 SASS, tile 256×128, block=256, 218 regs（少数固定 tile）
  - sm80_xmma_gemm: XMMA 代码生成, tile 192×128, block=256, 204 regs（更多 tile 选择）
  - ampere_fp16_s1688gemm: 手写 SASS, tile 256×64, block=128, 234 regs
  - 两者使用相同 MMA 指令（m16n8k16），峰值吞吐相同，差异来自 tile-problem 匹配度
  - **关键发现**：M=1728→1729 处 heuristic 做出次优选择——切换到 tile 更大的 xmma kernel，
    导致 CTA=160（vs ampere 的 216），第二 wave SM 利用率仅 48%，latency 跳升 ~30%
  - 论点：cuBLAS heuristic 并非总是最优，压缩产生的非标准维度可能触发次优 kernel 选择
- N 和 K 维度最敏感（影响 leading dimension），M 维度不敏感（但受 heuristic 切换影响）
- 数据: results/alignment_sweep.csv (2116 行), ncu profiling logs

### 3.3 Hardware Level: Memory & Compute
- **Vectorized memory access**: 行宽决定 load 宽度 (128-bit/32-bit/16-bit)
- **CTA wave quantization**: total CTAs vs SM count → SM 利用率
- **MMA 指令降级**: sm80 m16n8k16 → sm75 m16n8k8，每条 MMA 计算量减半
- **Shared memory bank conflict**: 不对齐维度可能增加 conflict（待补充）

### Constraint Summary Table

| 层级 | 机制 | Constraint | 违反代价 |
|------|------|-----------|---------|
| PyTorch | SDPA backend selection | head_dim % 8 == 0 | Cliff: 2-5× 慢 (Math fallback) |
| FA2 | Template tile selection | head_dim ∈ {64,96,128,160} | Bc 减半 → staircase |
| FA2 | is_even_K 边界检查 | head_dim == template 边界 | 额外 bounds check |
| FA2 | Padding waste | head_dim 接近 template 上界 | 浪费 15-25% 计算 |
| cuBLAS | Kernel alignment tier | dim % 8 == 0 | ~25% 降级 |
| cuBLAS | Kernel selection heuristic | dim 落在 heuristic 甜区 | 40-60% 降级 |
| Hardware | Vectorized memory access | leading dim % 8 == 0 | Load 带宽降 4-8× |
| Hardware | CTA wave efficiency | total CTAs ≈ k × SM_count | SM 空闲浪费 |

---

## 4. GAC Framework

### Step 1: Computation Graph Analysis
- 分析 LLM 的主要算子（GEMM, GeMV, SDPA）
- 确定每个算子的哪些维度受压缩影响
- 映射：压缩参数 → 算子维度 → 性能约束

### Step 2: Empirical Profiling
- 对每种算子，在 LLM 典型维度范围做 profiling
- 建立 profiling table: (算子类型, 维度值) → 延迟
- 标记 cliff 点和 staircase 台阶
- 已有数据: alignment_sweep.csv

### Step 3: System-Constrained Budget Allocation
**原始问题（以 SVD 为例）:**
```
minimize  Σ_l  loss(W_l, rank_l)
s.t.      Σ_l  rank_l ≤ total_budget
```

**GAC 版本:**
```
minimize  Σ_l  sensitivity_l × |a_l - r_l|     (accuracy loss proxy)
s.t.      Σ_l  param(a_l) ≤ total_budget       (memory constraint)
          a_l ∈ A_l  ∀l                          (alignment + no cliff)
```

A_l = {d : d % 8 == 0, d 不在 cliff 上}（从 profiling table 得到）

**算法: Multi-choice Knapsack DP**
- 每层 L 个候选 aligned rank（r_l 附近的 8 倍数，排除 cliff）
- DP 复杂度 O(L × B × |A|)，L~32-80 层, |A|~3-5 候选，完全可解
- 比 round-to-nearest 好：全局平衡，sensitivity 高的层向上 round 保精度，sensitivity 低的层向下 round 省 budget

**候选集构造:**
```python
def get_candidates(r_original, dim_name, profile_table):
    candidates = []
    for r in range(r_original - 16, r_original + 17):
        if r <= 0 or r % 8 != 0:
            continue
        if is_cliff(r, dim_name, profile_table):
            continue
        candidates.append(r)
    return candidates
```

---

## 5. Experiments

### 5.1 Kernel-Level Demonstration
1. **Alignment Sensitivity Figure** (已有)
   - K/N/M sweep 图，展示三层 kernel 的锯齿/阶梯效应
   - 数据: results/alignment_sweep.csv
   - 图: Latex/figures/fig_alignment_sweep.pdf
   - **图中 kernel 简称（需写入正文/caption）:**
     - A = `ampere_fp16_s16816gemm` (hand-tuned SASS, 256×128 tile, block=256, 218 regs)
     - B = `sm80_xmma_gemm` (XMMA codegen, 192×128 tile, block=256, 204 regs)
     - C = `ampere_fp16_s1688gemm` (hand-tuned SASS, 256×64 tile, block=128, 234 regs)
   - 图中标注了 kernel 切换点（箭头 + 虚线），如 A→B, B→C, C→B
   - 正文需解释：三者都用 m16n8k16 MMA，性能差异来自 tile-problem 匹配和 wave efficiency

2. **Round-to-Aligned Speedup**
   - 取典型 PaLU rank（如 107, 214, 321）
   - 对比 as-is vs round-to-8 的 GEMM 延迟
   - 展示 "维度多 5% 但快 25%"
   - 可从现有 CSV 数据提取

3. **GAC Budget Allocation vs Baseline**
   - 在 Llama-3-8B 上跑 PaLU 原始 allocation
   - 统计每层 rank 分布（多少层落在非对齐维度）
   - 用 DP 重新分配，对比：
     - 每层 GEMM 延迟（从 profiling 表查）
     - 总推理延迟
     - Perplexity（需要跑一次 eval）

### 5.2 (Optional) End-to-End
- 如果时间允许：完整 LLM 推理对比
- 如果不允许：用 profiling 数据估算加速比

---

## 6. Discussion & Future Work
- 与 vLLM, TensorRT-LLM 的集成
- 扩展到其他硬件（H100, AMD MI300X）
- 扩展到其他压缩方法（quantization 的 group_size 对齐）
- 自动化 profiling + 压缩联合优化

---

## Existing Data & Experiments

### 已有数据
- `results/alignment_sweep.csv`: 2116 行, M/N/K sweep (M=1024-2048, N=1024-2048, K=64-128)
- `slurm_logs/ncu_profile_18723.out`: M 维度 ncu profiling (10 个点, 3 种 kernel)
- `slurm_logs/ncu_prof_n_18724.out`: N 维度 ncu profiling
- `slurm_logs/ncu_prof_k_18725.out`: K 维度 ncu profiling
- `slurm_logs/ncu_cutlass_18726.out`: CUTLASS align1/align2 kernel ncu profiling
- SDPA backend experiments (项目中已有)
- PaLU LLM experiments (项目中已有)

### 已有图
- `Latex/figures/fig_alignment_sweep.pdf`: 3-panel alignment sensitivity figure

### 关键发现总结
1. N 和 K 维度对 mod-8 对齐最敏感（影响 leading dimension → kernel 降级）
2. M 维度不受 mod-8 影响，但受 cuBLAS heuristic kernel 切换影响（阶梯效应）
3. 三层 kernel tier: align8 (cuBLAS-native sm80) > align2 (CUTLASS sm80) > align1 (CUTLASS sm75)
4. align1 不仅 load 慢，还退化到 sm75 的 m16n8k8 MMA 指令（双重打击）
5. cuBLAS heuristic 在 M=1728/1729 处做出次优 kernel 选择（wave efficiency 48% vs 100%，latency +30%）
6. ampere 手写 SASS kernel 和 sm80_xmma 代码生成 kernel 使用相同 MMA 指令，性能差异来自 tile-problem 匹配
6. 计时稳定可靠（±0.5% 波动，200 次平均）
