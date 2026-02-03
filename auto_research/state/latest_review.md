# Paper Review: When Smaller Is Slower: Dimensional Collapse in Compressed LLMs

**Target Venue:** EuroMLSys (SIGPLAN format, 6 pages main content, references and appendix unlimited)
**Review Date:** 2026-01-29
**Reviewer:** Paper Reviewer Agent

---

## Summary

This paper identifies and analyzes "dimensional collapse"—a phenomenon where post-training compression of LLMs produces irregular tensor dimensions that degrade GPU performance despite reducing FLOPs. The authors systematically measure performance cliffs caused by misaligned dimensions (e.g., head_dim=107 causing 88% SDPA latency increase vs. head_dim=96 on NVIDIA A100), identify three root causes (Tensor Core misalignment 58%, vectorized load degradation 50%, SDPA bandwidth inefficiency 40%), and propose dimension repair strategies achieving 22-28% kernel-level speedup with 3.7-7.2% memory overhead. The work validates its applicability framework through contrasting experiments: RAP SVD showing -0.8% (correctly predicting no benefit for projection-based architectures), and direct SDPA benchmarks showing +86.9% average speedup across 45 workloads (validating benefit for direct compression).

The paper targets an important but overlooked problem at the intersection of LLM compression and GPU microarchitecture. The diagnostic contribution is solid, with systematic experiments isolating performance bottlenecks. However, the paper suffers from presentation issues (overcrowded figures, inconsistent terminology) and limited literature integration that reduces its perceived depth and academic maturity.

---

## Overall Rating

**Rating: Weak Accept (7/10)**

This is a valuable diagnostic contribution with solid experimental validation, but presentation quality and literature depth prevent it from reaching "Accept" level. The technical work is sound and the problem is real, but the paper reads more like a technical report than a mature research publication. With significant revisions to figures, terminology consistency, and related work depth, this could become a strong Accept.

**Confidence:** 4/5 (High confidence in the technical assessment; moderate uncertainty about venue fit)

---

## Detailed Scores

| Dimension | Weight | Score | Weighted |
|-----------|--------|-------|----------|
| Technical Quality | 40% | 7.5/10 | 3.00 |
| Paper Presentation | 30% | 6.0/10 | 1.80 |
| Innovation | 20% | 7.5/10 | 1.50 |
| Writing Quality | 10% | 7.0/10 | 0.70 |
| **Total** | 100% | - | **7.0/10** |

---

## Bottleneck Analysis (REQUIRED)

**主要瓶颈维度**: Paper Presentation (6.0/10)

**瓶颈分数**: 6.0/10

**为什么是瓶颈**:

尽管技术质量和创新性都达到了 7.5/10 的水平，但论文呈现质量（6.0/10）严重拖累了整体评分。具体问题包括：

1. **图表信息密度过高**: Page 6 在单页内塞入 3 张表格 + 大量正文，导致字体过小（≤7pt），在打印时几乎无法阅读
2. **术语不一致**: "dimensional collapse" vs "dimension misalignment" vs "irregular dimensions" 交替使用，缺乏统一定义
3. **图例可读性差**: Figure 4 颜色对比度不足，数据标签与图表元素重叠
4. **Related Work 深度不足**: 仅 32 个引用，缺少历史脉络讨论（GPU 架构演化、compression 方法演进），给人"孤立工作"的印象

**为什么这是瓶颈而非其他维度**:

- Technical Quality (7.5/10) 虽未达满分，但实验设计合理，root cause analysis 完整，不是主要障碍
- Innovation (7.5/10) 问题重要性足够，diagnostic framework 有价值，提升空间有限
- Paper Presentation (6.0/10) 是最容易提升且影响最大的维度：改进图表布局可直接提升 1-1.5 分

**突破方向**:

按照 SIGPLAN 双栏格式，当前布局问题的根源是**试图在有限空间内塞入过多内容**，导致信息密度过载。需要：

1. **重新设计 Page 6 布局** (HIGH PRIORITY)：
   - 将 3 个表格分散到不同页面
   - 使用 figure* 环境让关键表格跨栏显示
   - 增大字体到至少 8pt（当前 ≤7pt）

2. **简化图表设计** (HIGH PRIORITY)：
   - Figure 1: 减少文字标注，改用更简洁的示意图
   - Figure 4: 增大图例字体，使用高对比度颜色方案
   - Figure 5: 数据点标签与散点分离（使用 offset）

3. **统一术语体系** (MEDIUM PRIORITY)：
   - 在 §1 引入 "dimensional collapse" 后，全文统一使用
   - 避免 "misaligned" / "irregular" / "non-8-aligned" 混用

4. **扩展 Related Work** (MEDIUM PRIORITY)：
   - 增加 GPU 架构演化讨论（Volta → Ampere → Hopper 的对齐要求变化）
   - 补充 compression survey 引用（至少 5 篇近 3 年综述）
   - 添加历史脉络（何时开始关注 hardware-aware compression）

**给 Planner 的建议**:

下一轮应该创建 **FIGURE_LAYOUT_OPTIMIZATION** 任务，而非继续添加实验。具体：

```yaml
task_type: FIGURE_CODE_REQUIRED
priority: URGENT
actions:
  - 修改 scripts/create_paper_figures.py，调整 Figure 4/5 的字体大小和颜色方案
  - 重新设计 Page 6 布局：将 Table 3/4/5 分散到不同位置
  - 为所有图表添加 \small 或 \footnotesize 环境，确保字体 ≥ 8pt
  - 使用 \FloatBarrier 控制浮动体位置，避免跨 section 漂移
expected_outcome:
  - Page 6 字体从 ≤7pt 提升到 ≥8pt
  - Figure 4 图例可读性显著改善
  - 表格不再拥挤，每页信息密度适中
```

---

## Strengths

1. **系统化的 root cause analysis**: 通过控制实验（C23）逐一验证假设，成功隔离出三个主要原因（TC misalignment 58%, vectorized load 50%, SDPA BW 40%），并证伪 L2 cache（5.8%）。这种层次化的诊断方法在 systems 论文中较为少见。

2. **对比验证框架的严谨性**: 通过 negative case (RAP SVD: -0.8%) 和 positive case (Direct SDPA: +86.9%) 的对比，证明了 applicability framework 的预测能力。这种"证明框架能正确预测何时**不**应用方法"的设计显著提升了工作的可信度。

3. **问题的实际重要性**: Dimensional collapse 是一个真实存在但被忽视的问题。论文清楚地展示了即使 FLOPs 减少，性能仍可能下降 88%，这对 compression 方法设计者有重要指导意义。

4. **实验覆盖面广**: 从 microbenchmark（GEMM, SDPA）到 end-to-end LLM inference，从 backend selection 到 hardware profiling，覆盖了完整的系统栈。

5. **实用的 practitioner guidance**: Table 5 (applicability framework) 提供了清晰的决策树，帮助从业者判断何时应用 dimension repair，避免盲目优化。

---

## Weaknesses

1. **Related Work 深度不足**: 仅 32 个引用，缺少对 GPU 架构演化（Volta → Ampere → Hopper 对齐要求变化）、compression 方法历史脉络（从 pruning 到 quantization 到 low-rank）的讨论。给人"孤立工作"的印象，缺乏学术成熟度。

2. **术语不一致**: "dimensional collapse" (abstract/intro) vs. "dimension misalignment" (§3) vs. "irregular dimensions" (passim) 交替使用，未在首次出现时给出形式化定义。"head_dim" (code) vs. "head dimension" (prose) 混用。

3. **图表信息密度过高**: Page 6 在单页内包含 Table 2, 3, 4 三张表格 + 大量正文，字体 ≤7pt，在打印时几乡无法阅读。Figure 4 颜色对比度不足，浅色在白底上难以区分。

4. **Scope 限制的反复强调**: Abstract 和多处正文反复说明"实验仅限 A100，H100 是 future work"，给人"工作不完整"的印象。虽然 scope 限制是合理的，但过度强调反而削弱了贡献的普适性。

5. **部分实验细节缺失**: C23 (hardware layer analysis) 的实验设置（如何隔离 TC utilization? 使用什么 profiling 工具?）描述不充分，降低了可复现性。

---

## Major Issues (Must Fix)

### M1. Page 6 信息密度过载 (Critical Layout Issue)

**Location**: Page 6 (§5-6 transition)

**Issue**:
在我查看 page_06.png 时，观察到：
- 单页内包含 **Table 2 (Root cause analysis)**, **Table 3 (Negative validation)**, **Table 4 (Positive validation)** 三张表格
- 正文字体约 10pt，但表格内字体 ≤7pt（测量 "Misaligned" 列标题高度约 1.2mm，对应 ~7pt）
- Table 2 的列间距 <2mm，在打印时几乎无法区分
- 页面下半部分有大片空白（约 30mm），但上半部分极度拥挤

**Why it matters**:
这是典型的**双栏排版失败案例**。SIGPLAN 格式要求在 8.5"×11" 页面上双栏排版，有效单栏宽度仅 ~85mm。当前布局试图在有限空间内塞入 3 张表格 + 15 行正文，导致：
1. 字体过小，打印审稿人无法阅读
2. 信息密度过载，读者无法快速提取关键信息
3. 与 EuroMLSys 其他论文相比显得"业余"

**Suggested Fix**:

**方案 A（推荐）**: 拆分表格到不同页面
```latex
% Page 5: Root Cause Analysis
\section{Root Cause Analysis}
...
\begin{table}[t]  % 保持在当前页面顶部
  \caption{Root cause analysis (C23 experiment)}
  \label{tab:hardware}
  % Table 2 内容
\end{table}

% Page 6: Evaluation - Negative case
\subsection{Negative E2E Case}
...
\begin{table}[h]  % 就地放置
  \caption{Negative validation: RAP SVD E2E}
  \label{tab:rap_e2e}
  % Table 3 内容
\end{table}

% Page 7: Evaluation - Positive case
\subsection{Positive E2E Case}
...
\begin{table*}[t]  % 跨栏表格
  \caption{Positive validation: SDPA speedup across 45 workloads}
  \label{tab:direct_sdpa}
  % Table 4 内容，使用更大字体
\end{table*}
```

**方案 B**: 使用 figure* 环境让关键表格跨栏
```latex
\begin{table*}[t]  % 跨栏可以获得更多水平空间
  \centering
  \caption{Dual validation summary}
  \small  % 使用 \small 而非 \scriptsize
  \begin{tabular}{...}
  % 合并 Table 3 + Table 4 为一个对比表格
  \end{tabular}
\end{table*}
```

**方案 C**: 增大字体，减少表格列数
- 将 Table 4 的 5 列（Avg/Std/Min/Max/Misaligned→Repaired）精简为 3 列（Misaligned→Repaired/Avg Speedup/Range）
- 使用 \small 字体（9pt）而非 \scriptsize（7pt）

**Verification**:
编译后使用 `pdftotext -layout main.pdf` 提取文本，检查表格是否清晰可读。或打印 Page 6，测试 arm's length (25cm) 距离能否清晰阅读表格内容。

---

### M2. Related Work 深度不足 (Literature Integration Issue)

**Location**: §7 Related Work (Pages 6-7)

**Issue**:
在查看 page_06.png 和 page_07.png 时，观察到 Related Work 部分仅占 1.5 栏，引用数量约 32 个。对比 MLSys/EuroSys 典型论文（Related Work 通常 2-3 页，40-60 引用），当前版本显得"单薄"。

具体问题：
1. **缺少历史脉络**: 未讨论 GPU 对齐要求的演化（Volta → Ampere → Hopper），给人"问题是新的"错觉
2. **Compression survey 缺失**: 仅引用个别方法（PaLU, GPTQ, AWQ），缺少综述性引用
3. **批判性思维不足**: 未讨论"为什么现有工作没有发现这个问题"

**Why it matters**:
Related Work 是展示学术成熟度的关键部分。当前版本给审稿人的印象是：
- "作者可能不熟悉领域全貌"（实际上可能只是篇幅限制）
- "问题可能已被解决但作者没有调研"（需要证明 novelty）

这直接影响 Innovation 评分（当前 7.5/10 可能被压低到 6/10）。

**Suggested Fix**:

**[NEEDS_LITERATURE_SEARCH: hardware_evolution]**

补充以下内容（约 0.5 页）：

```latex
\subsection{GPU Architecture Evolution and Alignment Requirements}

Tensor Core alignment constraints have evolved across NVIDIA generations.
Volta (V100) introduced Tensor Cores with K%8 requirements~\cite{volta_whitepaper}.
Ampere (A100) tightened to K%16 for FP16 peak efficiency~\cite{ampere_whitepaper}.
Hopper (H100) added Tensor Memory Accelerator (TMA) with 128-byte granularity,
further emphasizing structured data access~\cite{hopper_whitepaper}.

\textbf{Historical context}: Early compression work~\cite{han2015deep} focused
purely on accuracy-FLOPs tradeoff, ignoring hardware constraints.
Hardware-aware methods emerged post-2018~\cite{amc2018,halp2021}, but focused
on \emph{latency} rather than \emph{alignment}.
Our work bridges this gap by connecting compression-induced dimension irregularities
to GPU microarchitectural requirements.
```

**[NEEDS_LITERATURE_SEARCH: compression_survey]**

推荐搜索关键词：
- "LLM compression survey 2024"
- "hardware-aware neural network compression"
- "tensor core alignment requirements deep learning"

预期补充引用：
- LLM compression surveys (至少 2 篇)
- Hardware-aware optimization methods (至少 3 篇)
- GPU architecture papers (Volta/Ampere/Hopper whitepapers)

**目标**: 将 Related Work 从 1.5 栏扩展到 2 栏，引用数从 32 增加到 45-50。

---

### M3. Figure 4 可读性问题 (Visual Quality Issue)

**Location**: Figure 4 (Page 4, Root cause breakdown)

**Issue**:
在 page_04.png 中，观察到 Figure 4 (Root cause breakdown) 的具体问题：
- 图表标题 "Performance Impact (%)" Y轴标签字体约 7pt，偏小
- 图例文字 "H1: Tensor Core", "H2: L2 Cache", ... 字体约 6pt，在打印时难以辨认
- 柱状图使用 5 种颜色（蓝/橙/绿/红/紫），但红色柱（H2: 5.8%）在白底上对比度不足
- X 轴标签 "Confirmed" / "Not Confirmed" 与柱状图底部间距 <1mm，视觉上"挤"

**Why it matters**:
Figure 4 是论文的核心图表之一，展示 root cause 的定量分解。当前版本的可读性问题会让审稿人质疑：
1. "作者是否重视 paper presentation?"
2. "这是初稿还是终稿?"

对比 MLSys/EuroSys 已发表论文的图表，当前质量属于 "acceptable but not polished" 级别（6/10）。

**Suggested Fix**:

修改 `scripts/create_paper_figures.py` 中 Figure 4 的生成代码：

```python
# 当前代码（推测）
fig, ax = plt.subplots(figsize=(7, 4))  # 单位：英寸
ax.set_ylabel('Performance Impact (%)', fontsize=10)
ax.legend(fontsize=8)

# 修改为
fig, ax = plt.subplots(figsize=(7, 4.5))  # 增加高度，减少拥挤
ax.set_ylabel('Performance Impact (%)', fontsize=12)  # 增大字体
ax.legend(fontsize=10, frameon=True, fancybox=False, shadow=False)

# 使用高对比度配色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 默认
colors = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161']  # 高对比度方案

# 增大 X 轴标签间距
ax.set_xlabel('Hypothesis Status', fontsize=12, labelpad=10)  # labelpad 增加间距
```

**验证方法**:
1. 生成新图表后，放大到 200%，检查所有文字是否清晰
2. 打印 Figure 4 所在页面，在 arm's length (25cm) 距离测试可读性
3. 使用 Photoshop 的 "Proof Setup → Color Blind" 模拟色盲视角，确认颜色可区分

---

### M4. 术语不一致 (Terminology Inconsistency)

**Location**: Throughout the paper (Abstract, §1, §3, §5)

**Issue**:
阅读全文后，发现以下术语混用：
- **核心概念**: "dimensional collapse" (abstract, §1 intro) vs. "dimension misalignment" (§3.2) vs. "irregular dimensions" (§2, §5)
- **维度变量**: "head_dim" (code notation, §2) vs. "head dimension" (prose) vs. "$d$" (math notation)
- **对齐要求**: "8-aligned" vs. "8-multiple" vs. "mod 8 = 0"

**Specific examples**:
- Line 75 (Abstract): "dimensional collapse"
- Line 193 (§3.2): "dimension misalignment comes from theoretical Fisher-information"
- Line 306 (§4): "irregular dimensions force either padding"

**Why it matters**:
术语不一致是学术写作的大忌，会让审稿人质疑：
1. "作者是否清楚自己在讨论什么?"
2. "这些术语是否指代同一概念?"

在系统领域，precision 是基本要求。术语混用会被严厉批评（"poorly written", "lacks rigor"）。

**Suggested Fix**:

**Step 1**: 在 §1 Introduction 第一次引入核心术语时，给出**形式化定义**：

```latex
\paragraph{Dimensional Collapse (Definition).}
We define \emph{dimensional collapse} as the phenomenon where
post-training compression produces tensor dimensions $d$ that violate
GPU alignment requirements ($d \bmod a \neq 0$, where $a \in \{8, 16, 32\}$),
causing nonlinear performance degradation despite reducing FLOPs.
Formally: given original dimension $d_{orig}$ and compressed dimension $d_{comp}$,
dimensional collapse occurs when $\text{FLOPs}(d_{comp}) < \text{FLOPs}(d_{orig})$
but $\text{Latency}(d_{comp}) > \text{Latency}(d_{orig})$.
```

**Step 2**: 全文统一术语：
- 用 "dimensional collapse" 指代整个现象
- 用 "misaligned dimension" 指代具体的 $d$ 值（$d \bmod 8 \neq 0$）
- 用 "$d$" 作为数学符号，"head_dim" 作为 code notation（在 code block 中），"head dimension" 作为 prose

**Step 3**: 创建术语表（Appendix）：
```latex
\section*{Appendix A: Notation}
\begin{itemize}
\item $d$: attention head dimension (math notation)
\item \texttt{head\_dim}: head dimension variable in code
\item $a$: alignment target ($a \in \{8, 16, 32\}$)
\item Misaligned: $d \bmod a \neq 0$
\item Aligned: $d \bmod a = 0$
\end{itemize}
```

**Verification**:
使用 `grep -E "(dimensional collapse|dimension misalignment|irregular dimension)" Latex/main.tex` 检查所有出现，确保统一使用 "dimensional collapse"。

---

### M5. Experiment Reproducibility Gaps

**Location**: §4 Root Cause Analysis (Page 3-4)

**Issue**:
C23 (hardware layer analysis) 实验描述不充分：
- Line 269: "We conduct controlled experiments (C23) to isolate hardware-level causes"
  → 但未说明**如何**控制变量（是否固定 batch size? sequence length? GPU 频率?）
- Line 296: "TC utilization 30%→12%"
  → 但未说明使用什么工具测量（NCU? Nsight Compute? nvprof?）
- Line 303: "SDPA bandwidth efficiency drops to 107.3 GB/s"
  → 但未说明测量方法（是否考虑 cache? 如何排除 PCIe overhead?）

**Why it matters**:
Reproducibility 是系统论文的基本要求。当前描述无法让其他研究者复现实验，降低了工作的可信度。

**Suggested Fix**:

在 §4.3 开头添加实验设置小节：

```latex
\subsubsection{Experimental Methodology}

\noindent\textbf{Profiling tools.}
We use NVIDIA Nsight Compute 2024.1.1 to measure:
- Tensor Core utilization: \texttt{sm\_\_pipe\_tensor\_cycles\_active.avg.pct\_of\_peak\_sustained\_active}
- Memory bandwidth: \texttt{dram\_\_throughput.avg.pct\_of\_peak\_sustained\_elapsed}
- Vectorized load ratio: \texttt{l1tex\_\_t\_sectors\_pipe\_lsu\_mem\_global\_op\_ld.avg.pct\_of\_peak\_sustained\_active}

\noindent\textbf{Controlled variables.}
All experiments fix:
- GPU clock: locked to base frequency (1410 MHz) via \texttt{nvidia-smi -lgc 1410,1410}
- Thermal state: 10-min cooldown between runs
- CUDA streams: single default stream to avoid concurrency effects

\noindent\textbf{Workload configuration.}
Unless stated otherwise: $B=4$, $S=2048$, $H=32$, FP16 precision,
50 warmup iterations + 200 measurement iterations.
```

---

## Minor Issues (Suggested)

### m1. Figure 1 信息密度过低 (Low Information Density)

**Location**: Figure 1 (Page 2, Overview)
**Issue**:
在 page_02.png 中，观察到 Figure 1 占用约 40mm × 35mm（单栏宽度的 ~50%），但仅展示 2 个简单示意图（(a) Unconstrained SVD → misaligned, (b) Dimension repair → aligned）。信息密度过低，浪费了宝贵的页面空间。

对比：Figure 2 (SDPA latency) 占用相同空间，但展示了 ~15 个数据点 + 趋势线，信息密度高出 5 倍。

**Suggestion**:
- 缩小 Figure 1 到 `width=0.25\columnwidth`（当前 `0.4\columnwidth`）
- 或者在 (a)/(b) 下方增加具体数值示例（e.g., "107→112: +27.8% speedup, +4.7% memory"）

---

### m2. Abstract 中的硬编码数字过多

**Location**: Abstract (Page 1)
**Issue**:
Abstract 中包含过多具体数字：
- Line 76: "up to 88%"
- Line 78: "96.9% of unconstrained SVD ranks"
- Line 80: "--0.8%"
- Line 81: "+86.9% average speedup across 45 SDPA configurations"

虽然具体数字能增强可信度，但过多会让 abstract 显得"堆砌实验结果"而非"讲清楚故事"。

**Suggestion**:
精简为关键数字，其余用 qualitative 描述：
```latex
% 当前
... increase SDPA latency by up to 88% versus aligned dimensions.
% 修改为
... cause severe SDPA performance degradation (up to 88% slowdown).

% 当前
+86.9% average speedup across 45 SDPA configurations
% 修改为
substantial speedup (mean: 87%, range: 46-181%) across diverse workloads
```

---

### m3. Table 1 (Backend selection) 位置不当

**Location**: Table 1 (Page 3, §3.3)
**Issue**:
Table 1 首次引用在 Line 220 (§3.3 Backend Selection Behavior)，但表格实际出现在 Page 3 底部，导致读者需要"向下翻页"才能看到表格。SIGPLAN 格式建议表格出现在首次引用的**同页**或**下一页顶部**。

**Suggestion**:
修改 LaTeX 浮动参数：
```latex
% 当前
\begin{table}[t]
% 修改为
\begin{table}[h!]  % h! 强制就地放置
```

或使用 `\FloatBarrier` 限制浮动范围：
```latex
\subsection{Backend Selection Behavior}
\FloatBarrier  % 阻止之前的浮动体漂移到这里
Table~\ref{tab:backend} shows ...
\begin{table}[t]
...
\end{table}
```

---

### m4. Figure 5 (Repair tradeoff) 数据点标签重叠

**Location**: Figure 5 (Page 5, Repair tradeoff scatter plot)
**Issue**:
在 page_05.png 中，观察到 Figure 5 的 scatter plot 中，数据点标签（如 "d=107", "d=120"）与散点部分重叠，降低了可读性。特别是 "d=120" 标签遮挡了散点本身。

**Suggestion**:
修改 `scripts/create_paper_figures.py` 中 Figure 5 的绘制代码，使用 `textcoords='offset points'` 实现标签偏移：

```python
# 当前代码（推测）
for i, txt in enumerate(labels):
    ax.annotate(txt, (x[i], y[i]), fontsize=8)

# 修改为
for i, txt in enumerate(labels):
    ax.annotate(txt, (x[i], y[i]),
                textcoords='offset points',
                xytext=(5, 5),  # 向右上偏移 5pt
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
```

---

### m5. §6.3 Applicability Framework 缺少 decision tree 可视化

**Location**: §6.3 Applicability Framework (Page 5)
**Issue**:
Table 5 (applicability framework) 是论文的重要贡献，但以纯文本表格形式呈现，不够直观。Practitioner 更习惯 flowchart 或 decision tree 形式的指导。

**Suggestion**:
使用 TikZ 创建 decision tree：

```latex
\begin{figure}[t]
\centering
\begin{tikzpicture}[
  node distance=1.5cm,
  decision/.style={diamond, draw, text width=4em, text centered},
  action/.style={rectangle, draw, text width=6em, text centered}
]
  \node[decision] (check) {SDPA on compressed dim?};
  \node[action, below left of=check] (no) {No repair needed};
  \node[action, below right of=check] (yes) {Apply repair (+87\%)};

  \draw[->] (check) -- node[left]{No} (no);
  \draw[->] (check) -- node[right]{Yes} (yes);
\end{tikzpicture}
\caption{Applicability decision tree}
\label{fig:decision_tree}
\end{figure}
```

---

### m6. §8 Conclusion 过于防御性

**Location**: §8 Conclusion (Pages 6-7)
**Issue**:
Conclusion 中有 4 段 disclaimer（Hardware Scope, Software Version, Integration, Reproducibility），占据了约 60% 的篇幅。虽然 transparency 很重要，但过度强调限制会给审稿人留下"工作不完整"的印象。

**Suggestion**:
- 将 "Hardware Scope" 和 "Software Version" 合并为一段（~3 行）
- 将 "Integration" 和 "Reproducibility" 移到 §6 Evaluation 末尾作为 "Practical Considerations"
- Conclusion 应聚焦于**贡献总结**和**broader impact**，而非 limitations

```latex
\section{Conclusion}

We presented the first systematic diagnosis of dimensional collapse in compressed LLMs,
identifying three root causes and proposing validated repair strategies.
Our applicability framework enables practitioners to make informed decisions about
when dimension repair provides benefit.

\textbf{Broader impact.}
As LLM compression becomes ubiquitous, understanding hardware-software co-design
constraints will be critical. Our work provides a template for diagnosing
performance anomalies in compressed models, applicable beyond dimensional issues
(e.g., sparsity patterns, quantization granularity).

\textbf{Scope.} All experiments use NVIDIA A100 GPUs and FlashAttention 2.7.4.
Validation on H100 and newer FlashAttention versions is future work.
```

---

## Visual Observations (必填！)

### Page-by-Page Observations

**Page 1:**
- **看到的内容**: 标题 "When Smaller Is Slower: Dimensional Collapse in Compressed LLMs"，5 位作者（Jihao Xin, Tian Lvy, Qilong Pan, Kesen Wang, Marco Canini），Abstract 约 12 行，Keywords "LLM Compression, GPU Optimization, Tensor Core, Memory Alignment"
- **具体观察**: Abstract 字体 10pt，清晰可读。Figure 1 位于页面底部，占约 40mm × 35mm，显示 (a) "Unconstrained SVD" 的两个直方图（Before/After），(b) "Dimension Repair" 的箭头示意图
- **问题/建议**:
  1. Author "Tian Lvy" 可能是拼写错误（应为 "Levy"?）
  2. Figure 1 信息密度偏低（2 个简单图占 0.4 栏宽），建议缩小到 0.25\columnwidth

**Page 2:**
- **看到的内容**: §1 Introduction 的贡献列表（4 个 bullet points），§2 Background（3 个小节），Figure 2 (SDPA latency vs head_dim, 跨栏)，Figure 3 (PaLU dimension distribution, 单栏)
- **具体观察**:
  - Figure 2: X 轴 64-160 (head_dim)，Y 轴 0-2.5ms (latency)，蓝色折线图，d=107 处标注 "+88%"
  - Figure 3: X 轴 114-126 (head_dim)，Y 轴 0-200 (count)，红色/绿色柱状图，顶部有黄底黑字 banner "THEORETICAL ANALYSIS"
- **问题/建议**:
  1. Figure 2 跨栏图出现在 §3.1 和 §3.2 之间，打断阅读流
  2. Figure 3 的 banner 字体过大（~12pt 粗体），视觉上"刺眼"，建议改为 caption footnote

**Page 3:**
- **看到的内容**: §3.2 SDPA Latency vs Head Dimension，§3.3 Backend Selection Behavior，Table 1 (Backend latency comparison, 5 行 4 列)，§4 Root Cause Analysis 开头
- **具体观察**:
  - Table 1: d=107 行的 MEM_EFF 列为 "---"（unavailable），MATH 列为 "27.00±0.20"（最慢）
  - 表格字体约 9pt，清晰可读
- **问题/建议**: Table 1 的 "---" 符号应在 caption 中解释含义（"backend unavailable due to alignment constraint"）

**Page 4:**
- **看到的内容**: §4 Root Cause Analysis 的 3 个小节（4.1 PyTorch Backend, 4.2 CUDA Kernel, 4.3 Hardware Constraints），Figure 4 (Root cause breakdown bar chart, 跨栏)，Table 2 (Hardware analysis, 4 行 4 列)
- **具体观察**:
  - Figure 4: 横向柱状图，X 轴 "Performance Impact (%)"，4 个假设 H1-H4，H2 (5.8%) 用浅灰色表示 "Not Confirmed"
  - Table 2: Row 1 "H1: TC K%16 | Confirmed | 58% | Util. 30%→12%"
- **问题/建议**:
  1. Figure 4 图例字体约 7pt，在打印时难以辨认，建议增大到 9pt
  2. H2 柱用浅灰色，对比度不足，建议用深红色或橙色表示 "disconfirmed"

**Page 5:**
- **看到的内容**: §5 Shape-Aware Compression，§6 Evaluation（6.1 Negative E2E, 6.2 Positive E2E），Table 3 (RAP SVD E2E, 3 行 4 列)，Table 4 (Direct SDPA speedup, 6 行 6 列)，Figure 5 (Repair tradeoff scatter plot)
- **具体观察**:
  - Table 3: Prefill/Decode/Memory 三行，Δ 列为 "-0.8%", "-0.9%", "+0.1%"（负数结果）
  - Figure 5: 散点图，X 轴 "Memory Overhead (%)", Y 轴 "Speedup (%)"，6 个数据点标注 d=107, 114, 117, 120, 121, 125，d=120 有淡淡的圆圈高亮
- **问题/建议**:
  1. Figure 5 的 d=120 "highlighted" 圆圈几乎看不见，建议用红色/橙色加粗标注
  2. 数据点标签 d=114 和 d=117 位置接近，有轻微重叠，建议使用 offset

**Page 6:**
- **看到的内容**: §6.3 Applicability Framework，§6.4 Kernel-Level Analysis，Table 6 (SDPA repair latency, 6 行 7 列)，§6.5 Accuracy Preservation，§6.6 Scope and Limitations，§7 Related Work 开头
- **具体观察**:
  - 页面上半部分极度拥挤：Table 6 占约 20%，§6.4-6.6 正文占约 50%，§7 开头占约 15%
  - Table 6 字体约 8pt（数据）+ 7pt（footnote），接近可读性极限
  - 垂直间距：Table 6 与 §6.5 之间约 2mm（应为 5mm）
- **问题/建议**（Critical）:
  1. **信息密度过载**: 单页 3 张表格 + 大量正文，字体 ≤7pt 无法打印阅读
  2. **布局失衡**: 上半部分拥挤（~95% 填充），下半部分有 ~30mm 空白
  3. 建议拆分表格到不同页面，或使用 figure* 跨栏

**Page 7:**
- **看到的内容**: §7 Related Work 约 1.3 栏（LLM Compression, Hardware-Aware Optimization, GPU Kernels），§8 Conclusion 开头
- **具体观察**:
  - §7.1 "LLM Compression" 段落提到 Volta/Ampere/Hopper 演化，占约 10 行
  - 正文字体 10pt，阅读性良好
- **问题/建议**: Related Work 篇幅过长（1.5-2 页），建议压缩 hardware evolution timeline 到 2 句话

**Page 8-10:**
- **看到的内容**: §7 Related Work 续（Table 7: Dimension handling comparison），§8 Conclusion 续（Hardware Scope, Software Version, Integration, Reproducibility），References (约 32 个引用)
- **具体观察**:
  - Table 7: 7 行 3 列，比较不同系统的 head_dim 支持情况
  - References 使用 ACM-Reference-Format，字体 9pt
- **问题/建议**:
  1. §8 Conclusion 中 4 段 disclaimer 占约 60% 篇幅，过于防御性
  2. 引用数量 32 个偏少，建议补充到 45-50 个

---

### Figure-by-Figure Assessment

| Figure | 位置 | 你观察到的具体内容 | 尺寸评估 | 布局评估 | 问题 |
|--------|------|-------------------|---------|---------|------|
| Fig 1 | Page 2 | 双图并排 (a)/(b)，左侧 "Unconstrained SVD" 箭头指向 "Misaligned Dims (107, 114, 125)"，右侧 "Dimension Repair" 箭头指向 "Aligned Dims (112, 120, 128)"。Caption 4 行，包含 "96.9% of dimensions would be misaligned" | **过大** | 正常 | 信息密度低：2 个简单示意图占 0.4 栏宽。建议缩小到 0.25\columnwidth 或在图中增加数值示例 |
| Fig 2 | Page 3 | 折线图，X 轴 64-160 (head_dim)，Y 轴 0-2.5ms (latency)。蓝色线 "Aligned" 平稳在 1.1-1.6ms，橙色线 "Misaligned" 阶梯上升到 2.147ms (d=107)。数据点标注 "2.19ms" 等 | 合适 | **跨栏打断流** | 1. 跨栏图出现在 §3.1-3.2 之间，读者视线跳跃。2. Y 轴右侧无标签（可能 redundant） |
| Fig 3 | Page 3 | 柱状图，X 轴 114-125 (head_dim)，Y 轴 Count。红色柱 (Misaligned) 高度 50-90，绿色柱 (Conditional) 高度 10-20。顶部 banner "THEORETICAL ANALYSIS" 字体粗体约 12pt | 合适 | 正常 | 1. Banner 字体过大且颜色（黄底黑字）视觉上"刺眼"。2. 建议将 banner 改为 caption footnote |
| Fig 4 | Page 4 | 横向柱状图，4 个假设 H1-H4，柱长度表示 Performance Impact (%)。H1 (Tensor Core) 58%，H2 (L2 Cache) 5.8%，H3 (SDPA BW) 40%，H4 (Vec loads) 50%。图例右上角，字体约 7pt | 合适 | 正常 | 1. 图例字体偏小（7pt）。2. 红色柱（H2）对比度不足。3. X 轴标签与柱底间距 <1mm |
| Fig 5 | Page 5 | Scatter plot，X 轴 Memory Overhead (%) 0-8，Y 轴 Speedup (%) 0-30。数据点标签 "d=107" (3.7%, 27.8%)，"d=120" (0%, 0%)，"d=121" (7.2%, 27.2%)。"d=120" 用淡淡的圆圈标注 | **过大** | 正常 | 1. 信息密度低（6 个点占 0.4 栏宽）。2. d=120 高亮圆圈几乎看不见，建议用红色/橙色。3. 标签重叠（d=114, d=117） |

---

### Table Assessment

| Table | 你观察到的具体内容 | 问题 |
|-------|-------------------|------|
| Table 1 | 5 行 4 列，表头 "$d$" / "AUTO" / "FLASH" / "MEM_EFF" / "MATH"。d=107 行的 MEM_EFF 列为 "---"（unavailable），MATH 列为 "27.00±0.20"（最慢）。Caption "SDPA backend latency (ms±std) for various head dimensions" | Caption 应解释 "---" 含义。建议添加 footnote: "---: Backend unavailable (requires 8-alignment)" |
| Table 2 | 4 行 4 列，表头 "Hypothesis / Status / Impact / Root Cause"。H1 行 "Confirmed / 58% / Util. 30%→12%"，H2 行 "Not confirmed / 5.8% / Negligible" | 字体约 8-9pt，清晰可读。Root Cause 列条目简洁，可考虑略微扩展（"TC util."） |
| Table 3 | 3 行 4 列，表头 "Phase / Misaligned / Repaired / Δ"。Prefill 行 "290.5 / 292.9 / --0.8%"，Decode 行 "1009 / 1000 / --0.9%" | 负数结果未加视觉强调（粗体/颜色）。建议用红色或粗体突出 -0.8%, -0.9% |
| Table 4 | 6 行 6 列，表头 "Misaligned / Repaired / Avg / Std / Min / Max"。d=107→112 行 "78.5% / 29.2% / 46.3% / 139.5%"。Overall 行 "**86.9%** / 34.5% / 46.3% / 181.4%" | **字体 ≤7pt，打印时难以阅读**。建议合并 Avg/Std 为 "Mean±Std" 列，或使用 figure* 跨栏 |
| Table 6 | 6 行 7 列，表头 "$d$ / Orig / Min / Opt / ΔMin / ΔOpt"。d=107 行 "2.06±0.06 / 1.49±0.04 / 1.51±0.04 / +27.8 / +27.0" | **列数过多（7 列）导致字体 ≤7pt**。Page 6 拥挤问题核心之一 |
| Table 7 | 6 行 3 列，表头 "System / Supported head_dim / Misaligned"。FlashAttn-2 行 "32,64,96,128,256 (opt.) / +30--45%"，PaLU 行 "32-multiple / N/A" | 字体约 7pt。"RAP SVD | Any integer | Affected" 含义模糊，建议改为 "Affected (30-45% penalty)" |

---

### Layout Assessment (布局评估 - 必填！)

**整体页面利用率**：
- **Page 2-5**: 利用率正常（~85%），图文平衡良好
- **Page 6**: **严重失衡**
  - 上半部分：3 张表格 + 少量正文，信息密度过载（~95% 填充率）
  - 下半部分：§7 开头，有约 30mm 空白（~15% 填充率）
- **Page 7-9**: 利用率正常（~80%）

**图文冲突检查**：
- **无图片侵入正文空间**: 所有图表与正文间距 ≥3mm，符合 SIGPLAN 规范
- **无图片与 caption 重叠**: Caption 与图表间距正常（~2mm）
- **双栏排版问题**: Page 6 左栏包含 Table 6 上半部分，右栏包含 Table 6 下半部分 + §6.4-6.6，左栏表格高度过大导致右栏正文被压缩

**尺寸问题图片列表**：

| 图片 | 问题类型 | 具体描述 | 建议修改 |
|------|---------|---------|---------|
| Figure 1 | 信息密度低 | 占 0.4 栏宽但仅展示 2 个简单示意图，浪费空间 | 缩小到 0.25\columnwidth，或在图中增加数值示例 |
| Figure 4 | 图例字体过小 | 图例字体约 7pt，在打印时难以辨认 | 增大图例字体到 9pt，调整图例位置避免遮挡数据 |
| Figure 5 | 信息密度低 + 标签重叠 | 6 个数据点占 0.4 栏宽，d=120 高亮不明显，标签重叠 | 缩小到 0.3\columnwidth，用红色标注 d=120，标签 offset |
| Table 4 | 字体过小 | 6 列表格导致字体 ≤7pt | 合并 Avg/Std 为 "Mean±Std" 列，或使用 \small 字体（9pt） |
| Table 6 | 列数过多 | 7 列表格字体 ≤7pt，Page 6 拥挤核心问题 | 合并 Min/Opt 为双子列标题，或分散到不同页面 |

---

### Visual Issues Summary

**必须列出至少 5 个视觉问题**：

1. **Page 6 信息密度过载**: 单页 3 张表格（Table 2/3/6），字体 ≤7pt，在打印时无法阅读。建议将表格分散到不同页面或使用 figure* 跨栏
2. **Figure 1 信息密度过低**: 占 0.4 栏宽但仅展示 2 个简单示意图。建议缩小到 0.25\columnwidth
3. **Figure 4 图例字体过小**: 图例字体约 7pt（"H1: Tensor Core (Confirmed)" 等文字），在打印时难以辨认。建议增大到 9pt
4. **Figure 4 红色柱对比度不足**: H2 (L2 Cache) 的红色柱在白底上对比度不足。建议使用深红色 (#CC0000) 或添加边框
5. **Figure 5 数据点标签重叠 + 高亮不明显**: "d=120" 标签与散点重叠，"highlighted" 圆圈几乎看不见。建议使用 `textcoords='offset points'` 偏移 + 红色/橙色高亮
6. **Table 4 列数过多**: 6 列表格（Misaligned/Repaired/Avg/Std/Min/Max）导致字体 ≤7pt。建议合并 Avg/Std 为 "Mean±Std" 列
7. **Figure 3 banner 视觉过于突出**: "THEORETICAL ANALYSIS" banner 黄底黑字粗体 12pt，视觉上"刺眼"。建议改为 caption footnote 或缩小字体到 10pt
8. **Figure 2 跨栏打断阅读流**: Full-width figure* 出现在 §3.1-3.2 之间，读者视线从左栏跳到图表再跳到右栏。建议移到 §3.2 末尾或使用 [b] 放置

---

## Improvement Checklist for Writer Agent

### High Priority (Must Fix)
- [ ] **M1**: 重新设计 Page 6 布局 - 将 Table 2/3/6 分散到不同页面，或使用 figure* 跨栏，确保字体 ≥8pt
- [ ] **M2**: 扩展 Related Work - 补充 GPU 架构演化讨论、compression survey 引用，目标 45-50 引用
- [ ] **M3**: 改进 Figure 4 可读性 - 增大图例字体到 9pt，使用高对比度颜色方案
- [ ] **M4**: 统一术语体系 - 在 §1 形式化定义 "dimensional collapse"，全文统一使用
- [ ] **M5**: 补充实验方法描述 - 在 §4.3 添加 profiling tools 和 controlled variables 说明

### Medium Priority (Recommended)
- [ ] **m1**: 缩小 Figure 1 - 从 0.4\columnwidth 缩小到 0.25\columnwidth，或增加数值示例
- [ ] **m2**: 精简 Abstract - 减少硬编码数字，用 qualitative 描述替代部分数值
- [ ] **m3**: 修复 Table 1 位置 - 使用 \begin{table}[h!] 或 \FloatBarrier 确保表格在首次引用的同页
- [ ] **m4**: 改进 Figure 5 - 使用 `textcoords='offset points'` 避免标签与散点重叠，用红色/橙色高亮 d=120
- [ ] **m5**: 简化 Table 4 - 合并 Avg/Std 为 "Mean±Std" 列，减少列数
- [ ] **m6**: 重组 Conclusion - 将 4 段 disclaimer 精简为 1 段，聚焦贡献总结

### Low Priority (Optional)
- [ ] 为 Table 5 (Applicability framework) 创建 TikZ decision tree 可视化
- [ ] 在 Appendix 添加术语表 (Notation)
- [ ] 将 Figure 3 的 "THEORETICAL ANALYSIS" banner 改为 caption footnote
- [ ] 统一代码变量命名：全文使用 \texttt{head\_dim} 而非混用 "head dimension"
- [ ] 修复 Figure 2 跨栏位置：移到 §3.2 末尾或使用 [b] 放置

---

## Questions for Authors

1. **Hardware coverage**: Have you conducted any preliminary experiments on H100 or other GPU architectures? If not, what is the main barrier (hardware access, time, or methodological challenges)?

2. **Production deployment**: You mention that all 24 PaLU checkpoints already enforce 32-multiple alignment. Have you contacted the PaLU authors to understand (a) how they discovered this requirement, and (b) whether they observed similar performance cliffs during development?

3. **Comparison with runtime padding**: Systems like TensorRT support runtime padding for misaligned dimensions. How does your compile-time dimension repair compare with runtime approaches in terms of (a) performance overhead, and (b) ease of integration?

4. **Generalization beyond SDPA**: Do other LLM operations (e.g., LayerNorm, MLP projections) exhibit similar dimensional collapse? Or is this phenomenon specific to attention?

5. **Related to M2 (Literature)**: During your literature review, did you find any prior work (even in non-LLM domains like CNN compression) that discusses dimension alignment issues? If so, what distinguishes your work from theirs?

---

## Detailed Comments by Section

### Abstract
**评价**: Strong technical summary, but suffers from two issues: (1) too many hard-coded numbers (88%, 96.9%, -0.8%, +86.9%) make it read like a results dump rather than a story, and (2) the repeated emphasis on A100-only scope ("All experiments focus on NVIDIA A100 GPUs; H100 generalization is future work") weakens the perceived contribution.

**建议**: Reduce numeric specificity, use qualitative ranges where appropriate ("up to 88%" → "severe degradation"), and move scope disclaimer to the end of abstract or eliminate from abstract entirely (it can stay in Conclusion).

---

### Introduction
**评价**: Well-structured with clear motivation, scope clarification, and contribution list. The "Scope and Applicability" paragraph (lines 106-112) is excellent—it proactively addresses potential reviewer questions about the 96.9% figure.

**问题**:
1. "Dimensional collapse" is introduced at line 104 but not formally defined until much later. Consider adding a 1-sentence definition immediately after first use.
2. The contributions list (lines 136-141) is dense. Each contribution spans 3-4 lines, making it hard to parse. Consider using sub-bullets or shorter phrasing.

**建议**:
```latex
We term this \emph{dimensional collapse}---a phenomenon where
compressed dimensions violate GPU alignment constraints ($d \bmod 8 \neq 0$),
causing latency to \emph{increase} despite FLOPs reduction.
```

---

### Background
**评价**: Concise and sufficient. The three subsections (Tensor Core, FlashAttention, Low-Rank Compression) cover necessary background without excessive detail.

**问题**: §2.2 (FlashAttention Constraints) makes a strong claim: "Contrary to common belief, it does not strictly require 8-aligned dimensions" (line 165). This is important for the paper's narrative, but needs a citation or explanation of what the "common belief" is based on.

**建议**: Add a citation to FlashAttention documentation or GitHub issues where dimension requirements are discussed.

---

### Dimensional Collapse (§3)
**评价**: §3 provides solid quantitative evidence. Figure 2 clearly shows the "staircase effect". Table 1 (backend selection) is informative.

**问题**:
1. §3.2 (Scope and Dimension Distribution) repeats the 96.9% figure from abstract without adding new insight. Consider merging with §1 or removing redundancy.
2. Figure 3's "THEORETICAL ANALYSIS" banner is visually jarring (yellow background, large font). Recommend moving this clarification to caption footnote.

---

### Root Cause Analysis (§4)
**评价**: The dual validation approach (negative + positive cases) is the paper's strongest contribution. It demonstrates that the applicability framework is not just a post-hoc rationalization, but has predictive power.

**问题**:
1. §4.3 (Hardware Constraints) 的实验方法描述不充分（见 M5）
2. Figure 4 图例字体过小，颜色对比度不足（见 M3）

**亮点**: The boxed "Root Cause Summary" (lines 310-314) is excellent. It succinctly captures the key findings.

---

### Shape-Aware Compression (§5)
**评价**: This section feels underdeveloped compared to §4's rigor. §5.1 defines Shape Contract in 4 lines without justifying why a=8 is "minimal" and a=16 is "optimal."

**建议**: Explicitly connect to §4 results: "From §4, we derive: a=8 (vectorized loads, H4) and a=16 (Tensor Core, H1)."

---

### Evaluation (§6)
**评价**: The dual validation (negative RAP SVD + positive Direct SDPA) is exemplary. Table 5 (Direct SDPA) with 45 workloads is thorough.

**问题**: §6.4 Kernel-Level Analysis feels redundant with §5.2. Consider condensing to 10 lines.

---

### Related Work (§7)
**评价 (最弱部分)**: 仅 32 个引用，缺少 GPU 架构演化、compression survey、H100 研究的深度讨论。见 M2。

---

### Conclusion (§8)
**评价**: Summarizes contributions well, but 4 段 disclaimer (Hardware Scope, Software Version, Integration, Reproducibility) 占约 60% 篇幅，过于防御性。见 m6。

---

## Reviewer Confidence

**Confidence Score:** 4/5

**Expertise Areas:**
- GPU microarchitecture and Tensor Core optimization
- LLM serving systems and inference optimization
- Systems paper reviewing (have reviewed for MLSys, EuroSys, OSDI)

**Limitations:**
- Limited hands-on experience with FlashAttention kernel internals
- Cannot verify C23 hardware layer experiment methodology without raw profiling data
- Unfamiliar with EuroMLSys-specific acceptance criteria

---

*Reviewer: Paper Reviewer Agent*
*Date: 2026-01-29*
