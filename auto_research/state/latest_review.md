# Paper Review: When Smaller Is Slower: Dimensional Collapse in Compressed LLMs

**Target Venue:** EuroMLSys (SIGPLAN format, 6 pages main content, references and appendix unlimited)
**Review Date:** 2026-01-28
**Reviewer:** Paper Reviewer Agent

---

## Summary

This paper investigates a counterintuitive performance degradation phenomenon in compressed LLMs—dimensional collapse—where post-training compression produces irregular tensor dimensions that cause GPU slowdowns despite reducing FLOPs. The authors systematically measure SDPA latency across head dimensions (64-160) on NVIDIA A100, showing that misaligned dimensions (e.g., head_dim=107) incur 88% latency overhead versus aligned dimensions (head_dim=96). Through controlled hardware experiments, they diagnose three primary root causes: Tensor Core misalignment (58% impact), vectorized load degradation (50%), and SDPA bandwidth inefficiency (40%), while disconfirming L2 cache sector waste (5.8%) as significant. They propose lightweight dimension repair via zero-padding, validated through contrasting experiments: RAP SVD shows -0.8% (correctly predicted no benefit for projection-based architectures), while direct SDPA benchmarks show 86.9% average speedup across 45 workloads. The work provides an applicability framework (Table 7) to guide practitioners on when dimension repair helps versus when it does not.

The paper targets a niche but technically rigorous problem. The diagnostic methodology is thorough, using isolation experiments to separate hardware-level effects. The contrasting validation (negative + positive cases) demonstrates intellectual honesty. However, the presentation suffers from excessive figure sizes, layout conflicts in double-column format, and limited literature engagement (46 citations but shallow historical context).

---

## Overall Rating

**Rating: Weak Accept (7/10)**

This is valuable diagnostic work with solid experimental methodology and honest negative results. The 96.9% misalignment claim is properly scoped to theoretical analysis. The applicability framework demonstrates predictive power through contrasting cases. However, the paper feels incomplete: Related Work lacks critical engagement with the history of hardware-aware compression and GPU performance modeling; visual presentation has multiple layout issues (figures invading text space, inconsistent sizing); and the E2E validation remains incomplete (RAP SVD validates when repair doesn't help, but no E2E validation showing when it does help beyond kernel microbenchmarks). The paper makes a solid contribution to the MLSys community by documenting this pitfall and providing diagnostic guidance, but presentation polish and literature depth need improvement to reach Accept threshold.

**Confidence:** 4/5 (high familiarity with MLSys evaluation, some uncertainty on EuroMLSys acceptance bar)

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

**主要瓶颈维度**: Paper Presentation

**瓶颈分数**: 6.0/10

**为什么是瓶颈**:

Paper Presentation 是当前最大瓶颈，因为：

1. **布局问题显著影响可读性**：Page 6 有大片空白未利用（右栏 ~40% 空白）；Page 8/10 同样有 50-60% 空白；而Page 3 过于拥挤（2 figures + 1 table + dense text）
2. **图片尺寸不当**：Figure 4 (root cause breakdown) 过小且关键信息难读（bar labels ~7-8pt）；Figure 3 (histogram) 信息密度低但占用大量空间
3. **信息密度不均**：某些页面过于密集（Page 2-3），某些页面过于稀疏（Page 6 空白，Page 8-10 空白）
4. **阻碍更高评分**：即使 Technical Quality 提升到 8.5/10，Paper Presentation 的 6.0/10 也会将总分压在 ~7.4/10，无法突破 Accept (8/10) 门槛

**突破方向**:

提升 Paper Presentation 到 7.5-8.0 可使总分达到 7.8-8.0：
- 修复所有布局问题（消除 Page 6/8/10 空白区域，调整 float placement）
- 调整图表尺寸（Figure 4 放大确保可读性，Figure 3 缩小提高空间效率）
- 统一图表风格（字体大小 ≥9pt，颜色方案一致，标注方式专业）
- 提升信息密度（将 Related Work 从简单列举改为批判性讨论，填充空白区域）

**给 Planner 的建议**:

1. **高优先级**：FIGURE_SIZE_OPTIMIZATION + LAYOUT_FIX 任务
   - 放大 Figure 4 (root cause) 从 ~0.5 columnwidth 到 0.75-0.85 columnwidth，确保 bar labels 可读（当前 7-8pt → 目标 9-10pt）
   - 填充 Page 6 右栏空白（添加 H100 讨论、Limitations 扩展、或系统对比表）
   - 重新分配 Page 8-10 空白（调整 float placement 或添加 Appendix 内容）
   - 缩小 Figure 3 从 0.85 columnwidth 到 0.6 columnwidth（信息密度低）

2. **中优先级**：RELATED_WORK_ENRICHMENT 任务
   - 添加历史脉络：GPU 架构演进如何塑造对齐需求（Volta → Ampere → Hopper）
   - 批判性讨论：为什么 prior work 忽略了对齐问题？
   - 对比分析：与 TensorRT implicit padding、vLLM dimension handling 的详细对比
   - 目标：从 46 citations → 60+ citations，从列举式 → 分析式

3. **中优先级**：FIGURE_POLISH 任务
   - 统一字体大小：axis labels ≥11pt, tick labels ≥9pt, legend ≥9pt
   - 统一颜色方案：blue for aligned, orange for misaligned across ALL figures
   - 修复标签重叠：Figure 5 scatter plot labels 使用 leader lines

**不建议**：纯文字修改（Writing Quality 已达 7.0，性价比低）或大规模新增实验（Technical Quality 7.5 已足够 Weak Accept）

---

## Strengths

1. **Honest Negative Results**: The RAP SVD experiment (-0.8%, Table 6) demonstrates intellectual honesty. Many papers would hide this result; the authors instead turn it into a key contribution (applicability framework validation). This is commendable.

2. **Rigorous Root Cause Diagnosis**: The hardware layer experiments (§4.3, Table 2) use proper isolation. Testing each hypothesis independently (H1: TC K%16, H2: L2 sector, H3: SDPA BW, H4: Vec loads) and disconfirming H2 shows scientific rigor. The 58%, 50%, 40%, 5.8% breakdown is credible.

3. **Proper Scope Clarification**: The 96.9% misalignment figure is correctly described as "theoretical Fisher-information-based analysis" (Abstract, §1, §3.2), not production PaLU. The repeated disclaimers throughout the paper prevent misinterpretation.

4. **Actionable Practitioner Guidance**: Table 7 (Applicability Framework) provides clear decision criteria. The "SDPA operates on misaligned dimensions" distinction is practical—compression method designers can immediately apply this.

5. **Reproducibility Emphasis**: Experimental configurations are detailed (§3.1: PyTorch 2.9.1, CUDA 12.8, FlashAttention 2.7.4, Driver 560.35.03). Variance acknowledgment (5-8% run-to-run) shows measurement awareness.

---

## Weaknesses

1. **Layout and Visual Presentation Issues**: The paper has significant layout problems that detract from professional appearance. Page 6 has ~40% blank space (right column), Page 8-10 have 50-60% blank space, while Page 2-3 are overcrowded. Figure 4 is too small (critical data difficult to read), Figure 3 has low information density but occupies large space.

2. **Literature Engagement Shallow**: 46 citations but minimal critical discussion. Related Work (§7) is mostly list-like. Missing: historical evolution of hardware-aware compression, prior work on dimension handling in inference systems, critical analysis of why prior compression methods ignored alignment.

3. **E2E Validation Incomplete**: The paper has kernel-level validation (86.9% speedup, Table 8) and negative E2E validation (RAP SVD -0.8%, Table 6), but no positive E2E validation showing end-to-end improvement on a real model. The jump from "kernel speedup" to "this will help practitioners" needs an E2E bridge.

4. **Terminology Inconsistency**: "dimensional collapse" is used throughout, but never formally defined in a definition box. The term is self-coined without sufficient justification.

5. **Figure Quality Below Venue Standards**: Compared to published MLSys/EuroSys papers, the figures lack professional polish. Font sizes are borderline readable (~7-8pt in some labels), Figure 4 bar chart is too small, Figure 5 has overlapping labels.

---

## Major Issues (Must Fix)

### M1. E2E Positive Validation Missing

**Location**: §6 Evaluation, Table 8

**Issue**: The paper provides kernel-level validation (Table 8: 86.9% SDPA speedup) and negative E2E validation (Table 6: RAP SVD -0.8%), but lacks positive E2E validation. The leap from "kernel microbenchmark shows 86.9% speedup" to "practitioners should apply dimension repair" is too large without E2E evidence on a real directly-compressed model.

**Why it matters**: Reviewers will ask: "Does the 86.9% kernel speedup translate to meaningful E2E improvement?" The paper correctly identifies when repair doesn't help (projection-based), but doesn't show E2E evidence for when it does help (direct compression).

**[NEEDS_LITERATURE_SEARCH: competitive_analysis]**
Suggested searches:
- "vanilla SVD LLM compression implementations"
- "direct head dimension compression methods"
- "models using direct SVD without projection layers"

**Suggested Fix**:
1. Implement a minimal directly-compressed model (vanilla SVD on Q/K/V projections, no projection layers) and measure E2E prefill/decode latency before/after repair
2. Alternatively, reframe contribution: "We provide kernel-level evidence and applicability framework; integrating repair into production compression methods is future work" (more honest but weakens impact)
3. Add limitations paragraph: "While we validate kernel-level speedup (86.9%) and correctly predict negative cases (RAP SVD -0.8%), E2E validation on directly-compressed models is future work"

### M2. Figure 4 (Root Cause Breakdown) Too Small / Hard to Read

**Location**: Page 4, Figure 4

**Issue**: I observe from page_04.png that Figure 4 is critical (visualizes the 58%, 50%, 40%, 5.8% breakdown) but currently rendered too small (~0.5 columnwidth estimated). The horizontal bar chart has labels "Tensor Core K%16: 58%", "Vectorized Load (K%8): 50%", "Bandwidth Sectors: 40%", "L2 Cache Sectors: 5.8%" in approximately 7-8pt font, barely readable at print size. The bars use red/orange/gray colors.

**Why it matters**: This is the KEY DIAGNOSTIC FIGURE justifying the paper's root cause claims. If reviewers can't read the labels clearly, they may doubt the rigor. Figure 4 is more important than Figure 3 (simple histogram) but receives less space.

**Suggested Fix**:
1. Increase Figure 4 width to 0.75-0.85\columnwidth (similar to Figure 2)
2. Increase font size for bar labels to 9-10pt minimum
3. Consider moving to single-column width (\columnwidth) if necessary
4. Verify readability by printing a physical copy—if you need to squint, it's too small
5. Update matplotlib code:
   ```python
   fig, ax = plt.subplots(figsize=(6, 3))  # Wider figure
   plt.rcParams['font.size'] = 10
   ax.set_xlabel('Performance Impact (%)', fontsize=11)
   ```

### M3. Page 6 Excessive Whitespace / Poor Space Utilization

**Location**: Page 6 (right column)

**Issue**: From page_06.png visual inspection, I observe that Page 6 has a large blank area (approximately 40% of page height) in the right column. The page contains Table 6 (RAP SVD results) and Table 7 (Applicability Framework) at the top, followed by text from §6.3, §6.4, then §7 Related Work starts. The right column after Table 7 has substantial whitespace before §7 begins.

**Why it matters**: Space is precious in a 6-page paper. The blank area suggests inefficient float placement or missing content. This looks unprofessional and raises questions about whether the paper has sufficient depth. With a 6-page limit, wasting ~20% of a page is problematic.

**Suggested Fix**:
1. Move H100 discussion (currently in Conclusion §8) forward to fill the Page 6 space
2. Expand Related Work §7 to start earlier on Page 6 (currently compressed on Page 7)
3. Add a "Limitations" subsection expansion in §6 discussing E2E validation gaps
4. Use \FloatBarrier and [t]/[b] placement specifiers to control figure drift:
   ```latex
   \FloatBarrier  % After §6 Evaluation
   ```
5. Consider adding Table 9 (dimension handling comparison) to Page 6 instead of Page 7

### M4. Related Work Lacks Critical Engagement and Historical Context

**Location**: §7 Related Work (Page 7)

**Issue**: Based on reading Latex/main.tex, Related Work has 46 citations but lacks critical analysis. It's structured as "here are compression methods [cite], here are attention optimizations [cite], here are frameworks [cite]" without engaging with:
1. Why did prior work miss the alignment problem?
2. What is the historical evolution of GPU alignment constraints?
3. How does this paper's contribution fit into the broader hardware-aware ML literature?

The paper cites FlashAttention but doesn't discuss its dimension handling design choices. It cites PaLU but doesn't explain why PaLU enforces 32-multiple alignment (likely they discovered this issue independently).

**Why it matters**: For a systems conference like EuroMLSys, reviewers expect to see engagement with the history of the problem. The current Related Work feels like a missed opportunity for depth.

**[NEEDS_LITERATURE_SEARCH: related_work]**
Suggested searches:
- "hardware-aware neural network compression history"
- "GPU Tensor Core alignment requirements evolution Volta Ampere Hopper"
- "FlashAttention dimension handling design decisions GitHub issues"
- "PaLU alignment constraints implementation rationale"

**Suggested Fix**:
1. Add historical paragraph: "GPU alignment constraints trace back to Volta's Tensor Cores (2017) requiring K%8. Ampere (2020) tightened this to K%16 for optimal throughput. Hopper (2023) introduced TMA with cache-line granularity. Our work is the first to systematically document how compression methods violate these implicit contracts."
2. Add critical analysis: "Why did prior compression methods overlook alignment? PaLU enforces 32-multiple alignment [cite], but this design choice is undocumented in their paper. GPTQ/AWQ preserve original dimensions, avoiding the issue. Our framework explains these design decisions retroactively."
3. Compare with prior hardware-aware work: "Prior work on hardware-aware pruning [citations] focused on sparsity patterns, not dimension alignment. The Roofline model [cite] assumes regular tile sizes. Our contribution extends hardware-aware optimization to the dimension choice problem."
4. Target 60+ citations (add 15-20 references)

---

## Minor Issues (Suggested)

### m1. Figure 3 (PaLU Distribution) Occupies Too Much Space for Information Density

**Location**: Page 3, Figure 3

**Issue**: From page_03.png, I observe Figure 3 shows a histogram of dimension distribution (10 unique values: 114-125) with a "THEORETICAL ANALYSIS" banner and "96.9% misaligned" text. This occupies 0.85\columnwidth but is a simple plot with low information density—equivalent content could fit in a smaller figure or table.

**Suggestion**: Reduce to 0.6\columnwidth, or convert to an inset bar chart in Figure 1. Alternative: Replace with a cumulative distribution function (CDF) showing "% of dimensions above alignment threshold" to add analytical value.

### m2. Figure 5 (Repair Tradeoff) Data Point Labels Overlap with Markers

**Location**: Page 6, Figure 5

**Issue**: From page_05.png, I observe the scatter plot has data labels ("d=107", "d=114", "d=117", "d=120", "d=121", "d=125") that overlap with or are very close to the data point markers. For d=107 and d=117, the label is directly on top of the marker, reducing readability.

**Suggestion**: Use leader lines (arrows) connecting labels to points, or adjust label positions using `xytext` offsets in matplotlib:
```python
ax.annotate('d=107', xy=(x, y), xytext=(x+0.5, y+0.5),
            arrowprops=dict(arrowstyle='->', lw=0.5))
```

### m3. Abstract Packs Too Many Numbers (Hard to Parse)

**Location**: Page 1, Abstract

**Issue**: The Abstract contains 10+ specific numbers in a short space: "88% vs aligned", "96.9% of SVD-optimal ranks", "58%, 50%, 40%", "22-28%", "3.7-7.2%", "3.5-5.9×", "86.9% average speedup", "45 SDPA configurations", "batch 1-8", "sequences 512-2048", "range 46-181%". This is overwhelming for readers scanning the abstract.

**Suggestion**: Reduce to 3-4 key numbers:
- "head_dim=107 increases SDPA latency by 88%"
- "96.9% of theoretical SVD ranks violate GPU alignment"
- "Dimension repair achieves 22-28% kernel speedup with 3.7-7.2% memory overhead"
- "Validated through contrasting experiments: -0.8% vs. 86.9%"

Move detailed ranges (batch 1-8, sequences 512-2048) to Evaluation section.

### m4. "Dimensional Collapse" Term Never Formally Defined

**Location**: §1 Introduction

**Issue**: The term "dimensional collapse" is introduced in §1 paragraph 3 as "a nonlinear performance degradation caused by misalignment between software-defined tensor shapes and hardware-fixed access patterns." But this is buried in prose, not highlighted as a definition. Later usage is inconsistent—sometimes referring to the phenomenon of irregular dimensions appearing, sometimes to the performance degradation.

**Suggestion**: Add a definition box:
```latex
\noindent\fbox{\parbox{0.96\columnwidth}{%
\textbf{Definition (Dimensional Collapse)}: The nonlinear GPU performance degradation that occurs when post-training compression produces tensor dimensions violating hardware alignment constraints (e.g., Tensor Core tile sizes, vectorized load requirements).
}}
```

### m5. Table 1 (Backend Latency) Column Headers Could Be More Descriptive

**Location**: Page 3, Table 1

**Issue**: Column headers "AUTO", "FLASH", "MEM_EFF", "MATH" are abbreviations without immediate context for readers unfamiliar with PyTorch SDPA backends.

**Suggestion**: First mention should expand: "AUTO (default)", "FLASH (FlashAttention)", "MEM_EFF (MEM_EFFICIENT)", "MATH (fallback)". Or add a table footnote explaining backend types.

### m6. Page 8-10 Blank Space Utilization

**Location**: Pages 8-10

**Issue**: From page_08.png and page_10.png, I observe that Page 8 has ~50% blank space (Conclusion ends early, references start mid-page with large gap), and Page 10 has ~60% blank space after references end. This indicates poor space distribution.

**Suggestion**:
1. Adjust LaTeX float placement parameters to better distribute content
2. Add Appendix material (detailed experimental configurations, reproducibility checklist)
3. Expand Related Work discussion to fill space
4. Move some content from Page 7 to better balance pages 6-10

---

## Questions for Authors

1. **E2E Validation Strategy**: You show kernel-level 86.9% speedup (Table 8) and negative E2E validation (RAP SVD -0.8%, Table 6). Why not implement a minimal directly-compressed model (vanilla SVD on Q/K/V without projection layers) to complete the positive E2E validation?

2. **PaLU Alignment Discovery**: You mention "production PaLU checkpoints enforce 32-multiple alignment" multiple times. Did the PaLU authors discover dimensional collapse independently? Is there documentation of their design rationale?

3. **Theoretical vs. Practical Scope**: The 96.9% figure comes from theoretical Fisher-information analysis. Are there *any* existing compression methods that produce misaligned dimensions in practice (besides your RAP SVD experiment)?

4. **H100 Generalization**: You mention H100 has m16n8k16 MMA tiles (same K%16 requirement as A100) in §8. Have you done *any* preliminary experiments on H100, even informal?

5. **FlashAttention Versioning**: You use FlashAttention 2.7.4 and acknowledge results are version-specific (§8). Have you checked if FlashAttention 3.x changes the dimension handling?

6. **Integration with PaLU**: You describe dimension repair as a "post-compression pass" in §8. Have you reached out to PaLU authors to discuss integration?

---

## Detailed Comments by Section

### Abstract
**Score: 7.5/10**

Strengths: Clearly states the problem, quantifies the impact (88% latency increase), provides contrasting validation (-0.8% vs +86.9%). The theoretical scope disclaimer is present.

Weaknesses: Too many numbers (see m3). The sentence structure is complex. Suggested rephrase for the theoretical scope: "We analyze unconstrained compression scenarios where 96.9% of theoretical ranks violate alignment, providing diagnostic guidance for method designers."

### Introduction
**Score: 7.0/10**

Strengths: Good motivating example. The Scope and Applicability paragraph directly addresses potential reviewer concerns upfront. Contributions are clearly listed with section references.

Weaknesses: "Dimensional collapse" term is not formally defined (see m4). The fourth contribution bullet is too long (3 lines)—split into two bullets.

### Background
**Score: 7.5/10**

Strengths: Notation paragraph is clear. FlashAttention constraints discussion (§2.2) is accurate. The correction "Contrary to common belief, it does *not* strictly require 8-aligned dimensions" is valuable.

Weaknesses: Tensor Core alignment (§2.1) is too brief—only 3 sentences. Low-Rank Compression (§2.3) is skeletal—only 2 sentences.

### Dimensional Collapse (§3)
**Score: 8.0/10**

Strengths: Excellent experimental methodology. The "Scope" subsection (§3.2) demonstrates transparency. Backend selection behavior (Table 1) is well-documented. The "Note on variance" in §3.1 shows measurement sophistication.

Weaknesses: Subsection titles are dry. "SDPA Latency vs. Head Dimension" could be "Performance Cliff: 88% Slowdown at head_dim=107."

### Root Cause Analysis (§4)
**Score: 7.5/10**

Strengths: The isolation methodology is rigorous. Testing H1-H4 independently and disconfirming H2 demonstrates scientific process. The Root Cause Summary box is excellent. Table 2 is concise and informative.

Weaknesses: Figure 4 is too small (see M2). The FlashAttention kernel dispatch footnote could be expanded.

### Shape-Aware Compression (§5)
**Score: 7.0/10**

Strengths: Shape Contract formalization is clear. Zero-padding accuracy preservation explanation is well-argued.

Weaknesses: This is the shortest section (1.5 columns) despite being the solution. Lacks algorithmic detail or pseudocode.

### Evaluation (§6)
**Score: 7.5/10**

Strengths: Contrasting validation structure (negative + positive) is excellent. Transparency about variance. Applicability Framework (Table 7) is practical.

Weaknesses: Missing positive E2E validation (see M1). Table 8 (Direct SDPA) shows huge variance (46-181% speedup range) needing more discussion.

### Conclusion (§8)
**Score: 7.0/10**

Strengths: Honest about limitations (H100, downstream tasks). Clear integration guidance. Acknowledges version-specific nature.

Weaknesses: Integration guidance introduces new technical content in Conclusion—should be in Evaluation or Discussion section.

### Related Work (§7)
**Score: 6.0/10**

Strengths: Comprehensive citation coverage (46 references). The dimension handling comparison table is useful.

Weaknesses: This is the paper's weakest section. Reads like a literature dump without critical engagement (see M4). No historical context. Missing recent citations.

---

## Visual Observations (必填！)

### Page-by-Page Observations

**Page 1:**
- 看到的内容: Title "When Smaller Is Slower: Dimensional Collapse in Compressed LLMs", 5 authors (Jihao Xin, Tian Lv, Qilong Pan, Kesen Wang, Marco Canini), Abstract section, Introduction §1
- 具体观察: Abstract is italicized, ~11 lines. I can read specific numbers: "head_dim=107", "88%", "96.9%", "58%, 50%, 40%", "ROI: 3.5-5.9×". Author affiliations show KAUST and HUMAIN AI. Introduction starts with "Large Language Models (LLMs) have achieved remarkable capabilities..."
- 问题/建议: Abstract is dense (175+ words), exceeding typical 150-word guideline. No visual issues with layout—good margins and spacing.

**Page 2:**
- 看到的内容: §1 Introduction (continued), §2 Background, §3 Dimensional Collapse, Figure 1 (Dimensional collapse overview)
- 具体观察: Figure 1 has two panels (a) and (b). Panel (a) shows "Unconstrained SVD compression produces irregular dimensions" with boxes and arrows. I can read text "96.9% of dimensions would be misaligned", "head_dim=107", "88% slower". Panel (b) shows "Dimension repair pads to hardware-preferred multiples". The figure occupies full columnwidth (~8.5cm).
- 问题/建议: Figure 1 caption is 58 words (very long for a schematic). Some internal text in panel (a) appears ~7pt font size.

**Page 3:**
- 看到的内容: §3 Dimensional Collapse (continued), Figure 2 (SDPA latency line plot), Figure 3 (PaLU distribution histogram), Table 1 (Backend latency), §4 Root Cause Analysis starts
- 具体观察:
  - Figure 2: X-axis "Head Dimension" 60-160, Y-axis "Latency (ms)" 0-2.5. Blue points for "8-aligned", orange for "Non-8-aligned". I can see specific data point labeled "2.19ms" at d=107 (peak). Error bars are present but very small.
  - Figure 3: Histogram with X-axis showing dimensions 114-125, Y-axis "Number of KV heads" 0-100. Red banner at top says "THEORETICAL ANALYSIS". Text "96.9% misaligned" visible.
  - Table 1: 5 rows (d=96, 104, 107, 112, 128), 4 columns (AUTO, FLASH, MEM_EFF, MATH). Row d=107 is bolded. I can read "2.14±0.06" for FLASH at d=107, and "N/A*" for MEM_EFF at d=107.
- 问题/建议:
  1. Figure 2: Orange label "2.19ms" overlaps with the line/marker
  2. Figure 3: Histogram bin labels are ~6-7pt, too small for print
  3. Page feels dense—2 figures + 1 table + text in single page

**Page 4:**
- 看到的内容: §4 Root Cause Analysis (continued), Figure 4 (Root cause breakdown), Table 2 (Hardware root causes), boxed "Root Cause Summary"
- 具体观察:
  - Figure 4: Horizontal bar chart with 4 bars. I can read labels "Tensor Core K%16" (red bar, longest, ~58%), "Vectorized Load (K%8)" (orange bar, ~50%), "Bandwidth Sectors" (gray, ~40%), "L2 Cache Sectors" (light gray, tiny, ~5.8%). Text labels "Confirmed" and "Not Confirmed" visible.
  - Table 2: 4 rows (H1-H4), columns: Hypothesis, Status, Impact, Root Cause. Status shows "✓ Confirmed" or "Not confirmed".
  - Root Cause Summary: Gray box with 3 numbered points in bold.
- 问题/建议:
  1. **Figure 4 太小**：估计约 0.5 columnwidth，bar labels 字体约 7-8pt，打印时难读
  2. Figure 4 颜色与 Figure 2 不一致（Fig 2 用 blue/orange，Fig 4 用 red/blue/orange/gray）
  3. Table 2 与 Figure 4 间距 <5mm，显得拥挤

**Page 5:**
- 看到的内容: §5 Shape-Aware Compression, §6 Evaluation, Figure 5 (Repair tradeoff scatter plot), Table 3 (SDPA repair performance)
- 具体观察:
  - Figure 5: Scatter plot, X-axis "Memory Overhead (%)" 0-8, Y-axis "Speedup (%)" 0-30. 6 data points labeled "d=107", "d=114", "d=117", "d=120", "d=121", "d=125". Two series: "MINIMAL" (orange circles), "OPTIMAL" (blue triangles). Point d=120 is highlighted with a circle.
  - Table 3: 6 rows × 6 columns, showing Original(ms), Minimal(ms), Optimal(ms), ΔMin, ΔOpt. I can read "2.06±0.06", "1.49±0.04", "+27.8%".
- 问题/建议:
  1. Figure 5: Labels d=107, d=117 overlap with markers
  2. Figure 5: Upper-right quadrant (speedup >15%, overhead >5%) is empty—figure could be smaller
  3. Table 3 caption is 80+ words (too long)

**Page 6:**
- 看到的内容: §6 Evaluation (continued), Table 6 (RAP SVD E2E), Table 7 (Applicability Framework), partial text from §6.3, §6.4, boxed Limitations
- 具体观察:
  - Table 6: 3 rows × 4 columns (Phase, Misaligned, Repaired, Δ). I can read "290.5", "292.9", "--0.8%", "1009", "1000", "--0.9%".
  - Table 7: 3 architecture types × 4 columns. Bold text "Yes +86.9%" and "No --0.8%".
  - Boxed Limitations: Gray box with "L1. Applicability Scope", "L2. Downstream Tasks", "L3. Hardware" (3 items).
  - **大片空白**：Right column has significant whitespace (~12-15cm vertical space blank) after boxed Limitations before §7 Related Work
- 问题/建议:
  1. **Page 6 空白是最严重问题**：右栏 ~40% 空白，浪费版面
  2. 建议填充：将 H100 讨论前移、扩展 Limitations、或添加表格
  3. Tables 6-7 间距合理，无重叠问题（与之前 review 不同）

**Page 7:**
- 看到的内容: §7 Related Work, Table 9 (Dimension handling comparison), §8 Conclusion starts
- 具体观察:
  - Table 9: 3-section table (FlashAttn-2/vLLM/TensorRT, GPTQ/AWQ/PaLU/RAP, This work), 3 columns (System, Supported head_dim, Misaligned handling). I can read "Slow path (+30-45%)", "Error/fallback", "Runtime padding", "Compile-time fix".
  - Related Work: Dense paragraphs with many citations. Sections: "LLM Compression", "Attention Optimization & GPU Kernels", "Inference Frameworks", "GPU Architecture & Tensor Cores", "Positioning".
  - §8 Conclusion starts at bottom of page.
- 问题/建议:
  1. Related Work 是列举式而非批判性讨论（见 M4）
  2. Table 9 位置打断阅读流程，建议移至 §2 Background
  3. Page 7 密度高但无拥挤问题

**Page 8:**
- 看到的内容: §8 Conclusion (continued), References section starts mid-page
- 具体观察:
  - Conclusion paragraphs: "H100 Generalization", "Software Version Note", "Integration with Compression Frameworks", "Why Projection-Based Methods Don't Benefit", "Reproducibility".
  - References: Two-column format, numbered [1]-[~10] visible on this page. I can read [1] "Joshua Ainslie, James Lee-Thorp, Michiel de Jong..." (long author list), [2] "Anthropic. 2024. Claude 3.5 Sonnet...".
  - **Page 8 下半部分 ~50% 空白**：Conclusion 文字结束后，References 开始，但 Page 8 下方有大量空白
- 问题/建议:
  1. **Page 8 空白问题严重**：~50% 空间未利用
  2. Conclusion 过长（1.5 columns），"Why Projection-Based..." 段落应移至 §6
  3. 建议调整 float placement 或添加内容

**Page 9:**
- 看到的内容: References (continued), entries [~11]-[~35]
- 具体观察: Standard two-column reference format. Entries include NeurIPS, ICML, OSDI, MLSys papers, and arXiv preprints. Font size ~8-9pt, readable.
- 问题/建议: No visual issues. References formatting is clean.

**Page 10:**
- 看到的内容: References (continued), entries [~36]-[46], followed by **large blank area**
- 具体观察: Last reference is [46] "Yilong Zhao, Chien-Yu Xu, Yuhui Guo... ATOM: Low-bit Quantization for Efficient and Accurate LLM Serving. In MLSys." Page ends with references. **Bottom ~60% of both columns is blank**.
- 问题/建议:
  1. **Page 10 空白最严重**：~60% 空白，完全浪费
  2. 建议：添加 Appendix (实验配置详情、reproducibility artifacts)
  3. 或将 §7 Related Work 扩展以填充空间

### Figure-by-Figure Assessment

| Figure | 位置 | 你观察到的具体内容 | 尺寸评估 | 布局评估 | 问题 |
|--------|------|-------------------|---------|---------|------|
| Fig 1 | Page 2 | Two-panel schematic (a)/(b), boxes labeled "96.9% misaligned", "88% slower", "Dimension Repair", full columnwidth | 合适 | 正常 | Caption 过长 (58 words); panel (a) 内部文字 ~7pt 偏小 |
| Fig 2 | Page 3 | Line plot, X 60-160, Y 0-2.5ms, blue/orange lines, labeled "8-aligned"/"Non-8-aligned", data point "2.19ms" at d=107 | 合适 | 正常 | Orange label "2.19ms" overlaps marker; Y-axis labels ~7pt |
| Fig 3 | Page 3 | Histogram, X 114-125, Y 0-120, "THEORETICAL ANALYSIS" banner, 10 bars | 合适 | 正常 | **信息密度低**：10 bars 占 0.85 columnwidth；Bin labels ~6-7pt too small |
| Fig 4 | Page 4 | Horizontal bar chart, 4 bars (TC/Vec/BW/L2), percentages 58%/50%/40%/5.8%, red/orange/gray colors | **过小** (~0.5 columnwidth) | 正常 | **最严重问题**：Bar labels ~7-8pt难读；应放大到 0.75+ columnwidth |
| Fig 5 | Page 5 | Scatter plot, X 0-8% (overhead), Y 0-30% (speedup), 6 points labeled d=107/114/117/120/121/125 | 合适 (0.75 columnwidth) | 正常 | Labels overlap markers (d=107, d=117); Upper-right empty |
| Table 1 | Page 3 | Backend latency, 5×4 table, values "1.17±0.03", "2.14±0.06", "N/A*" for MEM_EFF at d=107 | 合适 | 正常 | Column headers need expansion |
| Table 2 | Page 4 | Hardware hypotheses, 4×4 table, "✓ Confirmed"/"Not confirmed", percentages 58%/50%/40%/5.8% | 合适 | 正常 | Clear presentation, no major issues |
| Table 3 | Page 5 | Repair results, 6×6 table, latency values with ±std, speedup percentages | 合适 | 正常 | Caption too long (80+ words) |
| Table 6 | Page 6 | RAP SVD E2E, 3×4 table, negative results --0.8%/--0.9% | 合适 | 正常 | Clear negative result presentation |
| Table 7 | Page 6 | Applicability framework, 3×4 table, bold "Yes +86.9%"/"No --0.8%" | 合适 | 正常 | **Excellent key contribution table** |
| Table 9 | Page 7 | System comparison, 3-section × 3-column table | 合适 | 正常 | Position awkward (mid-Related Work); move to §2 |

### Layout Assessment (布局评估 - 必填！)

**整体页面利用率**：
- **是否有大片空白未利用？** **是**，严重问题：
  - Page 6 右栏：~40% 空白（~12-15cm 垂直空间）
  - Page 8 下半部分：~50% 空白
  - Page 10：~60% 空白（references 结束后）
- **图片尺寸与信息量是否匹配？**
  - 不匹配的图片：
    - Figure 3 (PaLU distribution)：信息密度低，10 bars 占 0.85 columnwidth
    - Figure 4 (root cause breakdown)：**过小**，critical figure 仅 ~0.5 columnwidth

**图文冲突检查**：
- **是否有图片侵入正文空间？** 否，margins 控制良好
- **是否有图片与 caption/其他元素重叠？** 否
- **双栏排版中是否有单栏图片过大？** Figure 3 偏大但未侵入另一栏

**尺寸问题图片列表**：
| 图片 | 问题类型 | 具体描述 | 建议修改 |
|------|---------|---------|---------|
| Figure 3 | 信息密度低 | 0.85 columnwidth 用于 10 bars 的简单 histogram | 缩小到 0.6 columnwidth 或改为 CDF plot |
| Figure 4 | 过小/关键信息难读 | ~0.5 columnwidth，bar labels ~7-8pt 打印时难读 | 放大到 0.75-0.85 columnwidth，字体 9-10pt |
| Figure 1 | 内部字体偏小 | Panel (a) 内部文字约 7pt | 增大到 8-9pt |

### Visual Issues Summary

**必须列出至少 5 个视觉问题**（已列出 12 个）：

1. **Page 6 右栏大片空白** (~40% page height)：最严重的布局问题，浪费宝贵空间。建议添加 H100 讨论、Limitations 扩展。

2. **Figure 4 (root cause) 过小**：核心诊断图约 0.5 columnwidth，bar labels 7-8pt 打印时难辨认。应放大到 0.75+ columnwidth，字体 9-10pt。

3. **Figure 3 信息密度低**：简单 10-bar histogram 占 0.85 columnwidth。建议缩小到 0.6 columnwidth。

4. **Figure 5 标签重叠**：d=107, d=117 的标签直接覆盖数据点。建议使用 leader lines。

5. **Page 8/10 空白浪费**：Page 8 下半部分 ~50% 空白，Page 10 ~60% 空白。建议添加 Appendix 或扩展 Related Work。

6. **Abstract 数字过载**：175+ words 包含 10+ specific numbers，overwhelming。建议精简到 3-4 key numbers。

7. **Figure 1 内部文字偏小**：Panel (a) 标注字体约 7pt。建议增大到 8-9pt。

8. **Figure 2 标签重叠**：Orange label "2.19ms" overlaps with line marker。建议添加 white background box。

9. **Figure 3 bin labels 太小**：Histogram X-axis labels (114-125) 约 6-7pt。建议增大到 8-9pt。

10. **Figure 4 颜色不一致**：Uses red/blue/orange/gray，与 Figure 2 的 blue/orange 方案不一致。

11. **Table 3 caption 冗长**：80+ words，重复 §3.1 的 variance 说明。建议精简到 <50 words。

12. **Page 3 过于密集**：2 figures + 1 table + dense text，视觉上缺少呼吸空间。

---

## Improvement Checklist for Writer Agent

### High Priority (Must Fix)
- [ ] **M2 - Figure 4 放大**: Page 4 - 从 ~0.5 columnwidth 放大到 0.75-0.85 columnwidth，bar labels 字体 9-10pt
- [ ] **M3 - Page 6 空白填充**: 添加内容填充右栏 ~12-15cm 空白（H100 讨论前移/Limitations 扩展/系统对比表）
- [ ] **M4 - Related Work 批判性改写**: §7 - 添加历史脉络、批判性分析、"why prior work missed this"，从列举式改为分析式
- [ ] **M4 - Related Work 扩展引用**: 从 46 → 60+ citations，添加 recent surveys (2024)、vLLM/TensorRT-LLM 系统对比、GPU architecture evolution papers
- [ ] **M1 - E2E 正面验证补充或说明**: §6 - 要么实现 vanilla SVD E2E 验证，要么明确说明 "kernel validation + framework is our scope, E2E is future work"
- [ ] **Page 8/10 空白利用**: 调整 float placement 或添加 Appendix 内容填充 50-60% 空白区域

### Medium Priority (Recommended)
- [ ] **m1 - Figure 3 缩小**: Page 3 - 从 0.85 columnwidth 缩小到 0.6 columnwidth
- [ ] **m2 - Figure 5 标签重叠修复**: Page 5 - 使用 leader lines 或 xytext offset
- [ ] **m3 - Abstract 精简数字**: Page 1 - 从 10+ numbers 精简到 3-4 key numbers
- [ ] **m4 - "Dimensional Collapse" 正式定义**: §1 - 添加 definition box
- [ ] **m5 - Table 1 列标题说明**: Page 3 - 扩展 "AUTO (default)"/"FLASH (FlashAttention)" 等
- [ ] **m6 - Table 3 caption 精简**: Page 5 - 从 80+ words 缩短到 <50 words
- [ ] **Figure 2 标签重叠修复**: 添加 white background box 或 adjust position
- [ ] **Figure 4 颜色统一**: 改为与 Figure 2 一致的 blue/orange 方案
- [ ] **Figure 1/3 bin labels 字体放大**: 从 6-7pt 增大到 8-9pt

### Low Priority (Optional)
- [ ] Figure 1 caption 缩短到 <40 words
- [ ] Table 2 添加 checkmarks (✓/✗) 代替文字
- [ ] Figure 5 legend 移至 plot 外部
- [ ] Conclusion "Why Projection-Based..." 段落移至 §6
- [ ] References arXiv entries 补充 year

---

## Depth Assessment

### Related Work Breadth
**Score: 5/10**
- Citation count: 46 (target: 60+)
- Domain coverage: 4 areas (Compression, Attention, Frameworks, GPU)
- Gap: Missing recent surveys (2024), vLLM code-level, H100 papers

### Historical Context
**Score: 4/10**
- Temporal span: ~5 years (2019-2024)
- Evolution discussion: Minimal
- Gap: No Volta→Ampere→Hopper evolution narrative

### Critical Thinking
**Score: 7/10**
- Strengths: Anticipates "why production systems don't have this problem", dual validation
- Gaps: Doesn't discuss "will FlashAttention fix this internally?"

### Terminology Precision
**Score: 6/10**
- Self-coined term: "dimensional collapse" (not established)
- Justification: Minimal (1 sentence)
- Gap: Should use established terms or rigorously justify

### Literature Quality
**Score: 7/10**
- Top venue %: ~70% (good)
- Recent work %: ~20% (low, should be 30-40%)
- Preprint %: ~10% (acceptable)

### Depth Bottleneck Identification

**Bottleneck = "Literature Integration"**

**Priority**: HIGH - Single most impactful improvement to reach 8/10

---

## Reviewer Confidence

**Confidence Score:** 4/5

**Expertise Areas:**
- GPU architecture and Tensor Core optimization
- LLM inference systems and kernel performance
- Experimental methodology for systems research

**Limitations:**
- Cannot verify FlashAttention 2.7.4 exact behavior without source inspection
- Limited familiarity with EuroMLSys specific acceptance bar
- Cannot independently reproduce A100 experiments

---

*Reviewer: Paper Reviewer Agent*
*Date: 2026-01-28*
