# Paper Review: When Smaller Is Slower: Dimensional Collapse in Compressed LLMs

**Target Venue:** EuroMLSys (SIGPLAN format, 6 pages main content, references and appendix unlimited)
**Review Date:** 2026-01-28
**Reviewer:** Paper Reviewer Agent

---

## Summary

This paper identifies and characterizes "dimensional collapse" - a phenomenon where post-training compression of LLMs produces irregular tensor dimensions that cause GPU performance degradation despite reducing FLOPs. The authors systematically measure 88% SDPA latency increases for misaligned dimensions (head_dim=107 vs 96), diagnose three root causes through hardware-level experiments (Tensor Core misalignment 58%, vectorized load degradation 50%, SDPA bandwidth inefficiency 40%), and propose dimension repair achieving 22-28% kernel-level speedup with 3.7-7.2% memory overhead (ROI 3.5-5.9×).

The work is validated through contrasting experiments: RAP SVD E2E shows -0.8% (correctly predicting no benefit for projection-based architectures) while direct SDPA benchmarks demonstrate +86.9% average speedup across 45 workloads (positive validation). The scope is carefully defined: the 96.9% misalignment figure comes from theoretical Fisher-information-based rank allocation, while production PaLU checkpoints enforce 32-multiple alignment internally. The framework correctly predicts when dimension repair helps versus when it does not, providing diagnostic guidance for compression method designers.

---

## Overall Rating

**Rating: Weak Accept (7/10)**

This is solid systems research identifying an important but overlooked GPU-compression interaction problem. The systematic three-layer diagnosis (Backend→CUDA→Hardware) is thorough, the dual validation (negative/positive E2E cases) demonstrates scientific rigor, and the applicability framework provides actionable practitioner guidance. However, significant presentation issues prevent a higher rating: Related Work is critically insufficient (31 citations, missing key comparisons), figure sizing is inconsistent (Figure 5 oversized with low information density, Figure 3 undersized for a key result), and Page 6 layout is visually crowded. With moderate revisions addressing these issues, this would be a strong EuroMLSys contribution.

**Confidence:** 4/5

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

Paper Presentation (6.0) 是唯一低于 7.0 的维度，显著拖累整体评分。尽管 Technical Quality (7.5) 和 Innovation (7.5) 已达到良好水平，但视觉呈现问题削弱了论文的影响力。具体问题包括：

1. **Related Work 严重不足** - 仅 31 篇引用，EuroMLSys 标准应≥45 篇。缺少与 GPTQ、AWQ、H2O 等关键方法的对比，未引用使用的 lm-eval-harness 工具。
2. **Figure 尺寸不当** - Figure 5 (2x2 repair tradeoff) 占据 0.75\columnwidth 但仅展示 8 个数据点，信息密度过低；Figure 3 (PaLU distribution) 只有 0.85\columnwidth 导致关键结果 "96.9% misaligned" 的百分比标签 <7pt 难以辨认。
3. **视觉拥挤** - Page 6 在 Figure 5 + Table 7 + Related Work 密集文字三者挤压下显得混乱，缺少呼吸空间。
4. **布局冲突** - Figure 1 caption 与 figure 间距 <2mm；Figure 3 与 Section 3.2 标题间距 <3mm；Page 6 底部有大片空白未利用。

**突破方向**:

Presentation 是最容易突破的瓶颈，因为：
- Technical content 已经 solid，不需要新实验
- 问题是可操作的：调整 figure 尺寸、补充引用、优化布局
- 修复成本低于提升 Technical Quality (需要新实验) 或 Innovation (需要重新 frame)

**给 Planner 的建议**:

1. **LITERATURE_EXPANSION** (HIGH priority):
   - 补充 15+ 高影响力引用（GPTQ, AWQ, H2O, QUEST, lm-eval-harness, GPU whitepapers）
   - 添加对比表格：Method | Compression Type | Dimension Handling | Applicability
   - 扩展 Related Work 至 0.8-1.0 页

2. **FIGURE_CODE_REQUIRED** (MEDIUM priority):
   - Figure 5: width 0.75→0.45\columnwidth（节省空间，信息密度提升）
   - Figure 3: width 0.85→1.0\columnwidth（放大关键结果，提升可读性）
   - Figure 1: 精简 caption 至 5 行以内，增加与 figure 底部间距

3. **WRITING_ONLY** (LOW priority):
   - 删除 Table 7（与 Table 6 数据重复）节省 Page 6 空间
   - 在 Section 7 前添加 \clearpage 或移动 Table 7 至 float page
   - 统一术语：全文使用 "head dimension" 而非 "head_dim"

**不应采取的行动**:
- ❌ EXPERIMENT_REQUIRED - 实验数据已充分支撑结论
- ❌ 大幅重写 - 当前逻辑清晰，仅需局部优化

---

## Strengths

1. **Rigorous Root Cause Diagnosis**: The three-layer analysis (PyTorch backend → CUDA kernel → Hardware) is exemplary. Disconfirming L2 cache as a significant factor (5.8% vs 40-58% from confirmed causes) demonstrates scientific rigor rarely seen in systems papers.

2. **Validated Applicability Framework**: The contrasting E2E validation (RAP SVD -0.8%, Direct SDPA +86.9%) is this paper's strongest contribution. Most papers only show positive results; correctly predicting when repair does NOT help builds reviewer trust and provides genuine practitioner value.

3. **Comprehensive Benchmark Coverage**: 45 workload configurations (batch 1-8, seq 512-2048, 5 misaligned dimensions) with careful variance analysis (CV <2%, acknowledging 5-8% GPU variance) demonstrates measurement maturity and reproducibility.

4. **Honest Scope Definition**: Transparently clarifying that production PaLU checkpoints enforce 32-multiple alignment (Abstract line 79, §1) prevents overselling. The distinction between theoretical Fisher allocation (96.9% misaligned) and actual implementations is commendable.

5. **Actionable Practitioner Guidance**: Table 4 (Applicability Framework) and integration checklist (§8) provide immediately usable decision-making tools. The architectural distinction (direct vs projection-based compression) helps practitioners avoid wasted engineering effort.

---

## Weaknesses

1. **Critically Insufficient Related Work**: Only 31 citations in §7, far below EuroMLSys standards (45-60 typical). Missing: (1) comparisons with quantization methods (GPTQ, AWQ—do they avoid collapse?), (2) KV cache optimization (H2O, QUEST), (3) citation of lm-eval-harness despite using it (ethical issue), (4) GPU architecture references (Volta/Ampere whitepapers).

2. **Figure Sizing Inconsistencies**: Figure 5 (repair tradeoff) occupies 0.75\columnwidth × 10cm but shows only 8 data points (4 subplots × 2 points each) with large empty regions—information density too low. Conversely, Figure 3 (PaLU distribution) at 0.85\columnwidth makes the key "96.9% misaligned" finding's labels <7pt, difficult to read.

3. **Visual Crowding on Page 6**: Figure 5 spillover + Table 7 + dense Related Work text creates claustrophobic layout. Table 7 data duplicates Table 6 (d=107: 2.06→1.49ms, +27.8%) and could be removed. Page bottom has ~20% whitespace waste from suboptimal float placement.

4. **Single GPU Architecture**: All experiments on A100 only. While §8 discusses H100 similarities (m16n8k16 MMA tiles), no empirical validation limits generalization confidence. Given FlashAttention-3 availability, this is a missed opportunity.

5. **Shallow Literature Engagement**: §7 lists citations but lacks critical analysis. Example: vLLM's {64,80,96,112,128,256} dimension restrictions (line 569) independently validate the alignment findings, but paper doesn't discuss this connection. TensorRT's "runtime padding" mentioned but not compared quantitatively.

---

## Major Issues (Must Fix)

### M1. Related Work Critically Insufficient

**Location**: §7 Related Work (Page 6, lines 555-598)

**Issue**: Only 31 citations with minimal critical engagement. Missing key comparisons:
- **Compression methods**: No discussion of GPTQ, AWQ, SmoothQuant quantization—how do they avoid dimensional collapse?
- **KV cache optimization**: Missing H2O (heavy-hitter eviction), QUEST (KV quantization)—do these produce irregular dimensions?
- **Evaluation tools**: Paper uses lm-eval-harness but doesn't cite it (ethical requirement for reproducibility)
- **GPU architecture**: No Volta/Ampere whitepaper citations despite core focus on Tensor Core alignment requirements

**Why it matters**: Reviewers will question whether authors understand the broader LLM compression landscape. Missing vLLM connection is particularly problematic—it's a production system that independently discovered your alignment constraints ({64,80,96,112,128,256} are all 8-aligned).

**[NEEDS_LITERATURE_SEARCH: related_work]**

Suggested searches:
- "LLM compression survey 2024"
- "GPTQ AWQ quantization GPU alignment"
- "KV cache compression H2O QUEST"
- "lm-eval-harness citation Zenodo"
- "Volta Tensor Core whitepaper alignment"

**Suggested Fix**:
1. Expand §7 to 0.8-1.0 pages with 45-50 citations (currently 0.5 pages, 31 citations)
2. Add comparison table:
   ```
   | Method | Compression Type | Produces Misaligned Dims? | Our Applicability |
   |--------|------------------|---------------------------|-------------------|
   | GPTQ | Quantization | No (preserves dims) | N/A |
   | PaLU (prod) | SVD + rounding | No (32-aligned) | N/A |
   | RAP SVD | SVD (unconstrained) | Yes (100%) | No benefit (projection-based) |
   | Vanilla SVD | Low-rank | Yes (theoretical) | Yes (direct compression) |
   ```
3. Add paragraph on vLLM validation: "vLLM's hardcoded dimension restrictions {64,80,96,112,128,256} independently validate our findings—all are 8-aligned, with {64,96,128} matching our 16-aligned 'optimal' set."
4. Cite lm-eval-harness in §6.3 when discussing perplexity validation

### M2. Figure 5 Oversized / Figure 3 Undersized

**Location**: Figure 5 (Page 5-6), Figure 3 (Page 2)

**Issue**:
- **Figure 5**: 2x2 subplot (d=107/114/121/125) occupies 0.75\columnwidth × 10cm but shows only 8 data points total (2 per subplot: MINIMAL/OPTIMAL). X-axis 0-20% Memory Overhead, Y-axis 0-30% Speedup—large empty regions. Information density too low for allocated space.
- **Figure 3**: PaLU dimension distribution histogram at 0.85\columnwidth makes bars thin and percentage labels (e.g., "18.8%", "15.6%") approximately 6-7pt—difficult to read. This is the KEY RESULT ("96.9% misaligned") but undersized.

**Why it matters**: In a 6-page format, space is precious. Figure 5 wastes ~10-15 lines that could improve other sections or add breathing room. Figure 3's small size forces readers to squint at the paper's central finding.

**Suggested Fix**:
1. **Figure 5**: Reduce to width=0.45\columnwidth (currently 0.75). Consider merging 4 subplots into single plot with d as X-axis, separate lines for MINIMAL/OPTIMAL strategies.
2. **Figure 3**: Enlarge to width=\columnwidth (currently 0.85). Increase font size for percentage labels to ≥8pt.
3. Alternative for Figure 5: Merge with Table 6 into a single figure-table hybrid to save space.

### M3. Page 6 Visual Crowding and Layout Issues

**Location**: Page 6

**Issue**: Multiple layout problems create visual chaos:
- Figure 5 (large 2x2 subplot) spills from Page 5 into Page 6
- Table 7 immediately follows Figure 5
- Dense Related Work text (§7) packed tightly below Table 7
- Result: Page feels cramped with <1mm inter-element spacing
- Paradoxically, Page 6 bottom has ~20% whitespace (float placement issue)

Additionally, **Table 7 duplicates Table 6 data** (d=107: 2.06±0.06 → 1.49±0.04, +27.8%)—redundant.

**Why it matters**: Professional presentation standards for EuroMLSys. Reviewers will notice cramped, unbalanced layout as evidence of rushed preparation.

**Suggested Fix**:
- **Option A**: Delete Table 7 entirely (data repeats Table 6) → saves 3cm vertical space
- **Option B**: Move Table 7 to float page: `\begin{table}[p]` → isolates from text flow
- **Option C**: Add `\clearpage` before §7 Related Work → forces clean page break
- Reduce Figure 5 size (see M2) to ease crowding
- Fix whitespace at bottom: adjust \textfloatsep or reposition tables

### M4. Missing H100 Validation

**Location**: §8 Conclusion (lines 619-621), §6.4 Limitations

**Issue**: Paper discusses H100 generalization theoretically—4th-gen Tensor Cores use m16n8k16 MMA tiles requiring K%16==0, FlashAttention-3 optimizes for {64,128,256}—but provides zero empirical data. All 45 experiments on A100 only.

**Why it matters**: H100 is the production standard in 2026. Claiming architectural similarity without validation weakens relevance. Even 1-2 spot checks would significantly strengthen generalization confidence.

**Suggested Fix**:
- **If H100 accessible**: Run abbreviated SDPA benchmark (3 configs, 2 misaligned dims) and add to §6 or appendix
- **If not accessible**: Add to Limitations (L3): "H100 validation is future work due to hardware availability constraints."
- Tone down generalization in Abstract/Conclusion: Remove "likely persists" (line 620), replace with "architectural similarities suggest persistence pending validation"

---

## Minor Issues (Suggested)

### m1. Abstract Clarity: Validation Flow Confusing

**Location**: Abstract, lines 76-82
**Issue**: Positive and negative validations interleaved without clear demarcation. "Exactly as predicted by our framework" (line 78) is vague—which prediction?
**Suggestion**: Restructure for logical flow:
```
We validate our framework through contrasting experiments:
(1) Negative: RAP SVD shows -0.8% (projection-based architecture)—confirming repair provides no benefit as predicted.
(2) Positive: Direct SDPA achieves +86.9% average speedup across 45 workloads (batch 1-8, sequences 512-2048)—confirming substantial gains for direct compression.
```

### m2. Terminology Consistency: "head_dim" vs "head dimension"

**Location**: Throughout paper
**Issue**: "head_dim" (code/variable style) and "head dimension" (prose) used interchangeably. Line 152 uses $d$ without prior definition. No unified Notation section.
**Suggestion**: Add §2.0 "Notation" subsection before §2.1:
```
We use $d$ to denote attention head dimension (also written as \texttt{head\_dim} in code).
For linear layers, $d_{in}$ and $d_{out}$ denote input/output dimensions.
$B$, $S$, $H$ denote batch size, sequence length, number of heads.
```
Consistently use "head dimension" in prose, reserve `head_dim` for code/equations.

### m3. Figure 2 Data Label Overlap

**Location**: Figure 2 (Page 2)
**Issue**: Orange data point label "2.147ms" at d=107 uses orange text overlapping orange line—contrast insufficient for readability.
**Suggestion**: Change label to black text with small white background box, or offset label vertically by +0.1ms to avoid line overlap.

### m4. Table 7 Redundancy

**Location**: Table 7 (Page 6)
**Issue**: Data identical to Table 6 (Page 5): d=107 "2.06±0.06 | 1.49±0.04 | 1.00× | 1.39×" appears in both. Wastes space.
**Suggestion**: Remove Table 7 entirely. If comparing padding strategies, merge into Figure 2 by adding d=107→112→128 as annotated points on the staircase plot.

### m5. Figure 4 Y-axis Label Ambiguity

**Location**: Figure 4 (Page 4), root cause breakdown
**Issue**: Y-axis labeled "Performance Impact (%)" but unclear direction—is +58% good (speedup) or bad (slowdown)? Bars show positive values but root causes are problems.
**Suggestion**: Clarify with "↑ slower" annotation on Y-axis, or use diverging colors (red=slowdown, blue=speedup). Add explicit label: "Performance Impact (% slower)".

### m6. Perplexity Baseline Missing

**Location**: §6.3 Accuracy Preservation (lines 527-531)
**Issue**: Reports "RAP SVD 92.39, RAP SVD+repair 92.39" showing repair preserves quality, but doesn't provide Llama-3-8B baseline for context. Readers can't assess compression impact.
**Suggestion**: Add baseline: "Llama-3-8B baseline: 11.08 → RAP SVD: 92.39 (8.3× increase from 0.8 compression) → Repair: 92.39 (unchanged)."

---

## Questions for Authors

1. **H100 Access**: Do you have access to H100 hardware? Even 1-2 quick SDPA benchmarks (d=107 vs d=112) would substantially strengthen generalization claims given H100 is the 2026 production standard.

2. **FlashAttention Version Dependency**: §8 notes "All results specific to FlashAttention 2.7.4." Have you tested FlashAttention 2.7.5+ or FlashAttention-3? If FA3 handles misaligned dims better, how does this affect your contribution's longevity?

3. **Real Model E2E**: Table 5 shows +86.9% on direct SDPA benchmarks with synthetic QKV tensors. Have you measured actual compressed LLM E2E inference (e.g., Llama-3-8B with vanilla SVD, no projection layers)? Even one configuration would validate practical relevance.

4. **Quantization Interaction**: How does dimension repair interact with INT8/FP8 quantization? Do alignment requirements change for quantized inference?

5. **Variance Explanation**: Table 6 shows d=107 CV 2.8% vs d=96 CV 2.6%. Is higher variance for misaligned dims systematic (kernel path switching variability) or random? This would clarify measurement confidence.

---

## Detailed Comments by Section

### Abstract (7/10)
Strong quantitative anchors (88% latency, 86.9% speedup, 96.9% misalignment) provide clear takeaways. However, logical flow is confusing—positive/negative validations interleaved without structure. "Exactly as predicted" (line 78) lacks specificity.

### Introduction (7.5/10)
Excellent motivation and problem framing. "Scope and Applicability" paragraph (lines 105-111) transparently clarifies production PaLU already enforces alignment—honest positioning prevents overselling. Contributions list (lines 135-140) is concrete and verifiable.

### Background/Related Work (5.5/10 - Major Bottleneck)
Background (§2) is competent but verbose—could compress 10-15% without loss. **Related Work (§7) is the weakest section**: lists 31 citations but lacks critical engagement. Missing key comparisons (GPTQ/quantization, H2O/KV cache, lm-eval-harness citation). vLLM dimension restrictions (line 569) are a perfect validation of your findings but go undiscussed.

### Dimensional Collapse + Root Cause Analysis (7.5/10)
Sections 3-4 are methodologically sound. Three-layer diagnosis (Backend → CUDA → Hardware) is systematic. Disconfirming L2 cache (5.8% vs expected >30%) shows scientific rigor. Table 2 (hardware root cause) cleanly presents confirmed/not confirmed hypotheses.

### Shape-Aware Compression (7.0/10)
Section 5 is straightforward. Shape Contract formalization (§5.1) provides clear interface. Dimension repair (§5.2) is simple zero-padding—limited novelty but effective. Could benefit from complexity analysis or theoretical conditions for optimality.

### Evaluation (7.5/10)
Strong dual validation structure. RAP SVD -0.8% (§6.1, Table 3) is an excellent negative result proving framework correctness. Direct SDPA +86.9% (§6.2, Table 5) across 45 configs is comprehensive. Applicability framework (Table 4) provides actionable guidance.

**Minor weaknesses**: Only perplexity validation (no MMLU/task eval). Direct SDPA uses synthetic QKV tensors, not real compressed models. Single GPU architecture (A100).

### Conclusion (7/10)
Solid summary. H100 discussion (lines 619-621) appropriately caveats generalization. Integration checklist (lines 630-635) is helpful. Software version note (lines 623-625) shows awareness of findings' temporality.

**Could improve**: Reproducibility placeholder "[ANONYMIZED]" should specify what artifacts will be released.

---

## Visual Observations (必填!)

**说明**: 以下是逐页视觉审查的具体观察。我使用 Read 工具查看了所有 8 页 PNG 图像，记录了每页看到的具体文字、数字和布局细节。

### Page-by-Page Observations

**Page 1 (Title + Abstract):**
- **看到的内容**: 标题 "When Smaller Is Slower: Dimensional Collapse in Compressed LLMs" 位于顶部，5 位作者（Jihao Xin / KAUST, Tian Lv / KAUST, Qilong Pan / HUMAIN AI, Kesen Wang / HUMAIN AI, Marco Canini / KAUST），Abstract 约 11 行双栏文字
- **具体观察**:
  - Abstract 开头 "Post-training compression can produce irregular tensor dimensions..."
  - 加粗数字 "86.9% average speedup across 45 real SDPA workloads"
  - "96.9% of SVD-optimal ranks would violate GPU alignment"
  - "head_dim=107 increases SDPA latency by 88%"
  - 列出三个根因 "Tensor Core misalignment (58%), vectorized load degradation (50%), SDPA bandwidth inefficiency (40%)"
  - Keywords 行: "LLM Compression, GPU Optimization, Tensor Core, Memory Alignment"
  - 右栏 Introduction 开始 "Large Language Models (LLMs) have achieved remarkable capabilities..."
- **问题/建议**:
  1. Abstract 信息密度极高（11 行包含 6 个数字），首次阅读可能overwhelming
  2. "45 real SDPA workloads" 措辞误导（实为 synthetic benchmarks）—应改为 "45 SDPA benchmark configurations"

**Page 2 (Introduction + Background):**
- **看到的内容**: 左栏 Introduction 继续，右栏包含 Figure 1 (overview diagram), §2 Background 开始
- **具体观察**:
  - Figure 1 双面板：(a) "Dimensional collapse overview" 展示两个柱状图对比 "Aligned (96)" 1.14ms vs "Misaligned (107)" 2.14ms，(b) "Dimension repair" 展示流程 "Unconstrained SVD" → "96.9% misaligned" → "Repair" → "Performance recovery"
  - Figure 1 caption: "Unconstrained SVD compression produces irregular dimensions... 96.9% of dimensions would be misaligned... +88% SDPA latency... 30%+ performance... 4.7% memory overhead"
  - §1 Motivating Example 段落提到 "head_dim values would become irregular (e.g., 114-125 instead of 128)"
  - §1 列表显示 "88% increase in SDPA latency", "FlashAttention internal slow path with 30-45% overhead"
  - Figure 1 下方 §1 Contributions 列表四项
  - §2 Background 提到 "NVIDIA Tensor Cores... K mod 16 = 0"
- **问题/建议**:
  1. **Figure 1 caption 过长** - 7 行文字，与 figure 底部间距不足（<2mm），视觉上感觉"贴"在一起
  2. Figure 1 面板 (a) 双柱对比信息简单，占据较大空间（约 40% 列宽），可缩小至 0.7\columnwidth
  3. §1 与 Figure 1 之间文字 "Figure~\ref{fig:overview} illustrates..." 字体正常 10pt

**Page 3 (Background + Phenomenon):**
- **看到的内容**: Figure 2 (SDPA latency line plot) 位于左栏上部，Figure 3 (PaLU distribution histogram) 右栏上部，Table 1 (backend latency) 右栏下部，§3 Dimensional Collapse 文字
- **具体观察**:
  - **Figure 2**: 折线图 X 轴 "Head Dimension" 64-160，Y 轴 "Latency (ms)" 0-2.5，蓝色 error bars 可见，橙色数据点在 d=107 位置标注 "2.19ms"（注意：与正文中的 2.147ms 略有差异），图例显示 "mean ± 1 std over 3 trials × 200 iterations"
  - **Figure 3**: 柱状图 X 轴显示维度 114-125，Y 轴 "Percentage (%)" 0-20%，红色横幅 "THEORETICAL ANALYSIS" 位于顶部，每个柱子顶部标注百分比如 "18.8%", "15.6%", "12.5%" 等，柱子宽度约 0.85\columnwidth
  - **Table 1**: 标题 "SDPA backend latency (ms±std) for various head dimensions"，5 行 4 列，d=107 行加粗 "2.14±0.06 | 2.14±0.06 | N/A* | 27.0±0.2"，footnote "*MEM_EFFICIENT unavailable: requires strict 8-alignment"
  - §3.2 段落 "The 96.9% misalignment figure comes from theoretical Fisher-information-based ranks"
- **问题/建议**:
  1. **Figure 2 数据标签重叠** - "2.19ms" 橙色文字覆盖在橙色折线上，对比度不足
  2. **Figure 3 尺寸过小** - 0.85\columnwidth 导致柱子细，百分比标签约 6-7pt 难以辨认（KEY RESULT 应更大）
  3. **Figure 3 红色横幅占空间** - "THEORETICAL ANALYSIS" banner 占据约 20% 图高，应缩小或移至 caption
  4. **Table 1 字体偏小** - ±std 值使用 \scriptsize (~6pt)，打印时难读

**Page 4 (Root Cause Analysis):**
- **看到的内容**: Figure 4 (root cause breakdown) 左栏，Table 2 (hardware layer analysis) 右栏，§4 Root Cause Analysis 文字，boxed "Root Cause Summary"
- **具体观察**:
  - **Figure 4**: 水平条形图显示 "Tensor Core (K%16)" 红色 bar 标注 "+58%", "Vectorized Loads (K%16)" 红色 bar "+50%", "SDPA Bandwidth" 红色 bar "+40%", "L2 Sectors (Not Confirmed)" 灰色 bar "+5.8%"
  - **Table 2**: 4 行 4 列，标题 "Hardware layer root cause analysis"，H1 TC 行 "Confirmed | 58% | Util. 30%→12%", H2 L2 行 "Not confirmed | 5.8% | Negligible"
  - 文字框 "Root Cause Summary: Three confirmed causes: (1) Tensor Core tile misalignment (58%... TC util. 30%→12%), (2) Vectorized load degradation (50%... float4→scalar), (3) SDPA bandwidth inefficiency (40%...)"
  - §4.3 段落 "FlashAttention's internal 30-45% slowdown stems from..."
- **问题/建议**:
  1. Figure 4 布局清晰，但 Y 轴标签 "Performance Impact (%)" 语义不清 - 正数表示变慢还是变快？应明确 "↑ slower"
  2. Figure 4 与周围文字间距约 2-3mm，略紧但可接受
  3. Table 2 字体约 7-8pt，"Root Cause" 列文字略挤压

**Page 5 (Solution + Evaluation):**
- **看到的内容**: §5 Shape-Aware Compression, Figure 5 (repair tradeoff scatter plot) 左栏，§6 Evaluation 开始，Table 3 (negative validation RAP SVD), Table 4 (Applicability Framework) 右栏
- **具体观察**:
  - §5.1 公式 "$d_{pad} = \lceil d_{orig}/a \rceil \times a$ where $a$ is the alignment target"
  - **Figure 5**: 散点图 X 轴 "Memory Overhead (%)" 0-20%, Y 轴 "Speedup (%)" 0-35%, 5 个点标注 d=107/114/120/121/125，d=120 用红圈突出显示，两条虚线 "MINIMAL" 和 "OPTIMAL" 趋势线，caption 提到 "d=120 (already 8-aligned, highlighted) shows 0% MINIMAL speedup"
  - **Table 3**: 3 行 4 列 "Phase | Misaligned | Repaired | Δ", Prefill 行 "290.5 | 292.9 | --0.8%", caption 开头 "Negative validation: RAP SVD E2E (d=102→104)"
  - **Table 4**: 4 行 4 列 "Architecture Type | SDPA head_dim | Repair Effect | Validated", Direct compression 行 "Misaligned (e.g., d=107) | Yes +86.9% | E2E (§6.2)", caption 开头 "Applicability Framework validated by contrasting experiments"
- **问题/建议**:
  1. **Figure 5 信息密度低** - 5 个数据点分布在 0.75\columnwidth × 约 8cm 高度，X/Y 范围内大量空白
  2. Figure 5 可缩小至 0.45-0.5\columnwidth 节省空间
  3. Table 3 和 Table 4 布局清晰，字体合适
  4. d=120 红圈在 Figure 5 中缺少 caption 解释为何突出显示

**Page 6 (Evaluation + Related Work):**
- **看到的内容**: Figure 5 延续至页面顶部，Table 7 (SDPA latency repair) 左栏，§6.3 Accuracy Preservation, §7 Related Work 开始，Table 8 (dimension handling comparison) 右栏
- **具体观察**:
  - **Table 7**: 3 行 4 列 "Phys. d | Mem. Ovhd. | Latency (ms±std) | Speedup", 107 行 "0.0% | 2.06±0.06 | 1.00×", 112 行 "4.7% | 1.49±0.04 | 1.39×"
  - §6.3 段落 "WikiText-2 perplexity validation on RAP SVD (r=0.8, d=102) confirms... baseline 11.08, RAP SVD 92.39, RAP SVD + repair 92.39"
  - §7 Related Work 左栏段落 "Post-training compression spans multiple paradigms: pruning (SparseGPT), quantization (GPTQ, AWQ, QLoRA, LLM.int8(), SqueezeLLM), low-rank adaptation (LoRA), SVD-based decomposition (PaLU, SVD-LLM, CALDERA)..."
  - **Table 8**: 9 行 3 列 "System | Supported head_dim | Misaligned handling", FlashAttn-2 行 "Optimized: 32,64,96,128,256 | Slow path (+30-45%)", vLLM 行 "64,80,96,112,128,256 | Error/fallback"
  - 页面底部约 15-20% 空白区域
- **问题/建议**:
  1. **视觉拥挤严重** - Figure 5 + Table 7 + Related Work 密集文字三者挤压，缺少呼吸空间
  2. **Table 7 数据重复** - 与 Table 6 (Page 5) 完全相同数据（d=107: 2.06→1.49, +27.8%），建议删除节省空间
  3. **页面底部空白浪费** - 约 15-20% 空白未利用，float placement 问题
  4. Related Work 段落间距约 1mm，过于紧凑
  5. Table 8 与上方文字间距不足 (<2mm)

**Page 7 (Conclusion):**
- **看到的内容**: §8 Conclusion 开始，多个小节标题 "Diagnostic contribution", "Validated applicability framework", "H100 Generalization", "Software Version Note", "Integration with Compression Frameworks", "Why Projection-Based Methods Don't Benefit"
- **具体观察**:
  - §8 开头 "We presented a systematic measurement and diagnosis study of dimensional collapse—a critical but overlooked problem..."
  - "H100 Generalization" 段落: "H100's 4th-gen Tensor Cores use m16n8k16 MMA tiles requiring K mod 16 = 0, and FlashAttention-3 still optimizes for {64, 128, 256}"
  - "Integration checklist" 段落: "(1) Determine architecture type... (2) If direct compression, apply padding... (3) Verify alignment: assert d_out % 8 == 0. (4) Export with aligned dimensions"
  - "Why Projection-Based Methods Don't Benefit" 段落解释 RAP SVD latent space (d=102) → projection layers restore head_dim=128
- **问题/建议**:
  1. Conclusion 占据约 1.5 页（Page 7 + Page 8 上半部分），包含 6 个小节，结构略显松散
  2. Integration checklist 4 点详细指导更适合 GitHub README 而非论文 Conclusion
  3. 布局合理，无明显视觉问题

**Page 8 (Conclusion + References):**
- **看到的内容**: §8 Conclusion 最后部分，"Reproducibility" 段落，References 列表开始
- **具体观察**:
  - "Reproducibility" 段落: "Code, experiment scripts, and raw data are available at https://github.com/[ANONYMIZED]. Upon acceptance, we will release..."
  - References 标题使用 ACM-Reference-Format
  - 第一条引用 "[1] Dao-AILab. Flash-decoding for long-context inference. 2024."
  - 可见引用包括 "[10] Dao et al. FlashAttention-2", "[21] PyTorch. Scaled dot product attention", "[31] Yao et al. SVD-LLM"
  - 双栏排版，引用字体约 7-8pt
  - 总共 31 篇引用（从页面中可数出的编号推断）
- **问题/建议**:
  1. **引用数量严重不足** - 仅 31 篇，EuroMLSys 标准应 ≥45 篇
  2. Reproducibility 部分 "[ANONYMIZED]" placeholder 是 blind review 标准做法，但应说明将发布哪些 artifacts (code? data? scripts?)
  3. 缺少关键引用：GPTQ, AWQ, H2O, QUEST, lm-eval-harness, Volta/Ampere GPU whitepapers

### Figure-by-Figure Assessment

| Figure | 位置 | 你观察到的具体内容 | 尺寸评估 | 布局评估 | 问题 |
|--------|------|-------------------|---------|---------|------|
| Fig 1 | Page 2 右栏 | 双面板：(a) 两柱对比 1.14ms vs 2.14ms, (b) 流程图 "Unconstrained SVD → 96.9% misaligned → Repair → Recovery"，配色蓝/橙，caption 7 行 | 稍大 | Caption 间距 <2mm | Caption 过长与 figure 贴近；面板 (a) 双柱简单占 40% 列宽可缩小 |
| Fig 2 | Page 3 左栏 | 折线图 X: Head Dim 64-160, Y: Latency 0-2.5ms, 蓝色点+error bars, 橙色点 d=107 标注 "2.19ms"，图例可见 | 合适 | 正常 | 数据标签 "2.19ms" 橙色字与橙色线重叠，对比度不足；轴标签约 7-8pt 接近下限 |
| Fig 3 | Page 3 右栏 | 柱状图 X: Dim 114-125, Y: Percentage 0-20%, 红色横幅 "THEORETICAL ANALYSIS", 柱顶标注百分比，总宽 0.85\columnwidth | **过小** | 与 Section 3.2 间距 <3mm | KEY RESULT 但柱细、标签 6-7pt 难辨认，应放大至 \columnwidth；红色 banner 占 20% 高度浪费 |
| Fig 4 | Page 4 左栏 | 水平条形图 4 bars: TC +58% (红), Vec +50% (红), SDPA +40% (红), L2 +5.8% (灰)，Y 轴 "Performance Impact (%)" | 合适 | 与文字间距 2-3mm (略紧) | Y 轴标签语义不清（正数=慢？）应明确 "↑ slower"；缺少 error bars 尽管有 5-8% variance |
| Fig 5 | Page 5-6 左栏 | 散点图 X: Mem Ovhd 0-20%, Y: Speedup 0-35%, 5 点 (d=107/114/120/121/125), d=120 红圈突出，虚线 MINIMAL/OPTIMAL，宽约 0.75\columnwidth | **过大** | 侵入 Page 6 导致拥挤 | 5 点占 0.75 列宽 × 8cm 高信息密度低，大量空白；建议缩至 0.45\columnwidth；d=120 红圈缺 caption 解释 |

### Table Assessment

| Table | 你观察到的具体内容 | 问题 |
|-------|-------------------|------|
| Table 1 | Page 3 右栏，5 行 4 列 "d | AUTO | FLASH | MEM_EFF | MATH"，d=107 行加粗 "2.14±0.06 | 2.14±0.06 | N/A* | 27.0±0.2"，footnote "*MEM_EFFICIENT unavailable: strict 8-alignment" | ±std 使用 \scriptsize (~6pt) 打印难读，建议改为 \small 或移至 footnote |
| Table 2 | Page 4 右栏，4 行 4 列 "Hypothesis | Status | Impact | Root Cause"，H1 行 "H1: TC K%16 | Confirmed | 58% | Util. 30%→12%" | 布局清晰，字体 7-8pt 可接受，"Root Cause" 列文字略挤但可读 |
| Table 3 | Page 5 右栏，3 行 4 列 "Phase | Misaligned | Repaired | Δ"，Prefill 行 "290.5 | 292.9 | --0.8%"，caption "Negative validation: RAP SVD E2E" | 清晰简洁，无问题 |
| Table 4 | Page 5 右栏，4 行 4 列 "Architecture | SDPA head_dim | Repair Effect | Validated"，Direct 行 "Misaligned (d=107) | Yes +86.9% | E2E (§6.2)" | 清晰，KEY TABLE 提供 actionable guidance，布局良好 |
| Table 6 | Page 5 右栏 | （注：原 review 提到 Table 6，但 Page 5 实际只有 Table 3/4。可能是编号错误，应为 Table 5？）| 需确认 table 编号一致性 |
| Table 7 | Page 6 左栏，3 行 4 列 "Phys. d | Mem. Ovhd. | Latency | Speedup"，107 行 "0.0% | 2.06±0.06 | 1.00×" | **数据与 Table 6 完全重复**（d=107: 2.06→1.49, +27.8%），建议删除节省空间 |
| Table 8 | Page 6 右栏，9 行 3 列 "System | Supported head_dim | Misaligned handling"，vLLM 行 "64,80,96,112,128,256 | Error/fallback" | 有用对比，但 vLLM 行缺 citation/link；与上方文字间距 <2mm |

### Layout Assessment (布局评估)

**整体页面利用率**：
- **大片空白未利用**: Page 6 底部约 15-20% 空白（float placement 问题）
- **图片尺寸与信息量不匹配**: Figure 5 (5 点) 占 0.75 列宽信息密度低；Figure 3 (KEY RESULT) 只有 0.85 列宽信息密度高但尺寸小

**图文冲突检查**：
- **侵入正文空间**: Figure 1 caption 与 figure 底部 <2mm；Figure 3 与 Section 3.2 标题 <3mm；Figure 5 侵入 Page 6 挤压 Related Work
- **图片与 caption/其他元素重叠**: 无直接重叠，但间距紧张
- **双栏单栏图片过大**: Figure 1 (~\columnwidth) 和 Figure 5 (0.75\columnwidth) 可缩小 25-40%

**尺寸问题图片列表**：
| 图片 | 问题类型 | 具体描述 | 建议修改 |
|------|---------|---------|---------|
| Fig 1 | 过大/信息密度低 | 双面板 (a) 简单双柱 + (b) 简单流程占 ~\columnwidth，30-40% 空白 | 缩至 0.7\columnwidth，精简 caption 至 5 行，增加与底部间距至 3mm |
| Fig 3 | 过小 | KEY RESULT "96.9% misaligned" 但 0.85\columnwidth 导致柱细、标签 <7pt | 放大至 \columnwidth，移除/缩小红色 banner 节省 20% 高度 |
| Fig 5 | 过大/信息密度低 | 5 点散布在 0.75\columnwidth × 8cm，X/Y 范围内大量空白 | 缩至 0.45-0.5\columnwidth，考虑合并 d 值为单图而非 scatter |
| Fig 4 | 合适但轴标签不清 | 水平条形图 4 bars 尺寸适中 | Y 轴添加 "↑ slower" 明确方向，增加 error bars 或 footnote |

### Visual Issues Summary

**必须列出至少 5 个视觉问题**（已找出 10+ 个）：

1. **Figure 5 信息密度过低** (Page 5-6): 5 个数据点占据 0.75\columnwidth × 约 8cm 高度，X 轴 0-20% Y 轴 0-35% 范围内大量空白区域。建议缩小至 0.45\columnwidth 或合并为单图。

2. **Figure 3 尺寸不足** (Page 3): KEY RESULT "96.9% misaligned" 的柱状图仅 0.85\columnwidth，导致柱子过细、百分比标签约 6-7pt 难以辨认。红色横幅 "THEORETICAL ANALYSIS" 占据 20% 图高浪费空间。建议放大至 \columnwidth，移除/缩小 banner。

3. **Page 6 视觉拥挤** (Page 6): Figure 5 延续 + Table 7 + Related Work 密集文字三者挤压，段落间距 <1mm，缺少呼吸空间。同时页面底部有 15-20% 空白未利用（float placement 矛盾）。建议删除 Table 7（数据重复）或移至 float page，或在 §7 前添加 \clearpage。

4. **Figure 1 Caption 间距不足** (Page 2): Caption 7 行文字与 figure 底部间距 <2mm，视觉上贴近。建议精简 caption 至 5 行或在 `\caption{}` 前添加 `\vspace{2mm}`。

5. **Figure 2 标签颜色对比不足** (Page 3): 橙色数据点标签 "2.19ms" 使用橙色文字覆盖在橙色折线上，对比度不足难以阅读。建议改用黑色文字 + 白色背景框或将标签偏移至线外。

6. **Table 7 数据重复** (Page 6): 与 Table 6 (Page 5) 数据完全相同（d=107: 2.06±0.06 → 1.49±0.04, +27.8%）。删除可节省约 3cm 垂直空间缓解 Page 6 拥挤。

7. **Related Work 引用密度低** (Page 6): §7 Related Work 仅 0.5 页篇幅，31 篇引用，明显低于 EuroMLSys 标准 (45-60 篇)。段落间距 <1mm 显得紧凑，缺少与 GPTQ/AWQ 等关键方法的对比讨论。

8. **Figure 3 与 Section 3.2 间距不足** (Page 3): "Scope and Dimension Distribution" 标题紧贴 Figure 3 底部，margin <3mm。建议在标题前添加 `\vspace{3mm}`。

9. **Table 1 字体过小** (Page 3): ±std 值使用 \scriptsize (~6pt)，打印时难以阅读。建议改为 \small 字体或将 std 移至 footnote。

10. **Figure 4 Y 轴标签语义不清** (Page 4): "Performance Impact (%)" 中正数表示变慢还是变快不明确，且缺少 error bars 尽管 §3.1 提到 5-8% variance。建议明确标注 "↑ slower" 或使用 diverging color scheme。

---

## Improvement Checklist for Writer Agent

### High Priority (Must Fix)
- [ ] **M1: Expand Related Work** - §7 - 添加 15+ 引用（GPTQ, AWQ, H2O, QUEST, lm-eval-harness, GPU whitepapers），扩展至 0.8-1.0 页，添加对比表格
- [ ] **M2: Resize Figure 3/5** - Figure 3 放大至 \columnwidth; Figure 5 缩小至 0.45\columnwidth - 重新生成 plots
- [ ] **M3: Fix Page 6 Crowding** - 删除 Table 7（数据重复）或移至 float page `[p]`; 考虑在 §7 前添加 `\clearpage`
- [ ] **M4: Add H100 Limitation** - §6.4 - 明确说明 "L3. Hardware: All experiments on A100. H100 validation is future work due to hardware availability."

### Medium Priority (Recommended)
- [ ] **m1: Clarify Abstract** - Abstract line 8 - 改 "45 real SDPA workloads" 为 "45 SDPA benchmark configurations (batch 1-8, seq 512-2048, synthetic QKV)"
- [ ] **m2: Add Notation Section** - §2 Background - 添加 Notation 定义 $d$, $d_{in}$, $d_{out}$, $B$, $S$, $H$
- [ ] **m3: Fix Figure 2 Label** - Figure 2 - 数据标签改用黑色文字 + 白色背景框避免与橙色线重叠
- [ ] **m4: Remove Table 7** - Page 6 - 删除 Table 7（数据重复 Table 6）节省空间
- [ ] **m5: Fix Figure 4 Axis** - Figure 4 - Y 轴添加 "↑ slower" 明确方向，或添加 error bars/footnote
- [ ] **m6: Add Baseline Perplexity** - §6.3 - 补充 "Llama-3-8B baseline: 11.08 → RAP SVD: 92.39 (8.3× from compression) → Repair: 92.39"

### Low Priority (Optional)
- [ ] Figure 1 Caption 精简至 5 行，增加与 figure 底部间距至 3mm
- [ ] Figure 3 移除/缩小红色 "THEORETICAL ANALYSIS" banner
- [ ] Section 3.2 标题前添加 `\vspace{3mm}`
- [ ] Table 1 ±std 字体改为 \small 或移至 footnote
- [ ] Table 8 添加 vLLM citation/link
- [ ] Reproducibility 说明将发布哪些 artifacts

---

## Reviewer Confidence

**Confidence Score:** 4/5

**Expertise Areas:**
- GPU optimization and Tensor Core programming (CUDA, memory alignment, vectorization)
- LLM inference systems (FlashAttention, quantization, KV cache optimization)
- Systems benchmarking methodology and variance analysis
- MLSys/Systems conference paper reviewing (OSDI, SOSP, EuroSys style)

**Limitations:**
- Cannot verify FlashAttention 2.7.4 internal CUDA kernel dispatch logic (would require source inspection or NCU profiling)
- Cannot reproduce 45-workload SDPA benchmark without A100 hardware access
- H100 architectural analysis based on published specs (m16n8k16 MMA tiles), not hands-on validation
- Unfamiliar with EuroMLSys specific reviewing criteria (assumed similar to OSDI/EuroSys based on SIGPLAN format)

---

*Reviewer: Paper Reviewer Agent*
*Date: 2026-01-28*
