# Paper Review: When Smaller Is Slower: Dimensional Collapse in Compressed LLMs

**Target Venue:** EuroMLSys (SIGPLAN format, 6 pages main content)
**Review Date:** 2026-01-27
**Reviewer:** Paper Reviewer Agent

---

## Summary

This paper investigates "dimensional collapse"---a phenomenon where post-training LLM compression produces irregular tensor dimensions that cause GPU performance degradation despite reducing FLOPs. The authors focus on low-rank SVD compression methods and demonstrate that misaligned dimensions (e.g., head_dim=107) can increase SDPA latency by 88% compared to aligned dimensions (e.g., head_dim=96).

The paper identifies three primary root causes through controlled experiments: (1) Tensor Core tile misalignment causing 58% slowdown when K%16≠0, (2) vectorized load degradation with 50% throughput loss when dimensions aren't 8-aligned, and (3) SDPA bandwidth inefficiency with 40% degradation. Notably, L2 cache sector waste (5.8%) is shown to be negligible, which contradicts initial intuition.

The proposed solution, "dimension repair," is a lightweight post-compression pass that pads dimensions to aligned values. The MINIMAL strategy (mod-8 alignment) achieves 25-28% kernel-level speedup with only 3.72% memory overhead, yielding a 6.9× ROI. The paper also contextualizes PaLU compression benefits, showing 11.5× decode throughput improvement (orthogonal to the alignment contribution).

An important clarification: the paper is transparent that the 96.9% misalignment figure comes from *theoretical* Fisher-information analysis (unconstrained SVD), not actual PaLU checkpoints. All available PaLU models use internal 32-multiple quantization and are 100% aligned. The contribution applies to compression methods without such constraints.

---

## Overall Rating

**Rating: Weak Accept (7.25/10)**

This is a solid systems paper that identifies a real and important problem with careful experimental methodology. The root cause analysis is thorough and the proposed solution is practical. However, there are notable limitations: (1) E2E validation shows compression benefits rather than repair benefits, and (2) the scope is limited to "theoretical" misalignment scenarios since production checkpoints are aligned.

**Confidence:** 4/5 (High confidence in the technical analysis; the experiments and methodology are sound)

---

## Detailed Scores

| Dimension | Weight | Score | Weighted |
|-----------|--------|-------|----------|
| Technical Quality | 40% | 7.5/10 | 3.00 |
| Paper Presentation | 30% | 7.0/10 | 2.10 |
| Innovation | 20% | 7.0/10 | 1.40 |
| Writing Quality | 10% | 7.5/10 | 0.75 |
| **Total** | 100% | - | **7.25/10** |

---

## Strengths

1. **Thorough Root Cause Analysis**: The paper systematically tests four hypotheses and provides quantitative evidence for each. The confirmation that L2 cache waste is NOT a primary cause (5.8%, negligible) is valuable negative result.

2. **Honest Scope Clarification**: The paper explicitly states that production PaLU checkpoints use internal alignment and that the repair targets "compression methods without such constraints." This transparency strengthens credibility.

3. **Clear ROI Metric**: Defining ROI = speedup/memory_overhead (6.9× for MINIMAL strategy) provides an actionable way to evaluate repair strategies.

4. **Comprehensive Hardware Analysis**: Testing across multiple layers (PyTorch backend selection, CUDA kernel paths, hardware constraints) provides a complete picture of where slowdowns originate.

5. **Reproducible Experimental Setup**: The paper specifies exact versions (PyTorch 2.9.1, CUDA 12.8, FlashAttention 2.7.4, Driver 560.35.03) and measurement methodology (warmup=50, measure=200, trials=3).

---

## Weaknesses

1. **Limited Practical Applicability**: The main weakness is that the "dimensional collapse" problem does not occur in currently available PaLU checkpoints. The 96.9% misalignment figure comes from theoretical Fisher-information-based rank allocation, not actual compressed models.

2. **No End-to-End Validation of Repair**: While kernel-level speedups (25-30%) are demonstrated, the paper lacks E2E validation on models that actually have misaligned dimensions. The E2E experiments (Table 6, Figure 6) use aligned PaLU models and thus only demonstrate compression benefits, not repair benefits.

3. **Missing Perplexity/Accuracy Evaluation**: The paper claims "bit-exact output preservation" but only validates with unit tests (30/30). Comprehensive perplexity evaluation (WikiText-2) is acknowledged as future work but is critical for a publication-ready paper.

4. **FlashAttention Version Specificity**: Results are specific to FlashAttention 2.7.4 and may not generalize to future versions. The paper acknowledges this but doesn't discuss how to make findings more robust.

5. **Incomplete Related Work**: The paper lacks comparison with other dimension handling strategies in production inference frameworks (vLLM, TensorRT-LLM padding strategies).

---

## Major Issues (Must Fix)

### M1. Misleading Figure 2 Caption and Data Source

**Location**: §3.2, Figure 2 (actually Figure 3 in the paper - the PaLU dimension distribution), and throughout

**Issue**: Figure caption says "Theoretical dimension distribution from Fisher-information-based rank allocation" but the paper uses this 96.9% misalignment figure as motivation throughout. Readers may incorrectly assume real compressed models have this distribution.

**Why it matters**: The gap between theoretical analysis and actual PaLU checkpoint behavior is the paper's main limitation. Burying the clarification in a note after "96.9%" is misleading.

**Suggested Fix**:
- Rename Figure to make "theoretical/unconstrained" more prominent in the title
- Add a paragraph in Section 3 explicitly comparing theoretical vs. production PaLU behavior
- Move the scope clarification from "Motivating Example" to a dedicated "Scope and Applicability" subsection

### M2. E2E Validation Gap

**Location**: §6.4, Table 6, Figure 6

**Issue**: The E2E evaluation shows PaLU compression benefits (11.5× decode speedup) but not dimension repair benefits. This is because the tested PaLU model has aligned dimensions.

**Why it matters**: Without E2E validation of the repair, the paper's claimed contribution (dimension repair achieving 25-30% speedup) is only demonstrated at kernel level with synthetic misaligned inputs.

**Suggested Fix**:
- Option A: Create a synthetic model with misaligned dimensions and validate E2E repair
- Option B: Use RAP SVD (which doesn't enforce alignment per findings.yaml) to generate a truly misaligned model
- Option C: If neither is feasible, significantly downscope the contribution to "kernel-level insights and guidelines" rather than "dimension repair solution"

### M3. Missing Accuracy Validation Beyond Unit Tests

**Location**: §6.5 Accuracy Preservation

**Issue**: "Unit tests confirm identical outputs (30/30 passed)" is insufficient for a systems paper targeting EuroMLSys. The claim of bit-exact preservation needs validation on meaningful workloads.

**Why it matters**: Reviewers will question whether padding affects model behavior in edge cases not covered by unit tests.

**Suggested Fix**:
- Run perplexity evaluation on WikiText-2 comparing original vs. repaired models
- Include task accuracy on at least one downstream benchmark
- If time-constrained, at minimum validate on a longer sequence generation task

---

## Minor Issues (Suggested)

### m1. Figure 1 Overview Diagram Clarity

**Location**: Page 1, Figure 1

**Issue**: The overview diagram is dense and the flow is not immediately clear. The "88% latency increase" and "30% recovery" labels are small (~6-7pt).

**Suggestion**: Increase font size for key numbers to 8pt minimum; add arrows showing the progression from compression → collapse → repair.

### m2. Table 1 Standard Deviation Formatting

**Location**: Page 2, Table 1

**Issue**: The scriptsize subscript format for std (e.g., {\scriptsize$\pm$.03}) is inconsistent and hard to read.

**Suggestion**: Use a consistent format: either parentheses (1.17±0.03) or separate std column.

### m3. Redundant Content in §6.4

**Location**: §6.4 Orthogonal Study: PaLU Compression Benefits

**Issue**: This section is marked as "orthogonal" to the main contribution, yet takes significant space (~0.4 columns). The 11.5× decode speedup is interesting but doesn't support the paper's thesis about dimension repair.

**Suggestion**: Condense to 2-3 sentences acknowledging compression benefits without detailed analysis, or move to appendix.

### m4. Missing Definition of "Staircase Effect"

**Location**: Figure 3 caption (the SDPA latency figure)

**Issue**: The term "staircase effect" is mentioned in the caption but not defined in the text.

**Suggestion**: Add a brief explanation: "We observe a staircase effect where latency jumps discontinuously at non-8-aligned boundaries."

### m5. Inconsistent Terminology

**Location**: Throughout

**Issue**: Minor inconsistency between "dimensional collapse" and "dimension collapse" in capitalization and wording.

**Suggestion**: Standardize to lowercase "dimensional collapse" in body text.

### m6. Figure 4 Y-axis Scale

**Location**: Page 3, Figure 4 (Root Cause Breakdown)

**Issue**: The Y-axis label "Performance Impact (%)" is ambiguous. Is higher better or worse?

**Suggestion**: Clarify as "Performance Degradation (%)" or add directional annotation.

### m7. Figure 5 Label Overlap - CRITICAL

**Location**: Page 4-5, Figure 5 (Repair Tradeoff scatter plot)

**Issue**: Data point labels (d=107, d=114, d=117, d=120, d=121, d=125) overlap severely in upper-left region where d=107, d=121, d=125 labels collide.

**Suggestion**: Use leader lines/arrows to offset labels from points, or stagger label positions.

---

## Questions for Authors

1. Have you attempted to run RAP SVD (which doesn't enforce alignment) to generate a truly misaligned model for E2E validation?

2. What is the overhead of your dimension repair pass in terms of compilation time? Is it suitable for JIT scenarios?

3. How do your findings apply to other attention implementations (e.g., xformers, Triton-based kernels)?

4. Have you measured the impact of dimension repair on memory bandwidth utilization, not just latency?

5. What is the expected behavior on H100/Hopper architecture where Tensor Core constraints may differ?

---

## Detailed Comments by Section

### Abstract
The abstract is well-written and concise. The key numbers (88% latency increase, 25-30% speedup, 3.7% overhead) are clearly stated. However, the abstract could clarify that the 96.9% misalignment figure is theoretical, not from production models.

### Introduction
The introduction effectively motivates the problem with a concrete example. The "Scope and Applicability" paragraph is crucial and well-placed. The contribution list is clear and verifiable. Minor improvement: the 11.5× decode speedup mentioned in the introduction is from compression, not repair---this could confuse readers.

### Background/Related Work
The background is adequate but could benefit from more detail on FlashAttention's internal kernel dispatch logic, since this is central to the root cause analysis. The related work section (§7) covers key papers but misses some recent work on dimension-aware inference frameworks.

**[NEEDS_LITERATURE_SEARCH: related_work]**
建议搜索：
- "vLLM dimension alignment handling"
- "TensorRT-LLM implicit padding"
- "LLM inference dimension constraints 2024-2025"

### Method/Approach (§4 Root Cause Analysis, §5 Shape-Aware Compression)
The root cause analysis is the paper's strongest section. The four-hypothesis framework is systematic, and the quantitative breakdown (TC 58%, Vec 50%, BW 40%, L2 5.8%) is valuable. The Shape Contract formalization is clean and actionable.

### Evaluation
The kernel-level evaluation (§6.1-6.3) is solid. The dimension repair validation (C4) provides good evidence for the claimed speedups. However, the E2E section (§6.4) is problematic as discussed in Major Issues. The limitations section (§6.6) is honest, which is appreciated.

### Conclusion
The conclusion accurately summarizes findings and acknowledges future work. The 6.9× ROI claim is well-supported by the data.

---

## Visual Observations (必填！)

**说明**: 此章节证明我真正查看了论文图像。

### Page-by-Page Observations

**Page 1:**
- 看到的内容: Title "When Smaller Is Slower: Dimensional Collapse in Compressed LLMs", 5 authors (Jihao Xin, Tian Lv, Qilong Pan, Kesen Wang, Marco Canini), Abstract, Keywords, Introduction section, Figure 1
- 具体观察:
  - Title uses large bold font (~14-16pt), clearly readable
  - Author affiliations: KAUST (Thuwal, Saudi Arabia) and HUMAIN AI (Riyadh, Saudi Arabia)
  - Abstract contains key numbers: "88%" latency increase, "25-30%" kernel speedup, "3.7%" memory overhead
  - Figure 1 appears in right column, shows workflow diagram with "Dimension Collapse" → "Dimension Repair"
  - Figure 1 caption mentions "88% latency increase" and "30% performance" recovery with "4.7% memory overhead"
  - Figure 1 uses blue/green for aligned (good) and orange/red for misaligned (bad)
  - Annotations visible: "TC Tile", "Vec Load", "BW" with percentages (58%, 50%, 40%)
- 问题/建议:
  1. Figure 1 text annotations are small (~6-7pt), may be hard to read when printed
  2. Need to increase font size for key numbers to 8pt minimum
  3. Consider adding clearer directional arrows showing the flow

**Page 2:**
- 看到的内容: End of Introduction, Section 2 (Background), Section 3 (Dimensional Collapse), Figures 2 and 3
- 具体观察:
  - Figure 2 (PaLU dimension distribution) shows bar chart with X-axis showing dimensions 112-126
  - Green bar only at d=120 (8-aligned), all others red (misaligned)
  - Y-axis shows Count up to ~150
  - Caption states "96.9% of the 512 theoretical KV head dimensions are misaligned"
  - Important Note in caption: "Production PaLU checkpoints use internal quantization that enforces alignment"
  - Figure 3 shows SDPA latency line plot, X-axis: 80-160 (Head Dimension), Y-axis: 1.0-2.5 ms
  - Clear "staircase" pattern visible in Figure 3: flat regions at aligned values, jumps at non-aligned
  - Light blue shaded confidence band (±1 std) visible but quite light
  - Data point at d=107 shows ~2.15ms vs ~1.14ms at d=96
- 问题/建议:
  1. Figure 2 X-axis labels (112, 114, 116...) are quite small, approximately 7-8pt
  2. Figure 3 confidence band is very light (alpha ~0.2)—may not reproduce well in print
  3. d=120 bar in Figure 2 should be more visually distinct (it's the only 8-aligned value)

**Page 3:**
- 看到的内容: Table 1 (Backend latency), Section 3.3 Backend Selection Behavior, Section 4 (Root Cause Analysis), Figure 4, Table 2
- 具体观察:
  - Table 1 shows 5 rows (d=96,104,107,112,128), 4 columns (AUTO, FLASH, MEM_EFF, MATH)
  - d=107 row: AUTO=2.14±.06, FLASH=2.14±.06, MEM_EFF="N/A*", MATH=27.0±.2
  - AUTO and FLASH columns show identical values—FlashAttention is the AUTO choice
  - MATH backend is 12.6× slower than FLASH
  - Footnote explains MEM_EFFICIENT requires strict 8-alignment
  - Table 2 shows "Hardware layer root cause analysis" with Hypothesis, Status, Impact, Root Cause columns
  - H1 (TC K%16): Confirmed, 58%, H2 (L2 sector): Not confirmed, 5.8%
  - H3 (SDPA BW): Confirmed, 40%, H4 (Vec loads): Confirmed, 50%
  - Figure 4 shows horizontal bar chart of "Root cause breakdown"
  - Bars labeled: "Tensor Core" 58%, "Vec. Load" 50%, "SDPA BW" 40%, "L2 Cache" 5.8%
- 问题/建议:
  1. Table 1: AUTO and FLASH columns redundant—consider merging or explaining
  2. Figure 4: L2 Cache bar (5.8%) is barely visible due to scale difference
  3. Figure 4: "Not Confirmed" (H2) should be visually distinct—use grey or hatching

**Page 4:**
- 看到的内容: Section 5 (Shape-Aware Compression), Section 6 (Evaluation), Tables 3, 4, 5, Figure 5
- 具体观察:
  - Table 3 shows "Padding rescue results for SDPA": Phys.d, Mem. Ovhd., Latency, Speedup columns
  - Table 3 data: d=107 (base) 0% overhead 2.064ms; d=112 4.7% overhead 1.490ms 1.39×; d=128 19.6% overhead 1.506ms 1.37×
  - Shows d=112 is optimal: better speedup than d=128 with much lower overhead
  - Table 4 shows "Memory overhead analysis": MINIMAL (mod 8) 3.72%, OPTIMAL (mod 16) 7.20%
  - Figure 5 shows scatter plot of "Per-dimension speedup vs. memory overhead"
  - X-axis: Memory Overhead (%) 0-10%, Y-axis: Speedup (%) 0-30%
  - Blue circles (MINIMAL) and orange squares (OPTIMAL) differentiate strategies
  - Points labeled: d=107 (~4.5%, 28%), d=114 (~4%, 24%), d=117 (~4%, 24%), d=120 (0%, 0%), d=121 (~4%, 27%), d=125 (~2%, 27%)
  - d=120 at origin (0%, 0%) validates alignment hypothesis (already 8-aligned, no improvement)
  - Table 5 shows SDPA latency before/after repair with Original, Minimal, Optimal columns
- 问题/建议:
  1. **CRITICAL - Figure 5**: Labels overlap severely (d=107, d=121, d=125 cluster in upper-left)
  2. d=120 point at origin is key validation but small—needs annotation/callout
  3. Speedup format inconsistent: Table 3 uses "1.39×" while Table 5 uses "+27.8%"

**Page 5:**
- 看到的内容: Table 6, Figure 6, Section 6.5 (Accuracy Preservation), Section 6.6 (Limitations), Section 7 (Related Work)
- 具体观察:
  - Table 6 shows E2E comparison: Baseline vs PaLU for Prefill, Decode, Memory
  - Key number: Decode 119 tok/s → 1371 tok/s (+11.5×)
  - Prefill: 9870 → 9672 (-2.0%), Memory: 19003 → 18896 (-0.6%)
  - Figure 6 shows dual-panel bar chart comparing "Baseline" vs "PaLU" for Prefill and Decode
  - Left panel (Prefill) shows almost identical heights with "-2.0%" label
  - Right panel (Decode) has dramatic height difference with "+11.5x" label
  - Figure 6 uses consistent color scheme (blue for Baseline, orange for PaLU)
  - Caption clarifies: "Compression benefit, not repair benefit"
  - Section 6.6 Limitations lists 4 items: accuracy scope, E2E integration, E2E impact estimate, validation gap
- 问题/建议:
  1. Figure 6: "11.5×" label is small (~7pt), should be larger given its importance
  2. Figure 6: Y-axis scale makes Prefill comparison invisible visually
  3. Limitations is a paragraph, not a subsection—consider making it §6.6 for visibility

**Page 6:**
- 看到的内容: Section 7 (Related Work continued), Section 8 (Conclusion), References
- 具体观察:
  - Related Work has three paragraphs: LLM Compression, KV Cache & Attention Optimization, Inference Frameworks
  - Covers: SparseGPT, GPTQ, AWQ, PaLU, MQA, GQA, StreamingLLM, FlashAttention, TensorRT, vLLM, TGI
  - References section lists ~14-21 citations in ACM format
  - Conclusion summarizes: "dimensional collapse", three root causes (FA slow paths +30-45%, TC 58%, Vec 50%), L2 5.8% not significant
  - Mentions 6.9× ROI and 96.9% theoretical dimensions would benefit
  - Final paragraph mentions future work: H100+ generalization, perplexity validation
- 问题/建议:
  1. References appear complete but consider adding more recent 2024-2025 papers
  2. Conclusion is well-written but could be slightly tightened

### Figure-by-Figure Assessment

| Figure | 位置 | 你观察到的具体内容 | 问题 |
|--------|------|-------------------|------|
| Fig 1 | Page 1 | Overview diagram showing "Compressed LLM" → "Dimensional Collapse" → "Dimension Repair" flow. Contains boxes with labels "head_dim=107" (red), "head_dim=112" (green). Caption states "88% latency increase" and "30% performance" recovery. Blue/orange colors. Annotations: TC 58%, Vec 50%, BW 40%. | Text within boxes is small (~6-7pt); increase to 8pt minimum |
| Fig 2 | Page 2 | Bar chart of per-head dimension distribution. X-axis: 112-126. Y-axis: Count 0-150. Green bar only at d=120 (8-aligned), all others red (misaligned). Total KV heads: 512, Misaligned: 96.9%. Caption notes "Production PaLU checkpoints use internal quantization that enforces alignment" | X-axis tick labels small (~7pt); d=120 bar should be more distinct |
| Fig 3 | Page 2 | Line plot of SDPA latency vs head dimension. X-axis: 80-160 (Head Dimension). Y-axis: 1.0-2.5 ms (Latency). Two lines: 8-aligned and Misaligned. Shaded confidence regions (±1std, light blue). Clear staircase pattern. d=107 at ~2.15ms vs d=96 at ~1.14ms. | Confidence band too light (alpha ~0.2) for print; Legend position inside plot may overlap with data at ~150 |
| Fig 4 | Page 3 | Horizontal bar chart showing root cause breakdown. Bars: "Tensor Core" 58%, "Vec. Load" 50%, "SDPA BW" 40%, "L2 Cache" 5.8%. Caption confirms primary causes. | L2 bar barely visible; "Not Confirmed" status should have visual distinction (grey/hatching) |
| Fig 5 | Page 4 | Scatter plot of speedup vs memory overhead. X-axis: 0-10% (Memory Overhead). Y-axis: 0-30% (Speedup). Blue circles (MINIMAL), orange squares (OPTIMAL). Points labeled d=107, 114, 117, 120, 121, 125. d=120 at (0%, 0%). | **CRITICAL**: Labels overlap severely (d=107, 121, 125 cluster). d=120 validation point needs callout. |
| Fig 6 | Page 5 | Dual-panel bar chart. Left: Prefill throughput (~9870 vs ~9672). Right: Decode throughput (119 vs 1371). Labels show "-2.0%" and "+11.5x". Caption: "Compression benefit, not repair benefit". | "11.5×" label small (~7pt); Prefill bars nearly identical visually—correct representation but dramatic contrast with Decode |

### Table Assessment

| Table | 你观察到的具体内容 | 问题 |
|-------|-------------------|------|
| Table 1 | Backend latency: d=96,104,107,112,128. AUTO=FLASH (identical). d=107 MEM_EFF="N/A*". MATH 26-28ms (12.6× slower). Scriptsize std format. | AUTO/FLASH columns redundant; std formatting inconsistent |
| Table 2 | Root cause: H1-H4 with Status (Confirmed/Not confirmed), Impact (58%/5.8%/40%/50%), Root Cause. H2 (L2) Not confirmed. | Well-formatted; clear visual hierarchy |
| Table 3 | Padding rescue: d=107 base 2.064ms, d=112 1.490ms 1.39×, d=128 1.506ms 1.37×. Shows d=112 optimal. | Speedup format "1.39×" differs from Table 5 percentage format |
| Table 4 | Memory: MINIMAL 3.72%, OPTIMAL 7.20%. Simple. | Potentially redundant with Table 5 info |
| Table 5 | SDPA repair: 6 dims, Original/Minimal/Optimal columns with Δ. d=120 shows 0% Δ for MINIMAL (validates hypothesis). | Many columns make it dense but comprehensive |
| Table 6 | E2E: Prefill 9870→9672 (-2%), Decode 119→1371 (+11.5×), Memory -0.6%. | Caption clarifies "compression benefit, not repair benefit"—crucial |

### Visual Issues Summary

**必须列出至少 3 个视觉问题**:

1. **Figure 5 (Page 4) - CRITICAL**: Data point labels (d=107, d=114, d=117, d=120, d=121, d=125) overlap severely in upper-left region. The cluster of d=107, d=121, d=125 is nearly unreadable. Need staggered labels or leader lines.

2. **Figure 2 (Page 2)**: The "theoretical" qualifier in caption is crucial but buried mid-sentence. Should be at beginning and bolded: "**Theoretical** dimension distribution from Fisher-information-based rank allocation..."

3. **Figure 4 (Page 3)**: The L2 Cache bar (5.8%) is barely visible due to scale. More importantly, the "Not Confirmed" status of H2 should be visually distinguished from confirmed hypotheses—use grey color or hatching.

4. **Figure 1 (Page 1)**: Text annotations ("88% latency increase", "30% performance recovery") are approximately 6-7pt, borderline readable at print scale. Should be 8pt minimum.

5. **Figure 3 (Page 2)**: The light blue confidence band (±1 std) is very light and may not reproduce well in print. Consider darker shading (alpha 0.3-0.4) or explicit error bars for key points.

6. **Figure 6 (Page 5)**: The "11.5×" annotation on the decode bar is small (~7pt). Given this is a key result, it should be more prominent (9-10pt).

7. **Table format inconsistency**: Table 3 uses multiplier (1.39×) while Table 5 uses percentage (+27.8%) for speedup—consider standardizing.

---

## Improvement Checklist for Writer Agent

### High Priority (Must Fix)
- [ ] **Figure 5**: Fix label overlap. Use leader lines, staggered positions, or annotation callouts. Add special callout for d=120 (0%, 0%) validating hypothesis.
- [ ] **Figure 2 caption**: Lead with "**Theoretical** dimension distribution..." Make "theoretical/unconstrained" prominent.
- [ ] **M1 - Scope clarification**: Add prominent clarification in §3 that 96.9% misalignment is from theoretical analysis, not production PaLU models
- [ ] **M2 - E2E validation**: Either add E2E experiments with truly misaligned models (RAP SVD), or explicitly downscope contribution to kernel-level insights
- [ ] **Figure 1**: Increase text annotation font to 8pt minimum

### Medium Priority (Recommended)
- [ ] **Figure 4**: Use grey/hatching for L2 Cache bar to distinguish "Not Confirmed" from confirmed causes
- [ ] **Figure 3**: Darken confidence band (alpha 0.3-0.4) or add explicit error bars for key points
- [ ] **Table 1**: Merge AUTO/FLASH columns or explain why both shown
- [ ] **Figure 6**: Increase "11.5×" font size to 9-10pt
- [ ] **Speedup format**: Standardize to percentage for <2× improvements throughout
- [ ] **§6.5 Limitations**: Make it a subsection header (§6.6) for visibility
- [ ] **§6.4**: Consider condensing the orthogonal PaLU study to save space

### Low Priority (Optional)
- [ ] **Table 4**: Consider removing (redundant with Table 5)
- [ ] **Figure 5 X-axis**: Reduce range from 0-20% to 0-12% to better spread data points
- [ ] **Related work**: Add discussion of dimension handling in vLLM/TensorRT-LLM
- [ ] **§4.2 Footnote**: Shorten or move FlashAttention kernel dispatch details to Related Work
- [ ] **Terminology**: Standardize "dimensional collapse" capitalization

---

## Reviewer Confidence

**Confidence Score:** 4/5

**Expertise Areas:**
- GPU architecture and Tensor Core optimization
- LLM inference systems
- Low-rank matrix compression
- PyTorch/CUDA kernel performance analysis

**Limitations:**
- Cannot verify experimental measurements (would need A100 hardware access)
- Cannot access paper_example reference papers for direct comparison (directory does not exist)
- Limited knowledge of PaLU internals beyond what's in findings.yaml
- Cannot assess novelty relative to unpublished/concurrent work

---

## Additional Notes for Writer Agent

### Data Consistency Check

Based on findings.yaml, the following data claims in the paper are **verified**:
- F1.1: head_dim=107 causes 88% latency increase vs head_dim=96 ✓
- F2.3.1: Tensor Core 58% slowdown, TC utilization 30%→12% ✓
- F2.3.2: L2 cache waste 5.8% (negligible) ✓
- F2.3.3: Vectorized load 50% throughput loss ✓
- F2.3.4: SDPA bandwidth 40% efficiency loss ✓
- F4.2: Dimension repair speedups 24-28% ✓

### Critical Insight from findings.yaml

The `c5_status_summary` section reveals:
> "所有可用的 PaLU checkpoint（ratio 0.5-0.9）都使用了 32-倍数量化约束，维度 100% 对齐"

This means the paper's scope must be clearly limited to **theoretical/unconstrained SVD compression**, not production PaLU. The paper does acknowledge this but could be even clearer.

### Suggested Scope Revision

Consider adding a sentence like:
> "Our repair targets compression methods that do not include internal alignment constraints. Production PaLU checkpoints enforce 32-multiple alignment, making our repair unnecessary for those specific models. However, vanilla SVD, Fisher-optimal rank allocation, and future accuracy-focused methods that relax alignment constraints would benefit directly from our approach."

---

*Reviewer: Paper Reviewer Agent*
*Date: 2026-01-27*
