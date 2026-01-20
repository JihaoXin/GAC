# Night Sweep Experiments Plan

## Overview

This document describes the 6 experiments (S1, S2, G3, G4, P1, HET1) designed as the next step for validating GPU irregular dimension effects. These experiments build upon the initial benchmark suite to provide deeper insights into dimension sensitivity, backend selection, and optimization strategies.

## Motivation

The initial benchmark suite validated that irregular dimensions (like `head_dim=107`) cause significant performance degradation. The night sweep experiments extend this work to:

1. **Stepwise cliffs**: Identify exact dimension thresholds where performance drops occur
2. **Backend forcing**: Understand backend selection behavior and forced fallbacks
3. **Padding rescue**: Quantify the trade-off between memory overhead and performance gain
4. **Heterogeneous batching**: Measure the penalty of mixed-dimension operations

## Experiments

### S1: SDPA Dense Sweep

**Purpose**: Identify stepwise performance cliffs across a dense range of head dimensions.

**Design**:
- Dense sweep: head_dim from 64 to 160
- Fine-grained steps (1 for 64-128, 2 for 128-160) to catch cliffs
- Two fixed shapes to test sensitivity across batch/sequence combinations

**Expected Insights**:
- Exact dimension thresholds where FlashAttention becomes unavailable
- Performance cliffs at alignment boundaries (64, 96, 128)
- Smooth vs. stepwise degradation patterns

### S2: SDPA Backend Forced

**Purpose**: Understand backend selection behavior and forced fallback performance.

**Design**:
- Test irregular dimensions [96, 104, 107, 112, 120, 128]
- Force each backend: AUTO, FLASH, MEM_EFFICIENT, MATH
- Record which backends are actually available/used

**Expected Insights**:
- Which backends reject irregular dimensions
- Performance difference between forced backends
- Backend selection heuristics in PyTorch

### G3: GEMM K Dimension Dense

**Purpose**: Measure Tensor Core sensitivity to reduction dimension alignment.

**Design**:
- Dense K sweep: 64 to 160 (same stepping as S1)
- Test both FP16 and BF16
- Fixed M=4096, N=4096

**Expected Insights**:
- K dimension alignment requirements for Tensor Core
- Performance cliffs at K=96, 112, 128
- FP16 vs BF16 sensitivity differences

### G4: GEMM N Dimension Dense (Projection-like)

**Purpose**: Measure projection operation sensitivity to output dimension alignment.

**Design**:
- Dense N sweep: 64 to 160
- Multiple M values: [1024, 4096, 16384]
- Fixed K=4096 (typical d_model)

**Expected Insights**:
- N dimension alignment effects
- M scaling behavior with irregular N
- Projection operation optimization opportunities

### P1: Padding Rescue

**Purpose**: Quantify the trade-off between memory overhead and performance gain from padding.

**Design**:
- Compare logical_dim=107 vs padded 112 and 128
- Test both SDPA and GEMM operations
- Measure latency improvement vs memory overhead

**Expected Insights**:
- Is padding worth the memory cost?
- Optimal padding target (112 vs 128)
- Operation-specific padding benefits

### HET1: Heterogeneous Head Batching Penalty

**Purpose**: Measure the performance cost of mixed-dimension operations vs uniform operations.

**Design**:
- Compare uniform GEMM (N=4096) vs heterogeneous groups
- Test mild, medium, and severe heterogeneity patterns
- Measure latency, GEMM call count, and effective TFLOPs

**Expected Insights**:
- Cost of splitting operations
- Heterogeneity penalty magnitude
- When uniform batching is worth the memory overhead

## Implementation Notes

### Timing Quality

- Warmup: 50 iterations (higher than initial suite)
- Measurement: 200 iterations per trial
- Trials: 3 trials per configuration
- Statistics: mean, std, p50, p90, p99

### Result Structure

Each experiment writes:
- `config.json`: Experiment configuration
- `env.json`: Environment metadata
- `raw.json`: Complete raw results
- `summary.json`: Summary statistics
- `plots/*.png`: Visualization plots

### JSON Schema

Stable schema with fields:
```json
{
  "metadata": {...},
  "config": {...},
  "measurements": {
    "raw": [...],
    "stats": {...}
  },
  "derived": {
    "tflops": ...,
    "bandwidth": ...
  }
}
```

## Expected Outcomes

1. **Dimension Cliffs**: Clear identification of performance thresholds
2. **Backend Behavior**: Understanding of when/how backends fallback
3. **Padding Strategy**: Data-driven padding recommendations
4. **Batching Guidelines**: When to use uniform vs heterogeneous batching

## Next Steps After Night Sweep

1. Analyze results for publication-ready insights
2. Develop optimization recommendations
3. Create performance prediction models
4. Extend to H100 architecture testing
