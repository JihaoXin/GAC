# Experiment Plan: GPU Irregular Dimensions Benchmark Suite

## Motivation

This benchmark suite validates the performance impact of irregular dimensions (particularly `head_dim=107`) on NVIDIA A100 GPUs, as documented in `G-Compress_DeepResearch.md`. The research identifies that irregular dimensions cause severe performance degradation through multiple mechanisms:

1. **Memory Alignment Issues**: 16-30% bandwidth waste from non-aligned memory access (32-byte L2 cache sectors, 128-byte DRAM bursts)
2. **Tensor Core Constraints**: K dimension must be multiple of 16 for optimal performance; irregular dimensions require padding (~16% wasted compute)
3. **FlashAttention Fallback**: Requires `head_dim` multiple of 8, optimized for {32, 64, 96, 128, 256}; irregular dims trigger fallback to Math path (O(N²) complexity)
4. **Hardware Acceleration Failures**: On H100, TMA/WGMMA require 16/32-byte aligned strides; irregular dimensions disable these accelerators

## Constraints from Research Document

### Physical Layer Constraints
- **L2 Cache Sector**: 32 bytes minimum transfer unit
- **DRAM Burst**: 128 bytes typical burst length
- **Memory Alignment**: Dimensions should align to 16/32 byte boundaries for efficient access
- **Example**: `head_dim=107` (214 bytes) cannot be evenly divided by 32, causing over-fetching

### Compute Architecture Constraints
- **Tensor Core**: K dimension must be multiple of 16 for optimal performance
- **A100 cp.async**: Requires 16-byte alignment for efficient async copy
- **H100 TMA**: Requires stride to be 16/32-byte aligned (not directly testable on A100, but dimension effects apply)

### Software Stack Constraints
- **FlashAttention**: Requires `head_dim` multiple of 8, optimized for {32, 64, 96, 128, 256}
- **CUTLASS**: Irregular dimensions require predicate guards, causing warp divergence
- **Padding Strategy**: Physical padding to 128 increases memory footprint by ~20% (107→128)

## Experiment Design

### Experiment A: GEMM Projection-Like Shapes

**Purpose**: Test QKV projection patterns `(M, K) @ (K, N)` where N (head_dim-like) varies.

**Fixed dimensions:**
- M (batch × tokens): [1024, 4096, 16384]
- K (d_model): 4096

**Varied dimensions:**
- N (head_dim-like): [96, 104, 107, 112, 120, 128]

**Rationale**: 
- 96, 112, 128 are aligned (multiples of 32 for FP16)
- 104 is even but not 32-aligned
- 107 is irregular (prime, not aligned)
- 120 is 8-aligned but not 32-aligned

**Expected Results**:
- Aligned dimensions (96, 112, 128) should show highest performance
- Irregular dimension (107) should show significant degradation
- Performance should correlate with alignment, not just dimension size

### Experiment B: GEMM Reduction Dimension Focus

**Purpose**: Test K as reduction dimension (critical for Tensor Core packing).

**Fixed dimensions:**
- M = 4096
- N = 4096

**Varied dimensions:**
- K ∈ [96, 104, 107, 112, 120, 128]

**Rationale**: Tensor Core efficiency heavily depends on K dimension alignment. K=107 should show worst performance due to padding requirements.

**Expected Results**:
- K=96, 112, 128 (16-aligned) should perform best
- K=107 should require padding to 112 or 128, wasting compute

### Experiment C: SDPA Backend Selection

**Purpose**: Test PyTorch `scaled_dot_product_attention` backend selection for irregular head_dim.

**Configuration:**
- Sequence length: [1024, 4096]
- Batch size: [1, 4, 8]
- Head count: 32 (fixed)
- Head dim: [96, 104, 107, 112, 120, 128]

**Backend Detection Strategy**:
1. Use `torch.backends.cuda.sdp_kernel()` context manager to detect available backends
2. Attempt to force each backend individually and measure timing
3. Infer backend used by comparing timing patterns
4. Record fallback behavior when FlashAttention rejects irregular dimensions

**Expected Results**:
- `head_dim=96, 112, 128`: FlashAttention backend should be used
- `head_dim=107`: Should fallback to Math backend (O(N²) complexity)
- Performance degradation should be 3-10x for irregular dimensions

## Metrics and Measurements

### Timing Metrics
- **Latency**: Mean, std, p50, p90, p99 (milliseconds)
- **Measurement Method**: CUDA events (`torch.cuda.Event`) with synchronization
- **Warmup**: 10 iterations to stabilize
- **Measurement**: 100 iterations for statistical significance

### Performance Metrics
- **TFLOPs**: `2 * M * N * K / (time_s * 1e12)` for GEMM
- **Bandwidth**: `(M*K + K*N + M*N) * dtype_size / (time_s * 1e9)` (GB/s)
- **Theoretical Peak**: A100 FP16 ~312 TFLOPS

### Backend Detection
- Record available backends (Flash, MemEfficient, Math)
- Attempt to infer backend used via timing heuristics
- Record errors/exceptions when backends fail

## Reproducibility Controls

### Deterministic Settings
- Fixed random seed (42)
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `torch.use_deterministic_algorithms(True)` (where possible)

### Environment Recording
- PyTorch version
- CUDA version (runtime and driver)
- GPU name and compute capability
- Backend settings (TF32, cuDNN, etc.)
- Python version
- System information

### Data Collection
- All results saved as JSON with metadata
- Configuration saved alongside results
- Environment metadata captured at start
- Raw timing data preserved for post-analysis

## Output Structure

```
results/
└── <experiment_name>/
    └── <timestamp>/
        ├── config.json          # Experiment configuration
        ├── env.json              # Environment metadata
        ├── gemm_results.json     # GEMM benchmark results
        ├── sdpa_results.json     # SDPA benchmark results
        └── plots/
            ├── gemm_latency_vs_dimension.png
            ├── gemm_tflops_vs_dimension.png
            ├── sdpa_latency_vs_head_dim.png
            └── sdpa_backend_vs_head_dim.png
```

## Validation Strategy

1. **Smoke Test**: Run single GEMM shape locally to verify measurement pipeline
2. **Reproducibility**: Fixed seed and deterministic flags ensure consistent results
3. **Quality Checks**: Verify measurements show expected patterns (irregular dims slower than aligned)
4. **No Internet**: All dependencies installable offline (standard PyPI packages)

## Success Criteria

1. Benchmark suite runs successfully on A100 via Slurm
2. Results clearly show performance degradation for irregular dimensions (107 vs 96/112/128)
3. SDPA results show backend fallback behavior (Flash → Math for irregular dims)
4. Plots are publication-ready with clear labels and error bars
5. JSON outputs are structured and parseable for further analysis
6. README enables users to reproduce experiments independently

## References

- `G-Compress_DeepResearch.md`: Comprehensive analysis of dimension effects on GPU performance
- Key findings: 16-30% bandwidth waste, Tensor Core padding overhead, FlashAttention fallback, TMA/WGMMA failures
