"""Configuration system for benchmarks."""
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path


@dataclass
class BenchmarkConfig:
    """Base configuration for all benchmarks."""
    experiment_name: str
    device: str = "cuda:0"
    dtypes: List[str] = field(default_factory=lambda: ["float16"])
    warmup_iterations: int = 10
    measurement_iterations: int = 100
    seed: int = 42
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def save(self, path: Path):
        """Save configuration to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class GEMMConfig(BenchmarkConfig):
    """Configuration for GEMM benchmarks."""
    # Experiment A: Projection-like shapes
    m_values: List[int] = field(default_factory=lambda: [1024, 4096, 16384])
    k_values: List[int] = field(default_factory=lambda: [96, 104, 107, 112, 120, 128])
    n_values: List[int] = field(default_factory=lambda: [96, 104, 107, 112, 120, 128])
    
    # Experiment B: Reduction dimension focus
    # When reduction_k is True, we test K as reduction dim with fixed M, N
    reduction_k: bool = False
    reduction_m: int = 4096  # Fixed M when reduction_k=True
    reduction_n: int = 4096  # Fixed N when reduction_k=True
    
    # Shape patterns to test
    test_projection_qkv: bool = True  # (M, K) @ (K, N)
    test_projection_output: bool = False  # (M, N) @ (N, K)


@dataclass
class SDPAConfig(BenchmarkConfig):
    """Configuration for SDPA benchmarks."""
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8])
    seq_lengths: List[int] = field(default_factory=lambda: [1024, 4096])
    n_heads: int = 32
    head_dims: List[int] = field(default_factory=lambda: [96, 104, 107, 112, 120, 128])
    
    # Backend detection
    detect_backend: bool = True
    try_all_backends: bool = False  # If True, try to force each backend


@dataclass
class ExperimentConfig:
    """Top-level configuration for running experiments."""
    output_dir: Path = field(default_factory=lambda: Path("results"))
    run_gemm: bool = True
    run_sdpa: bool = True
    gemm_config: Optional[GEMMConfig] = None
    sdpa_config: Optional[SDPAConfig] = None
    
    def __post_init__(self):
        """Initialize default configs if not provided."""
        if self.gemm_config is None:
            self.gemm_config = GEMMConfig(experiment_name="gemm_sweep")
        if self.sdpa_config is None:
            self.sdpa_config = SDPAConfig(experiment_name="sdpa_backend")
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "output_dir": str(self.output_dir),
            "run_gemm": self.run_gemm,
            "run_sdpa": self.run_sdpa,
            "gemm_config": self.gemm_config.to_dict() if self.gemm_config else None,
            "sdpa_config": self.sdpa_config.to_dict() if self.sdpa_config else None,
        }
    
    def save(self, path: Path):
        """Save configuration to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
