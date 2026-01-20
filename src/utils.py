"""Common utilities for benchmarks."""
import torch
import numpy as np
from typing import Tuple


def set_deterministic(seed: int = 42):
    """Set deterministic flags for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Try to enable deterministic algorithms (may not be available for all ops)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype."""
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def allocate_tensors(
    shape_a: Tuple[int, ...],
    shape_b: Tuple[int, ...],
    dtype: torch.dtype,
    device: str = "cuda:0",
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate and initialize tensors for GEMM."""
    torch.manual_seed(seed)
    a = torch.randn(shape_a, dtype=dtype, device=device)
    b = torch.randn(shape_b, dtype=dtype, device=device)
    return a, b


def compute_statistics(values: list) -> dict:
    """Compute statistical summary of measurements."""
    if not values:
        return {}
    
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "count": len(values),
    }
