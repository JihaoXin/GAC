"""Environment metadata collection."""
import json
import platform
import sys
from pathlib import Path
from typing import Dict, Any

import torch


def collect_environment() -> Dict[str, Any]:
    """Collect comprehensive environment metadata."""
    env = {
        "python_version": sys.version,
        "python_version_info": {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }
    
    # PyTorch information
    if torch.cuda.is_available():
        env["pytorch"] = {
            "version": torch.__version__,
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "device_count": torch.cuda.device_count(),
        }
        
        # GPU information
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                "index": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": props.total_memory / (1024**3),
                "multiprocessor_count": props.multi_processor_count,
            })
        env["gpu"] = gpu_info
        
        # CUDA driver version (if accessible)
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                env["cuda_driver_version"] = result.stdout.strip().split('\n')[0]
        except Exception:
            pass
        
        # PyTorch backend settings
        pytorch_backends = {
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        }
        # allow_tf32 may not exist in all PyTorch versions
        if hasattr(torch.backends.cuda, "allow_tf32"):
            pytorch_backends["cuda_allow_tf32"] = torch.backends.cuda.allow_tf32
        env["pytorch_backends"] = pytorch_backends
    else:
        env["pytorch"] = {
            "version": torch.__version__,
            "cuda_available": False,
        }
    
    return env


def save_environment(output_path: Path):
    """Collect and save environment metadata to JSON file."""
    env = collect_environment()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(env, f, indent=2)
    return env


if __name__ == "__main__":
    """CLI entrypoint for collecting environment."""
    import argparse
    parser = argparse.ArgumentParser(description="Collect environment metadata")
    parser.add_argument("--output", type=Path, default=Path("env.json"), help="Output JSON path")
    args = parser.parse_args()
    
    env = save_environment(args.output)
    print(f"Environment metadata saved to {args.output}")
    print(f"GPU: {env.get('gpu', [{}])[0].get('name', 'N/A')}")
    print(f"PyTorch: {env.get('pytorch', {}).get('version', 'N/A')}")
    print(f"CUDA: {env.get('pytorch', {}).get('cuda_version', 'N/A')}")
