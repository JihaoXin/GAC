"""Standalone script to collect environment metadata."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import save_environment, collect_environment
import argparse


def main():
    parser = argparse.ArgumentParser(description="Collect environment metadata")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("env.json"),
        help="Output JSON path"
    )
    args = parser.parse_args()
    
    env = save_environment(args.output)
    print(f"Environment metadata saved to {args.output}")
    print()
    print("Summary:")
    print(f"  GPU: {env.get('gpu', [{}])[0].get('name', 'N/A')}")
    print(f"  PyTorch: {env.get('pytorch', {}).get('version', 'N/A')}")
    print(f"  CUDA: {env.get('pytorch', {}).get('cuda_version', 'N/A')}")
    print(f"  Python: {env.get('python_version_info', {})}")


if __name__ == "__main__":
    main()
