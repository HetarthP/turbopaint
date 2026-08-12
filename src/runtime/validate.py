"""Print the PyTorch/CUDA environment without loading an AI model."""

from __future__ import annotations

import json

from src.runtime.cuda import inspect_cuda_environment


def main() -> int:
    try:
        import torch
    except ImportError:
        import platform

        print(f"Python version: {platform.python_version()}")
        print("PyTorch version: not installed")
        print("CUDA available: unavailable (PyTorch is not installed)")
        print("Install a CUDA-enabled PyTorch build using the NVIDIA Linux setup in README.md.")
        return 1

    environment = inspect_cuda_environment(torch)
    print(json.dumps(environment.to_dict(), indent=2))
    return 0 if environment.cuda_available else 1


if __name__ == "__main__":
    raise SystemExit(main())

