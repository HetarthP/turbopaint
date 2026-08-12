"""Build the fixed-shape FP32 SDXL-Turbo VAE decoder TensorRT engine."""

from pathlib import Path
from typing import Sequence

from src.tensorrt.build.cli import run_component


def main(argv: Sequence[str] | None = None) -> int:
    return run_component(
        "vae_decoder",
        Path("artifacts/onnx/vae_decoder/model.onnx"),
        Path("artifacts/tensorrt/vae_decoder/model.plan"),
        argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())

