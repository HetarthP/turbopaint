"""Build the fixed-shape FP16 SDXL-Turbo UNet TensorRT engine."""

from pathlib import Path
from typing import Sequence

from src.tensorrt.build.cli import run_component


def main(argv: Sequence[str] | None = None) -> int:
    return run_component(
        "unet",
        Path("artifacts/onnx/unet/model.onnx"),
        Path("artifacts/tensorrt/unet/model.plan"),
        argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())

