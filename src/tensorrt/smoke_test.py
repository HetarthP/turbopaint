"""Smoke-test the hybrid TensorRT SDXL-Turbo backend."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from src.backends import GenerationConfig
from src.baseline.smoke_test import validate_saved_image
from src.tensorrt.pipeline import SDXLTurboTensorRT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-id", default="stabilityai/sdxl-turbo")
    parser.add_argument(
        "--unet-engine", type=Path, default=Path("artifacts/tensorrt/unet/model.plan")
    )
    parser.add_argument(
        "--vae-engine",
        type=Path,
        default=Path("artifacts/tensorrt/vae_decoder/model.plan"),
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = GenerationConfig(prompt=args.prompt, seed=args.seed)
    backend = SDXLTurboTensorRT(
        model_id=args.model_id,
        unet_engine=args.unet_engine,
        vae_engine=args.vae_engine,
    )
    import torch

    with torch.inference_mode():
        backend.generate(config)
        torch.cuda.synchronize()
        image = backend.generate(config)
        torch.cuda.synchronize()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("outputs") / f"tensorrt_smoke_{timestamp}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    validate_saved_image(output, (512, 512))
    print(f"TensorRT smoke test passed: {output} (512x512)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

