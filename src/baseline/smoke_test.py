"""One warm-up and one validated SDXL-Turbo generation on CUDA."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from src.backends import GenerationConfig
from src.baseline.pipeline import SDXLTurboBaseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-test 512x512 FP16 SDXL-Turbo inference on CUDA."
    )
    parser.add_argument("prompt", help="Text prompt used for both generations")
    parser.add_argument("--model-id", default="stabilityai/sdxl-turbo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path (default: timestamped file in outputs/)",
    )
    return parser


def default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("outputs") / f"smoke_sdxl_turbo_{timestamp}.png"


def validate_saved_image(path: Path, expected_size: tuple[int, int]) -> None:
    if not path.is_file():
        raise RuntimeError(f"Smoke-test output was not created: {path}")
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        if image.size != expected_size:
            raise RuntimeError(
                f"Expected output dimensions {expected_size}, got {image.size}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = GenerationConfig(prompt=args.prompt, seed=args.seed)

    # Construction performs CUDA validation and model loading before inference.
    backend = SDXLTurboBaseline(model_id=args.model_id)
    import torch

    with torch.inference_mode():
        backend.generate(config)  # One unmeasured warm-up.
        torch.cuda.synchronize()
        image = backend.generate(config)  # One smoke-test generation.
        torch.cuda.synchronize()

    output_path = args.output or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)  # Saving occurs after GPU inference is complete.
    validate_saved_image(output_path, (config.width, config.height))
    print(f"Smoke test passed: {output_path} ({config.width}x{config.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

