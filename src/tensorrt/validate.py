"""Validate end-to-end TensorRT output against the PyTorch baseline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
from typing import Sequence

from src.backends import GenerationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-id", default="stabilityai/sdxl-turbo")
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=1e-1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/validation"))
    parser.add_argument(
        "--unet-engine", type=Path, default=Path("artifacts/tensorrt/unet/model.plan")
    )
    parser.add_argument(
        "--vae-engine",
        type=Path,
        default=Path("artifacts/tensorrt/vae_decoder/model.plan"),
    )
    return parser


def compare_images(reference, actual, *, rtol: float, atol: float) -> dict[str, float | bool]:
    import numpy as np

    reference_array = np.asarray(reference.convert("RGB"), dtype=np.float32) / 255.0
    actual_array = np.asarray(actual.convert("RGB"), dtype=np.float32) / 255.0
    if reference_array.shape != actual_array.shape:
        raise RuntimeError(
            f"Image shape mismatch: PyTorch {reference_array.shape}, TensorRT {actual_array.shape}"
        )
    absolute = np.abs(reference_array - actual_array)
    return {
        "passed": bool(np.allclose(reference_array, actual_array, rtol=rtol, atol=atol)),
        "rtol": rtol,
        "atol": atol,
        "mean_absolute_error": float(absolute.mean()),
        "max_absolute_error": float(absolute.max()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = GenerationConfig(prompt=args.prompt, seed=args.seed)
    import torch
    from src.baseline.pipeline import SDXLTurboBaseline

    baseline = SDXLTurboBaseline(args.model_id)
    reference = baseline.generate(config)
    torch.cuda.synchronize()
    del baseline
    gc.collect()
    torch.cuda.empty_cache()

    from src.tensorrt.pipeline import SDXLTurboTensorRT

    backend = SDXLTurboTensorRT(
        args.model_id, unet_engine=args.unet_engine, vae_engine=args.vae_engine
    )
    actual = backend.generate(config)
    torch.cuda.synchronize()
    result = compare_images(reference, actual, rtol=args.rtol, atol=args.atol)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = args.output_dir / "pytorch.png"
    actual_path = args.output_dir / "tensorrt.png"
    report_path = args.output_dir / "comparison.json"
    reference.save(reference_path)
    actual.save(actual_path)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_id,
        "prompt": args.prompt,
        "seed": args.seed,
        "resolution": [512, 512],
        "pytorch_image": str(reference_path),
        "tensorrt_image": str(actual_path),
        "numerical_comparison": result,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not result["passed"]:
        raise RuntimeError(
            f"TensorRT output failed numerical tolerance; see {report_path}"
        )
    print(f"TensorRT output validation passed: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

