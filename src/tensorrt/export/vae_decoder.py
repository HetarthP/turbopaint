"""Export and numerically validate the fixed-shape SDXL-Turbo VAE decoder."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.tensorrt.export.common import (
    MODEL_ID,
    export_onnx,
    load_export_dependencies,
    validate_onnx_output,
    write_validation_report,
)
from src.tensorrt.export.contracts import COMPONENT_CONTRACTS
from src.tensorrt.export.wrappers import make_vae_decoder_wrapper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/onnx/vae_decoder/model.onnx"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch, onnx = load_export_dependencies()
    try:
        from diffusers import AutoencoderKL
    except ImportError as exc:
        raise RuntimeError("Install requirements-onnx.txt before exporting") from exc

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    # Match the stock SDXL pipeline's force_upcast decode behavior.
    vae = AutoencoderKL.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    wrapper = make_vae_decoder_wrapper(torch, vae)
    inputs = (torch.randn((1, 4, 64, 64), device=device, dtype=torch.float32),)
    contract = COMPONENT_CONTRACTS["vae_decoder"]
    input_names = [item.name for item in contract["inputs"]]
    output_names = [item.name for item in contract["outputs"]]
    with torch.inference_mode():
        expected = wrapper(*inputs)
    export_onnx(
        torch_module=torch,
        onnx_module=onnx,
        model=wrapper,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        output_path=args.output,
        component="vae_decoder",
    )
    result = validate_onnx_output(
        onnx_path=args.output,
        input_names=input_names,
        inputs=inputs,
        expected=expected,
        rtol=args.rtol,
        atol=args.atol,
    )
    report_path = write_validation_report(
        output_path=args.output,
        component="vae_decoder",
        model_id=args.model_id,
        result=result,
    )
    print(f"Validated VAE decoder ONNX: {args.output}")
    print(f"Validation report: {report_path}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

