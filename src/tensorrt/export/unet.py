"""Export and numerically validate the fixed-shape FP16 SDXL-Turbo UNet."""

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
from src.tensorrt.export.wrappers import make_unet_wrapper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/onnx/unet/model.onnx")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch, onnx = load_export_dependencies()
    try:
        from diffusers import EulerAncestralDiscreteScheduler, UNet2DConditionModel
    except ImportError as exc:
        raise RuntimeError("Install requirements-onnx.txt before exporting") from exc

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    dtype = torch.float16
    unet = UNet2DConditionModel.from_pretrained(
        args.model_id, subfolder="unet", variant="fp16", torch_dtype=dtype
    ).to(device)
    unet.eval()
    if int(unet.config.in_channels) != 4 or int(unet.config.cross_attention_dim) != 2048:
        raise RuntimeError(
            "Loaded UNet does not match the SDXL export contract: expected "
            "in_channels=4 and cross_attention_dim=2048"
        )
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )
    scheduler.set_timesteps(1, device=device)
    timestep = scheduler.timesteps[:1].to(device=device, dtype=torch.float32)
    wrapper = make_unet_wrapper(torch, unet)
    inputs = (
        torch.randn((1, 4, 64, 64), device=device, dtype=dtype),
        timestep,
        torch.randn((1, 77, 2048), device=device, dtype=dtype),
        torch.randn((1, 1280), device=device, dtype=dtype),
        torch.tensor(
            [[512, 512, 0, 0, 512, 512]], device=device, dtype=dtype
        ),
    )
    contract = COMPONENT_CONTRACTS["unet"]
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
        component="unet",
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
        component="unet",
        model_id=args.model_id,
        result=result,
    )
    print(f"Validated UNet ONNX: {args.output}")
    print(f"Validation report: {report_path}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
