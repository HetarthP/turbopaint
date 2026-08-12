"""Shared loading, export, validation, and diagnostic helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
from typing import Any, Sequence

from src.runtime import require_cuda
from src.tensorrt.export.contracts import contract_as_dict
from src.tensorrt.export.fingerprint import onnx_bundle_sha256

MODEL_ID = "stabilityai/sdxl-turbo"
OPSET_VERSION = 17


def load_export_dependencies() -> tuple[Any, Any]:
    try:
        import torch
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export requires CUDA-enabled PyTorch and requirements-onnx.txt"
        ) from exc
    require_cuda(torch)
    return torch, onnx


def export_onnx(
    *,
    torch_module: Any,
    onnx_module: Any,
    model: Any,
    inputs: Sequence[Any],
    input_names: Sequence[str],
    output_names: Sequence[str],
    output_path: Path,
    component: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs: dict[str, Any] = {
        "input_names": list(input_names),
        "output_names": list(output_names),
        "opset_version": OPSET_VERSION,
        "do_constant_folding": True,
    }
    signature = inspect.signature(torch_module.onnx.export).parameters
    if "dynamo" in signature:
        export_kwargs["dynamo"] = False
    # Large SDXL graphs require external weights. Older PyTorch exporters do
    # this automatically above 2 GiB and do not expose the keyword.
    if "external_data" in signature:
        export_kwargs["external_data"] = True

    try:
        with torch_module.inference_mode():
            torch_module.onnx.export(
                model,
                tuple(inputs),
                str(output_path),
                **export_kwargs,
            )
        # Passing a path allows the checker to validate models above 2 GiB and
        # resolve their external tensor data correctly.
        onnx_module.checker.check_model(str(output_path))
    except Exception as exc:
        write_export_failure(output_path, component, exc)
        raise RuntimeError(
            f"Failed to export {component}: {type(exc).__name__}: {exc}. "
            f"Diagnostic written beside {output_path}. Check the diagnostic "
            "for unsupported operators; do not continue to TensorRT conversion."
        ) from exc


def write_export_failure(output_path: Path, component: str, error: Exception) -> Path:
    diagnostic = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "tensor_contract": contract_as_dict(component),
        "next_action": (
            "Identify the first unsupported operator in the exporter error. "
            "Upgrade only within documented compatibility, add a supported "
            "decomposition, or keep that subgraph in PyTorch."
        ),
    }
    path = output_path.with_suffix(".export_error.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
    return path


def validate_onnx_output(
    *,
    onnx_path: Path,
    input_names: Sequence[str],
    inputs: Sequence[Any],
    expected: Any,
    rtol: float,
    atol: float,
) -> dict[str, float | bool]:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "Validation requires NumPy and a CUDA-compatible onnxruntime-gpu build"
        ) from exc

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "onnxruntime-gpu does not expose CUDAExecutionProvider. Install an "
            "ONNX Runtime build matching PyTorch's CUDA and cuDNN major versions."
        )
    try:
        session = ort.InferenceSession(
            str(onnx_path), providers=["CUDAExecutionProvider"]
        )
    except Exception as exc:
        raise RuntimeError(
            "ONNX Runtime CUDA could not load the graph. This commonly means "
            "an unsupported ONNX operator/type or a CUDA/cuDNN package mismatch: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("ONNX Runtime did not activate CUDAExecutionProvider")

    feed = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(input_names, inputs, strict=True)
    }
    actual = session.run(None, feed)[0]
    reference = expected.detach().float().cpu().numpy()
    actual_f32 = actual.astype(np.float32)
    absolute = np.abs(actual_f32 - reference)
    result: dict[str, float | bool] = {
        "passed": bool(np.allclose(actual_f32, reference, rtol=rtol, atol=atol)),
        "rtol": rtol,
        "atol": atol,
        "max_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
    }
    if not result["passed"]:
        raise RuntimeError(f"ONNX numerical validation failed: {result}")
    return result


def write_validation_report(
    *,
    output_path: Path,
    component: str,
    model_id: str,
    result: dict[str, float | bool],
) -> Path:
    import onnx

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "model_id": model_id,
        "status": "validated",
        "opset_version": OPSET_VERSION,
        "fixed_shapes": True,
        "onnx_bundle_sha256": onnx_bundle_sha256(output_path, onnx),
        "tensor_contract": contract_as_dict(component),
        "numerical_validation": result,
    }
    path = output_path.with_suffix(".validation.json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path
