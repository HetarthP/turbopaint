"""Shared fixed-shape TensorRT engine builder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from src.runtime import require_cuda
from src.tensorrt.export.contracts import COMPONENT_CONTRACTS, contract_as_dict
from src.tensorrt.export.fingerprint import onnx_bundle_sha256


@dataclass(frozen=True)
class BuildRequest:
    component: str
    onnx_path: Path
    engine_path: Path
    workspace_gib: float = 4.0
    force: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.component not in COMPONENT_CONTRACTS:
            raise ValueError(f"Unknown component: {self.component}")
        if self.workspace_gib <= 0:
            raise ValueError("workspace_gib must be positive")


def precision_for(component: str) -> str:
    if component == "unet":
        return "fp16"
    if component == "vae_decoder":
        return "fp32"
    raise ValueError(f"Unknown component: {component}")


def _metadata_path(engine_path: Path) -> Path:
    return engine_path.with_suffix(".metadata.json")


def _failure_path(engine_path: Path) -> Path:
    return engine_path.with_suffix(".build_error.json")


def _runtime_identity(
    torch: Any,
    trt: Any,
    onnx: Any,
    component: str,
    onnx_path: Path,
) -> dict[str, Any]:
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    return {
        "component": component,
        "onnx_bundle_sha256": onnx_bundle_sha256(onnx_path, onnx),
        "tensorrt_version": str(trt.__version__),
        "gpu_name": str(torch.cuda.get_device_name(device)),
        "gpu_compute_capability": [int(capability[0]), int(capability[1])],
        "precision": precision_for(component),
        "tensor_contract": contract_as_dict(component),
    }


def _cache_is_valid(engine_path: Path, identity: dict[str, Any]) -> bool:
    metadata_path = _metadata_path(engine_path)
    if not engine_path.is_file() or engine_path.stat().st_size <= 0 or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(metadata.get(key) == value for key, value in identity.items())


def _trt_dtype_name(dtype: Any) -> str:
    value = str(dtype).lower()
    aliases = {
        "datatype.float": "float32",
        "datatype.half": "float16",
        "float": "float32",
        "half": "float16",
        "float32": "float32",
        "float16": "float16",
    }
    return aliases.get(value, value.rsplit(".", 1)[-1])


def validate_network_contract(network: Any, component: str) -> None:
    contract = COMPONENT_CONTRACTS[component]
    actual_inputs = {
        network.get_input(index).name: network.get_input(index)
        for index in range(network.num_inputs)
    }
    actual_outputs = {
        network.get_output(index).name: network.get_output(index)
        for index in range(network.num_outputs)
    }
    errors: list[str] = []
    for group, actual in (("inputs", actual_inputs), ("outputs", actual_outputs)):
        expected_names = {item.name for item in contract[group]}
        if set(actual) != expected_names:
            errors.append(
                f"{group} names: expected {sorted(expected_names)}, got {sorted(actual)}"
            )
        for item in contract[group]:
            tensor = actual.get(item.name)
            if tensor is None:
                continue
            shape = tuple(int(dimension) for dimension in tensor.shape)
            dtype = _trt_dtype_name(tensor.dtype)
            if shape != item.shape:
                errors.append(f"{item.name} shape: expected {item.shape}, got {shape}")
            if dtype != item.dtype:
                errors.append(f"{item.name} dtype: expected {item.dtype}, got {dtype}")
    if errors:
        raise RuntimeError("TensorRT network contract mismatch:\n- " + "\n- ".join(errors))


def parser_errors(parser: Any) -> list[str]:
    return [str(parser.get_error(index)) for index in range(parser.num_errors)]


def _write_failure(request: BuildRequest, identity: dict[str, Any], error: Exception) -> Path:
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        **identity,
        "request": {**asdict(request), "onnx_path": str(request.onnx_path), "engine_path": str(request.engine_path)},
        "error_type": type(error).__name__,
        "error": str(error),
    }
    path = _failure_path(request.engine_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def build_engine(request: BuildRequest) -> tuple[Path, Path, bool]:
    """Build or reuse a TensorRT plan; return engine, metadata, and cache-hit."""
    if not request.onnx_path.is_file():
        raise FileNotFoundError(f"Validated ONNX model not found: {request.onnx_path}")
    validation_path = request.onnx_path.with_suffix(".validation.json")
    if not validation_path.is_file():
        raise FileNotFoundError(
            f"ONNX validation report not found: {validation_path}. Run the exporter first."
        )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "validated" or validation.get("component") != request.component:
        raise RuntimeError(f"Invalid ONNX validation report: {validation_path}")
    if validation.get("tensor_contract") != contract_as_dict(request.component):
        raise RuntimeError(
            f"Stale ONNX validation contract in {validation_path}; rerun the exporter"
        )

    try:
        import torch
        import onnx
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "Engine building requires CUDA-enabled PyTorch and the full TensorRT Python package"
        ) from exc
    require_cuda(torch)
    identity = _runtime_identity(torch, trt, onnx, request.component, request.onnx_path)
    validated_hash = validation.get("onnx_bundle_sha256")
    if validated_hash is not None and validated_hash != identity["onnx_bundle_sha256"]:
        raise RuntimeError(
            f"ONNX bundle differs from {validation_path}; rerun numerical validation"
        )
    if not request.force and _cache_is_valid(request.engine_path, identity):
        return request.engine_path, _metadata_path(request.engine_path), True

    request.engine_path.parent.mkdir(parents=True, exist_ok=True)
    severity = trt.Logger.VERBOSE if request.verbose else trt.Logger.INFO
    logger = trt.Logger(severity)
    try:
        builder = trt.Builder(logger)
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(explicit_batch)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(request.onnx_path)):
            details = parser_errors(parser)
            raise RuntimeError(
                "TensorRT ONNX parser rejected the graph:\n"
                + "\n".join(f"[{index}] {message}" for index, message in enumerate(details))
            )
        validate_network_contract(network, request.component)

        config = builder.create_builder_config()
        workspace_bytes = int(request.workspace_gib * 1024**3)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        if request.component == "unet":
            if not builder.platform_has_fast_fp16:
                raise RuntimeError("GPU does not report fast FP16 support required by the UNet build")
            config.set_flag(trt.BuilderFlag.FP16)
        elif hasattr(trt.BuilderFlag, "TF32"):
            config.clear_flag(trt.BuilderFlag.TF32)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            details = parser_errors(parser)
            suffix = "\n".join(details) if details else "See TensorRT logger output above."
            raise RuntimeError(f"TensorRT engine build returned no serialized plan.\n{suffix}")
        temporary_engine = request.engine_path.with_suffix(".plan.tmp")
        temporary_engine.write_bytes(bytes(serialized))
        if temporary_engine.stat().st_size <= 0:
            raise RuntimeError("TensorRT wrote an empty engine file")
        temporary_engine.replace(request.engine_path)

        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "built",
            **identity,
            "workspace_bytes": workspace_bytes,
            "engine_path": str(request.engine_path),
            "engine_file_size_bytes": request.engine_path.stat().st_size,
            "onnx_path": str(request.onnx_path),
            "onnx_validation_report": str(validation_path),
        }
        metadata_path = _metadata_path(request.engine_path)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        failure_path = _failure_path(request.engine_path)
        if failure_path.exists():
            failure_path.unlink()
        return request.engine_path, metadata_path, False
    except Exception as exc:
        failure_path = _write_failure(request, identity, exc)
        raise RuntimeError(f"{exc}\nBuild diagnostic: {failure_path}") from exc
