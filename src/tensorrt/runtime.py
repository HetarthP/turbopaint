"""TensorRT 10/11 named-tensor execution backed by persistent CUDA buffers."""

from __future__ import annotations

import mmap
from pathlib import Path
from typing import Any, Mapping

from src.tensorrt.export.contracts import COMPONENT_CONTRACTS


def torch_dtype(torch: Any, dtype: str) -> Any:
    mapping = {"float16": torch.float16, "float32": torch.float32}
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported TensorRT tensor dtype: {dtype}") from exc


class TensorRTEngine:
    """Own a TensorRT engine/context and reuse its CUDA output buffers."""

    def __init__(self, component: str, engine_path: Path, torch: Any, trt: Any) -> None:
        if component not in COMPONENT_CONTRACTS:
            raise ValueError(f"Unknown component: {component}")
        if not engine_path.is_file() or engine_path.stat().st_size <= 0:
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        self.component = component
        self.engine_path = engine_path
        self._torch = torch
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        # mmap avoids a second large Python bytes allocation during load.
        with engine_path.open("rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as plan:
                self._engine = self._runtime.deserialize_cuda_engine(plan)
        if self._engine is None:
            raise RuntimeError(f"TensorRT could not deserialize engine: {engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(f"TensorRT could not create execution context: {engine_path}")

        contract = COMPONENT_CONTRACTS[component]
        actual_names = {
            self._engine.get_tensor_name(index)
            for index in range(self._engine.num_io_tensors)
        }
        expected_names = {item.name for group in contract.values() for item in group}
        if actual_names != expected_names:
            raise RuntimeError(
                f"TensorRT engine contract mismatch for {component}: expected "
                f"{sorted(expected_names)}, got {sorted(actual_names)}"
            )
        for item in (*contract["inputs"], *contract["outputs"]):
            shape = tuple(int(value) for value in self._engine.get_tensor_shape(item.name))
            dtype = str(self._engine.get_tensor_dtype(item.name)).lower()
            expected_dtype = {"float16": "float16", "float32": "float32"}[item.dtype]
            normalized_dtype = {
                "datatype.half": "float16",
                "datatype.float": "float32",
                "half": "float16",
                "float": "float32",
            }.get(dtype, dtype.rsplit(".", 1)[-1])
            if shape != item.shape or normalized_dtype != expected_dtype:
                raise RuntimeError(
                    f"TensorRT engine tensor {item.name} expected {item.shape} "
                    f"{expected_dtype}, got {shape} {normalized_dtype}"
                )
        self._inputs = {item.name: item for item in contract["inputs"]}
        self._outputs = {
            item.name: torch.empty(
                item.shape, dtype=torch_dtype(torch, item.dtype), device="cuda"
            )
            for item in contract["outputs"]
        }
        for name, tensor in self._outputs.items():
            if not self._context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"TensorRT rejected output binding: {name}")

    def execute(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        if set(inputs) != set(self._inputs):
            raise ValueError(
                f"{self.component} inputs must be {sorted(self._inputs)}, got {sorted(inputs)}"
            )
        for name, contract in self._inputs.items():
            tensor = inputs[name]
            if not tensor.is_cuda:
                raise ValueError(f"TensorRT input {name} must be a CUDA tensor")
            if tuple(tensor.shape) != contract.shape:
                raise ValueError(
                    f"TensorRT input {name} expected {contract.shape}, got {tuple(tensor.shape)}"
                )
            if tensor.dtype != torch_dtype(self._torch, contract.dtype):
                raise ValueError(
                    f"TensorRT input {name} expected {contract.dtype}, got {tensor.dtype}"
                )
            if not tensor.is_contiguous():
                raise ValueError(f"TensorRT input {name} must be contiguous")
            if not self._context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"TensorRT rejected input binding: {name}")

        stream = self._torch.cuda.current_stream()
        if not self._context.execute_async_v3(int(stream.cuda_stream)):
            raise RuntimeError(f"TensorRT execution failed for {self.component}")
        return self._outputs
