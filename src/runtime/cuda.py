"""CUDA environment inspection with no import-time PyTorch dependency."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import platform
from typing import Any


CUDA_REQUIRED_MESSAGE = (
    "TurboPaint requires an NVIDIA GPU and a CUDA-enabled PyTorch build. "
    "Install PyTorch using the CUDA command from "
    "https://pytorch.org/get-started/locally/ and verify the NVIDIA driver."
)


@dataclass(frozen=True)
class CudaEnvironment:
    python_version: str
    pytorch_version: str
    cuda_available: bool
    cuda_runtime_version: str | None
    gpu_name: str | None
    gpu_count: int
    total_gpu_memory_bytes: int | None
    memory_allocated_bytes: int | None
    memory_reserved_bytes: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def require_cuda(torch_module: Any) -> None:
    """Fail before model loading when CUDA is unavailable."""
    if not torch_module.cuda.is_available():
        raise RuntimeError(CUDA_REQUIRED_MESSAGE)


def inspect_cuda_environment(torch_module: Any) -> CudaEnvironment:
    available = bool(torch_module.cuda.is_available())
    count = int(torch_module.cuda.device_count()) if available else 0
    if not available:
        return CudaEnvironment(
            python_version=platform.python_version(),
            pytorch_version=str(torch_module.__version__),
            cuda_available=False,
            cuda_runtime_version=(
                str(torch_module.version.cuda) if torch_module.version.cuda else None
            ),
            gpu_name=None,
            gpu_count=0,
            total_gpu_memory_bytes=None,
            memory_allocated_bytes=None,
            memory_reserved_bytes=None,
        )

    device = torch_module.cuda.current_device()
    properties = torch_module.cuda.get_device_properties(device)
    return CudaEnvironment(
        python_version=platform.python_version(),
        pytorch_version=str(torch_module.__version__),
        cuda_available=True,
        cuda_runtime_version=str(torch_module.version.cuda),
        gpu_name=str(torch_module.cuda.get_device_name(device)),
        gpu_count=count,
        total_gpu_memory_bytes=int(properties.total_memory),
        memory_allocated_bytes=int(torch_module.cuda.memory_allocated(device)),
        memory_reserved_bytes=int(torch_module.cuda.memory_reserved(device)),
    )

