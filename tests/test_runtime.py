from src.runtime.cuda import inspect_cuda_environment, require_cuda

import pytest


class Version:
    cuda = None


class UnavailableCuda:
    @staticmethod
    def is_available():
        return False


class CpuTorch:
    __version__ = "test-torch"
    version = Version()
    cuda = UnavailableCuda()


def test_environment_inspection_without_cuda() -> None:
    environment = inspect_cuda_environment(CpuTorch())
    assert environment.pytorch_version == "test-torch"
    assert environment.cuda_available is False
    assert environment.gpu_count == 0
    assert environment.gpu_name is None
    assert environment.total_gpu_memory_bytes is None
    assert environment.memory_allocated_bytes is None


def test_require_cuda_has_actionable_error() -> None:
    with pytest.raises(RuntimeError, match="pytorch.org/get-started/locally"):
        require_cuda(CpuTorch())

