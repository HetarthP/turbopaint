"""GPU latency measurement based on CUDA events."""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

from dataclasses import dataclass

from src.benchmark.config import BenchmarkConfig
from src.benchmark.statistics import LatencyStatistics, calculate_statistics

T = TypeVar("T")


@dataclass(frozen=True)
class BenchmarkResult(Generic[T]):
    """Aggregated benchmark metrics and the final function result."""

    statistics: LatencyStatistics
    latencies_ms: tuple[float, ...]
    peak_gpu_memory_bytes: int
    last_output: T


def benchmark_cuda(
    operation: Callable[[], T],
    *,
    config: BenchmarkConfig,
) -> BenchmarkResult[T]:
    """Benchmark GPU work with per-run CUDA events.

    Model loading must happen before this function. Warm-ups are deliberately
    unmeasured. One final device synchronization waits for every recorded end
    event before elapsed times are read.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "CUDA benchmarking requires a CUDA-enabled PyTorch installation"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CUDA-event benchmarking")

    last_output: T | None = None
    with torch.inference_mode():
        for _ in range(config.warmup_runs):
            last_output = operation()

        # Ensure warm-up work cannot overlap the measured region.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        events: list[tuple[object, object]] = []
        for _ in range(config.measured_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            last_output = operation()
            end.record()
            events.append((start, end))

        torch.cuda.synchronize()
        peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated())

    samples = tuple(float(start.elapsed_time(end)) for start, end in events)
    assert last_output is not None
    return BenchmarkResult(
        statistics=calculate_statistics(samples),
        latencies_ms=samples,
        peak_gpu_memory_bytes=peak_gpu_memory_bytes,
        last_output=last_output,
    )
