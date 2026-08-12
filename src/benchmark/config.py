"""Benchmark configuration independent of CUDA availability."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup_runs: int = 5
    measured_runs: int = 20

    def __post_init__(self) -> None:
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")
        if self.measured_runs <= 0:
            raise ValueError("measured_runs must be positive")

