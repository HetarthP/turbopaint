"""Pure, unit-testable latency statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
from typing import Sequence


@dataclass(frozen=True)
class LatencyStatistics:
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    images_per_second: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Return a linearly interpolated percentile, matching NumPy's default."""
    if not values:
        raise ValueError("at least one latency measurement is required")
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile_value / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def calculate_statistics(latencies_ms: Sequence[float]) -> LatencyStatistics:
    samples = tuple(float(value) for value in latencies_ms)
    if not samples:
        raise ValueError("at least one latency measurement is required")
    if any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("latency measurements must be finite and positive")

    mean_ms = statistics.fmean(samples)
    return LatencyStatistics(
        mean_ms=mean_ms,
        median_ms=statistics.median(samples),
        p95_ms=percentile(samples, 95),
        min_ms=min(samples),
        max_ms=max(samples),
        images_per_second=1000.0 / mean_ms,
    )

