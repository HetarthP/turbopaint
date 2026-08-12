import pytest

from src.benchmark import BenchmarkConfig, benchmark_cuda, calculate_statistics
from src.benchmark.statistics import percentile


def test_benchmark_defaults() -> None:
    config = BenchmarkConfig()
    assert config.warmup_runs == 5
    assert config.measured_runs == 20


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"warmup_runs": -1}, "warmup_runs"),
        ({"measured_runs": 0}, "measured_runs"),
    ],
)
def test_benchmark_config_validation(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        BenchmarkConfig(**kwargs)


def test_statistics_for_odd_sample_count() -> None:
    stats = calculate_statistics([5.0, 1.0, 4.0, 2.0, 3.0])
    assert stats.mean_ms == pytest.approx(3.0)
    assert stats.median_ms == pytest.approx(3.0)
    assert stats.p95_ms == pytest.approx(4.8)
    assert stats.min_ms == pytest.approx(1.0)
    assert stats.max_ms == pytest.approx(5.0)
    assert stats.images_per_second == pytest.approx(1000.0 / 3.0)


def test_statistics_for_even_sample_count() -> None:
    stats = calculate_statistics([40.0, 10.0, 30.0, 20.0])
    assert stats.mean_ms == pytest.approx(25.0)
    assert stats.median_ms == pytest.approx(25.0)
    assert stats.p95_ms == pytest.approx(38.5)
    assert stats.min_ms == pytest.approx(10.0)
    assert stats.max_ms == pytest.approx(40.0)
    assert stats.images_per_second == pytest.approx(40.0)


def test_single_sample_statistics() -> None:
    stats = calculate_statistics([12.5])
    assert stats.mean_ms == stats.median_ms == stats.p95_ms == 12.5
    assert stats.min_ms == stats.max_ms == 12.5
    assert stats.images_per_second == pytest.approx(80.0)


@pytest.mark.parametrize("samples", [[], [0.0], [-1.0], [float("inf")]])
def test_invalid_statistics_samples(samples) -> None:
    with pytest.raises(ValueError):
        calculate_statistics(samples)


def test_percentile_bounds() -> None:
    assert percentile([1.0, 2.0, 3.0], 0) == 1.0
    assert percentile([1.0, 2.0, 3.0], 100) == 3.0
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_cuda_benchmark_fails_without_cuda(monkeypatch) -> None:
    class UnavailableCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = UnavailableCuda()

    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorch())
    with pytest.raises(RuntimeError, match="CUDA is required"):
        benchmark_cuda(lambda: None, config=BenchmarkConfig())
