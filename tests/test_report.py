import json

import pytest

from src.backends import GenerationConfig
from src.benchmark import BenchmarkConfig
from src.benchmark.report import RuntimeMetadata, build_report, save_report


def make_report():
    return build_report(
        timestamp="2026-08-12T15:30:00Z",
        backend="pytorch-fp16",
        model_name="stabilityai/sdxl-turbo",
        generation=GenerationConfig(prompt="test", seed=42),
        benchmark=BenchmarkConfig(warmup_runs=5, measured_runs=3),
        metadata=RuntimeMetadata(
            gpu_name="Test GPU",
            cuda_version="12.x",
            pytorch_version="2.x",
            python_version="3.x",
        ),
        latencies_ms=[10.0, 20.0, 30.0],
        peak_gpu_memory_bytes=123456,
        peak_device_memory_used_bytes=654321,
        comparison={"reference": "test"},
    )


def test_report_contains_configuration_metadata_and_samples() -> None:
    report = make_report()
    assert report["backend"] == "pytorch-fp16"
    assert report["resolution"] == {"width": 512, "height": 512}
    assert report["seed"] == 42
    assert report["latencies_ms"] == [10.0, 20.0, 30.0]
    assert report["peak_gpu_memory_bytes"] == 123456
    assert report["peak_device_memory_used_bytes"] == 654321
    assert report["comparison"] == {"reference": "test"}
    assert report["summary"]["mean_ms"] == 20.0


def test_report_rejects_sample_count_mismatch() -> None:
    with pytest.raises(ValueError, match="latency count"):
        build_report(
            timestamp="2026-08-12T15:30:00Z",
            backend="test",
            model_name="test",
            generation=GenerationConfig(prompt="test"),
            benchmark=BenchmarkConfig(measured_runs=2),
            metadata=RuntimeMetadata("gpu", "cuda", "torch", "python"),
            latencies_ms=[1.0],
        )


def test_save_report_writes_structured_json(tmp_path) -> None:
    report = make_report()
    path = save_report(report, tmp_path)
    assert path.parent == tmp_path
    assert json.loads(path.read_text(encoding="utf-8")) == report
