"""Structured benchmark report construction and persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

from src.backends import GenerationConfig
from src.benchmark.config import BenchmarkConfig
from src.benchmark.statistics import calculate_statistics


@dataclass(frozen=True)
class RuntimeMetadata:
    gpu_name: str
    cuda_version: str
    pytorch_version: str
    python_version: str


def build_report(
    *,
    timestamp: str,
    backend: str,
    model_name: str,
    generation: GenerationConfig,
    benchmark: BenchmarkConfig,
    metadata: RuntimeMetadata,
    latencies_ms: Sequence[float],
    peak_gpu_memory_bytes: int | None = None,
) -> dict[str, Any]:
    samples = tuple(float(value) for value in latencies_ms)
    if len(samples) != benchmark.measured_runs:
        raise ValueError("latency count must equal configured measured_runs")
    summary = calculate_statistics(samples)
    return {
        "timestamp": timestamp,
        "backend": backend,
        **asdict(metadata),
        "model_name": model_name,
        "prompt": generation.prompt,
        "resolution": {"width": generation.width, "height": generation.height},
        "inference_steps": generation.inference_steps,
        "guidance_scale": generation.guidance_scale,
        "seed": generation.seed,
        "warmup_runs": benchmark.warmup_runs,
        "measured_runs": benchmark.measured_runs,
        "latencies_ms": list(samples),
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "summary": summary.to_dict(),
    }


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def collect_torch_metadata(torch_module: Any) -> RuntimeMetadata:
    """Read genuine runtime metadata after CUDA availability is established."""
    return RuntimeMetadata(
        gpu_name=torch_module.cuda.get_device_name(torch_module.cuda.current_device()),
        cuda_version=str(torch_module.version.cuda),
        pytorch_version=str(torch_module.__version__),
        python_version=platform.python_version(),
    )


def save_report(report: Mapping[str, Any], directory: Path = Path("benchmarks")) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    compact_timestamp = str(report["timestamp"]).replace(":", "").replace("-", "")
    compact_timestamp = compact_timestamp.replace("T", "_").replace("Z", "")
    backend = str(report["backend"]).replace("/", "-")
    path = directory / f"{compact_timestamp}_{backend}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return path
