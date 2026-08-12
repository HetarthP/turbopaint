"""Benchmark the hybrid TensorRT SDXL-Turbo backend with CUDA events."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.backends import GenerationConfig
from src.benchmark import BenchmarkConfig, benchmark_cuda
from src.benchmark.report import (
    build_report,
    collect_torch_metadata,
    current_timestamp,
    save_report,
)
from src.tensorrt.pipeline import SDXLTurboTensorRT

PYTORCH_T4_REFERENCE = {
    "hardware": "NVIDIA Tesla T4",
    "backend": "pytorch-fp16",
    "mean_ms": 610.60,
    "p95_ms": 654.38,
    "images_per_second": 1.64,
    "peak_allocated_memory_bytes": int(7.48 * 1024**3),
}


def baseline_comparison(mean_ms: float, p95_ms: float, images_per_second: float) -> dict[str, Any]:
    return {
        "reference": PYTORCH_T4_REFERENCE,
        "measured_to_reference": {
            "mean_latency_ratio": mean_ms / PYTORCH_T4_REFERENCE["mean_ms"],
            "p95_latency_ratio": p95_ms / PYTORCH_T4_REFERENCE["p95_ms"],
            "throughput_ratio": images_per_second
            / PYTORCH_T4_REFERENCE["images_per_second"],
        },
        "note": "Reference and current result are directly comparable only on the same T4 environment.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-id", default="stabilityai/sdxl-turbo")
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--benchmark-runs", type=int, default=20)
    parser.add_argument(
        "--unet-engine", type=Path, default=Path("artifacts/tensorrt/unet/model.plan")
    )
    parser.add_argument(
        "--vae-engine",
        type=Path,
        default=Path("artifacts/tensorrt/vae_decoder/model.plan"),
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generation = GenerationConfig(prompt=args.prompt, seed=args.seed)
    benchmark = BenchmarkConfig(args.warmup_runs, args.benchmark_runs)
    backend = SDXLTurboTensorRT(
        model_id=args.model_id,
        unet_engine=args.unet_engine,
        vae_engine=args.vae_engine,
    )
    result = benchmark_cuda(lambda: backend.generate(generation), config=benchmark)
    stats = result.statistics
    comparison = baseline_comparison(stats.mean_ms, stats.p95_ms, stats.images_per_second)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("outputs") / f"tensorrt_{timestamp}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.last_output.save(output)

    import torch

    report = build_report(
        timestamp=current_timestamp(),
        backend=backend.name,
        model_name=backend.model_name,
        generation=generation,
        benchmark=benchmark,
        metadata=collect_torch_metadata(torch),
        latencies_ms=result.latencies_ms,
        peak_gpu_memory_bytes=result.peak_gpu_memory_bytes,
        peak_device_memory_used_bytes=result.peak_device_memory_used_bytes,
        comparison=comparison,
    )
    import tensorrt as trt

    report["tensorrt_version"] = str(trt.__version__)
    report["engines"] = {
        "unet": str(args.unet_engine),
        "vae_decoder": str(args.vae_engine),
    }
    report_path = save_report(report)
    print(f"Saved image: {output}")
    print(f"Saved benchmark: {report_path}")
    print(f"Mean GPU latency: {stats.mean_ms:.2f} ms")
    print(f"Median GPU latency: {stats.median_ms:.2f} ms")
    print(f"P95 GPU latency: {stats.p95_ms:.2f} ms")
    print(f"Minimum GPU latency: {stats.min_ms:.2f} ms")
    print(f"Maximum GPU latency: {stats.max_ms:.2f} ms")
    print(f"Throughput: {stats.images_per_second:.2f} images/second")
    print(f"Peak PyTorch allocated memory: {result.peak_gpu_memory_bytes / 1024**3:.2f} GiB")
    print(f"Peak device memory used: {result.peak_device_memory_used_bytes / 1024**3:.2f} GiB")
    print("PyTorch T4 reference mean: 610.60 ms; p95: 654.38 ms; throughput: 1.64 img/s")
    print("No speedup claim is made; inspect the measured JSON comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
