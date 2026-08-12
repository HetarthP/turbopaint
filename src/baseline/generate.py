"""Command-line entry point for SDXL-Turbo generation and benchmarking."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from src.backends import GenerationConfig
from src.baseline.pipeline import SDXLTurboBaseline
from src.benchmark import BenchmarkConfig, benchmark_cuda
from src.benchmark.report import (
    build_report,
    collect_torch_metadata,
    current_timestamp,
    save_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark 512x512 SDXL-Turbo inference with CUDA events."
    )
    parser.add_argument("prompt", help="Text prompt used to generate the image")
    parser.add_argument("--model-id", default="stabilityai/sdxl-turbo")
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--benchmark-runs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path (default: a timestamped file in outputs/)",
    )
    return parser


def default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("outputs") / f"sdxl_turbo_{timestamp}.png"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generation = GenerationConfig(
        prompt=args.prompt,
        width=512,
        height=512,
        inference_steps=args.steps,
        guidance_scale=0.0,
        seed=args.seed,
    )
    benchmark = BenchmarkConfig(
        warmup_runs=args.warmup_runs,
        measured_runs=args.benchmark_runs,
    )
    backend = SDXLTurboBaseline(model_id=args.model_id)

    result = benchmark_cuda(
        lambda: backend.generate(generation),
        config=benchmark,
    )

    output_path = args.output or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.last_output.save(output_path)

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
    )
    report_path = save_report(report)

    print(f"Saved image: {output_path}")
    print(f"Saved benchmark: {report_path}")
    stats = result.statistics
    print(f"Mean GPU latency: {stats.mean_ms:.2f} ms")
    print(f"Median GPU latency: {stats.median_ms:.2f} ms")
    print(f"P95 GPU latency: {stats.p95_ms:.2f} ms")
    print(f"Minimum GPU latency: {stats.min_ms:.2f} ms")
    print(f"Maximum GPU latency: {stats.max_ms:.2f} ms")
    print(f"Throughput: {stats.images_per_second:.2f} images/second")
    print(
        "Peak allocated GPU memory: "
        f"{result.peak_gpu_memory_bytes / (1024 ** 3):.2f} GiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
