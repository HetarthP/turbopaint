"""CLI helpers for component-specific TensorRT engine builds."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.tensorrt.build.engine import BuildRequest, build_engine


def run_component(
    component: str,
    default_onnx: Path,
    default_engine: Path,
    argv: Sequence[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=f"Build the {component} TensorRT engine.")
    parser.add_argument("--onnx", type=Path, default=default_onnx)
    parser.add_argument("--engine", type=Path, default=default_engine)
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    parser.add_argument("--force", action="store_true", help="Ignore a valid cached engine")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose TensorRT logging")
    args = parser.parse_args(argv)
    engine, metadata, cache_hit = build_engine(
        BuildRequest(
            component=component,
            onnx_path=args.onnx,
            engine_path=args.engine,
            workspace_gib=args.workspace_gib,
            force=args.force,
            verbose=args.verbose,
        )
    )
    print(f"TensorRT engine: {engine}")
    print(f"Metadata: {metadata}")
    print(f"Cache hit: {cache_hit}")
    return 0

