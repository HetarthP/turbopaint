"""Dedicated command for the full PyTorch FP16 baseline benchmark."""

from src.baseline.generate import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())

