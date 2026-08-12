from pathlib import Path

import pytest

from src.baseline.generate import build_parser, default_output_path


def test_parser_accepts_prompt_and_defaults() -> None:
    args = build_parser().parse_args(["a fast painting robot"])

    assert args.prompt == "a fast painting robot"
    assert args.model_id == "stabilityai/sdxl-turbo"
    assert args.warmup_runs == 5
    assert args.benchmark_runs == 20
    assert args.steps == 1


def test_parser_requires_prompt() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_default_output_is_png_in_outputs() -> None:
    path = default_output_path()

    assert path.parent == Path("outputs")
    assert path.suffix == ".png"
