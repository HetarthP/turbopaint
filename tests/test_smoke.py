from pathlib import Path

from PIL import Image
import pytest

from src.baseline.smoke_test import build_parser, validate_saved_image


def test_smoke_cli_defaults() -> None:
    args = build_parser().parse_args(["a test prompt"])
    assert args.model_id == "stabilityai/sdxl-turbo"
    assert args.seed == 0


def test_saved_image_validation(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (512, 512)).save(path)
    validate_saved_image(path, (512, 512))


def test_saved_image_validation_rejects_wrong_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (256, 256)).save(path)
    with pytest.raises(RuntimeError, match="Expected output dimensions"):
        validate_saved_image(path, (512, 512))
