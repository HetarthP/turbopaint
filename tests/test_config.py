import pytest

from src.backends import GenerationConfig


def test_generation_defaults() -> None:
    config = GenerationConfig(prompt="a robot painting")
    assert (config.width, config.height) == (512, 512)
    assert config.inference_steps == 1
    assert config.guidance_scale == 0.0
    assert config.seed == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": " "},
        {"prompt": "x", "width": 0},
        {"prompt": "x", "height": -8},
        {"prompt": "x", "width": 510},
        {"prompt": "x", "inference_steps": 0},
        {"prompt": "x", "guidance_scale": -1.0},
        {"prompt": "x", "seed": -1},
    ],
)
def test_generation_config_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        GenerationConfig(**kwargs)

