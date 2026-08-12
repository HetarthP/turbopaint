from pathlib import Path

import pytest

from src.backends import GenerationConfig
from src.tensorrt.benchmark import baseline_comparison, build_parser as benchmark_parser
from src.tensorrt.pipeline import SDXLTurboTensorRT, step_scheduler
from src.tensorrt.smoke_test import build_parser as smoke_parser
from src.tensorrt.validate import build_parser as validation_parser


def backend_without_loading() -> SDXLTurboTensorRT:
    return object.__new__(SDXLTurboTensorRT)


def test_tensorrt_fixed_generation_contract() -> None:
    backend = backend_without_loading()
    backend._validate_config(GenerationConfig(prompt="test"))
    with pytest.raises(ValueError, match="512x512"):
        backend._validate_config(GenerationConfig(prompt="test", width=768))
    with pytest.raises(ValueError, match="exactly 1"):
        backend._validate_config(GenerationConfig(prompt="test", inference_steps=2))
    with pytest.raises(ValueError, match="guidance_scale"):
        backend._validate_config(GenerationConfig(prompt="test", guidance_scale=1.0))


def test_tensorrt_command_defaults_are_comparable() -> None:
    smoke = smoke_parser().parse_args(["test"])
    benchmark = benchmark_parser().parse_args(["test"])
    validation = validation_parser().parse_args(["test"])
    assert smoke.seed == benchmark.seed == validation.seed == 0
    assert benchmark.warmup_runs == 5
    assert benchmark.benchmark_runs == 20
    assert smoke.unet_engine == Path("artifacts/tensorrt/unet/model.plan")
    assert smoke.vae_engine == Path("artifacts/tensorrt/vae_decoder/model.plan")


def test_reference_comparison_is_explicit_not_a_claim() -> None:
    comparison = baseline_comparison(610.60, 654.38, 1.64)
    ratios = comparison["measured_to_reference"]
    assert ratios["mean_latency_ratio"] == pytest.approx(1.0)
    assert ratios["p95_latency_ratio"] == pytest.approx(1.0)
    assert ratios["throughput_ratio"] == pytest.approx(1.0)
    assert "same T4" in comparison["note"]


def test_trace_stage_order_finds_first_divergence() -> None:
    from src.tensorrt.validate import STAGES

    assert STAGES == (
        "noise_pred",
        "scheduler_latents",
        "vae_output",
        "final_image_tensor",
    )


def test_scheduler_receives_fixed_generator() -> None:
    sentinel_generator = object()

    class Scheduler:
        def step(self, noise, timestep, latents, **kwargs):
            assert kwargs == {
                "generator": sentinel_generator,
                "return_dict": False,
            }
            return ("updated-latents",)

    assert (
        step_scheduler(Scheduler(), "noise", "timestep", "latents", sentinel_generator)
        == "updated-latents"
    )
