"""SDXL-Turbo inference backend implemented with PyTorch and Diffusers."""

from __future__ import annotations

from typing import Any

from src.backends import GenerationConfig, InferenceBackend
from src.runtime import require_cuda


class SDXLTurboBaseline(InferenceBackend):
    """Own and execute an FP16 SDXL-Turbo Diffusers pipeline on CUDA."""

    def __init__(self, model_id: str = "stabilityai/sdxl-turbo") -> None:
        # Keep heavyweight imports local so CLI help and unit tests work before
        # the optional GPU environment has been provisioned.
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required. Install its CUDA-enabled build using "
                "the NVIDIA Linux setup in README.md."
            ) from exc

        # Validate CUDA before importing Diffusers or downloading/loading a model.
        require_cuda(torch)
        try:
            from diffusers import AutoPipelineForText2Image
        except ImportError as exc:
            raise RuntimeError(
                "Diffusers is required. Install requirements.txt after PyTorch."
            ) from exc

        self._torch = torch
        self._model_name = model_id
        self.device = torch.device("cuda")
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)

    @property
    def name(self) -> str:
        return "pytorch-fp16"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, config: GenerationConfig) -> Any:
        """Generate and return one PIL image."""
        # Reset a device-local RNG for every generation. This holds model inputs
        # constant across warm-up, measured, and future backend runs.
        generator = self._torch.Generator(device=self.device).manual_seed(config.seed)
        result = self.pipeline(
            prompt=config.prompt,
            height=config.height,
            width=config.width,
            num_inference_steps=config.inference_steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
        )
        return result.images[0]
