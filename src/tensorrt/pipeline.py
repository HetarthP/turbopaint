"""Hybrid SDXL-Turbo backend with TensorRT UNet and VAE decoder."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from src.backends import GenerationConfig, InferenceBackend
from src.runtime import require_cuda
from src.tensorrt.runtime import TensorRTEngine


class SDXLTurboTensorRT(InferenceBackend):
    """Run text/scheduler logic in Diffusers and image networks in TensorRT."""

    def __init__(
        self,
        model_id: str = "stabilityai/sdxl-turbo",
        unet_engine: Path = Path("artifacts/tensorrt/unet/model.plan"),
        vae_engine: Path = Path("artifacts/tensorrt/vae_decoder/model.plan"),
    ) -> None:
        try:
            import torch
            import tensorrt as trt
            from diffusers import AutoPipelineForText2Image
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT inference requires CUDA PyTorch, Diffusers, and TensorRT"
            ) from exc
        require_cuda(torch)
        self._torch = torch
        self._model_name = model_id
        self.device = torch.device("cuda")

        # Load Diffusers once to retain its tested tokenizers, CLIP encoders,
        # scheduler, latent preparation, and image processor. Remove the two
        # networks replaced by TensorRT before deserializing their engines.
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        ).to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)
        self.pipeline.unet = None
        self.pipeline.vae = None
        gc.collect()
        torch.cuda.empty_cache()

        self.unet = TensorRTEngine("unet", unet_engine, torch, trt)
        self.vae_decoder = TensorRTEngine("vae_decoder", vae_engine, torch, trt)
        self._vae_input = torch.empty(
            (1, 4, 64, 64), dtype=torch.float32, device=self.device
        )
        self._time_ids = torch.tensor(
            [[512, 512, 0, 0, 512, 512]], dtype=torch.float16, device=self.device
        )

    @property
    def name(self) -> str:
        return "tensorrt-unet-fp16-vae-fp32"

    @property
    def model_name(self) -> str:
        return self._model_name

    def _validate_config(self, config: GenerationConfig) -> None:
        if (config.width, config.height) != (512, 512):
            raise ValueError("TensorRT engines support only 512x512 generation")
        if config.inference_steps != 1:
            raise ValueError("TensorRT baseline supports exactly 1 inference step")
        if config.guidance_scale != 0.0:
            raise ValueError("SDXL-Turbo TensorRT baseline requires guidance_scale=0.0")

    def generate(self, config: GenerationConfig) -> Any:
        self._validate_config(config)
        torch = self._torch
        pipe = self.pipeline
        generator = torch.Generator(device=self.device).manual_seed(config.seed)

        with torch.inference_mode():
            prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=config.prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            prompt_embeds = prompt_embeds.to(dtype=torch.float16).contiguous()
            pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=torch.float16).contiguous()

            pipe.scheduler.set_timesteps(config.inference_steps, device=self.device)
            latents = pipe.prepare_latents(
                1,
                4,
                config.height,
                config.width,
                prompt_embeds.dtype,
                self.device,
                generator,
                None,
            )
            for timestep in pipe.scheduler.timesteps:
                latent_input = pipe.scheduler.scale_model_input(latents, timestep).contiguous()
                timestep_input = timestep.reshape(1).to(dtype=torch.float32).contiguous()
                noise_pred = self.unet.execute(
                    {
                        "sample": latent_input,
                        "timestep": timestep_input,
                        "encoder_hidden_states": prompt_embeds,
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": self._time_ids,
                    }
                )["noise_pred"]
                latents = pipe.scheduler.step(
                    noise_pred, timestep, latents, return_dict=False
                )[0]

            self._vae_input.copy_(latents)
            decoded = self.vae_decoder.execute({"latents": self._vae_input})["images"]
            if getattr(pipe, "watermark", None) is not None:
                decoded = pipe.watermark.apply_watermark(decoded)
            # Diffusers converts the CUDA tensor to PIL here; this also ensures
            # TensorRT decode completion before the persistent buffer is reused.
            return pipe.image_processor.postprocess(decoded, output_type="pil")[0]
