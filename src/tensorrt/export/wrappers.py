"""Small adapters that expose Diffusers modules as tensor-only ONNX graphs."""

from __future__ import annotations

from typing import Any


def make_unet_wrapper(torch_module: Any, unet: Any) -> Any:
    class UNetWrapper(torch_module.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.unet = unet

        def forward(
            self,
            sample: Any,
            timestep: Any,
            encoder_hidden_states: Any,
            text_embeds: Any,
            time_ids: Any,
        ) -> Any:
            return self.unet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                return_dict=False,
            )[0]

    return UNetWrapper().eval()


def make_vae_decoder_wrapper(torch_module: Any, vae: Any) -> Any:
    class VAEDecoderWrapper(torch_module.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vae = vae

        def forward(self, latents: Any) -> Any:
            # Scaling is part of the graph so callers pass scheduler latents.
            scaled = latents / self.vae.config.scaling_factor
            return self.vae.decode(scaled, return_dict=False)[0]

    return VAEDecoderWrapper().eval()

