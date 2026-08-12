"""Tensor contracts shared by exporters, validators, and documentation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorContract:
    name: str
    shape: tuple[int, ...]
    dtype: str


COMPONENT_CONTRACTS: dict[str, dict[str, tuple[TensorContract, ...]]] = {
    "unet": {
        "inputs": (
            TensorContract("sample", (1, 4, 64, 64), "float16"),
            TensorContract("timestep", (1,), "float32"),
            TensorContract("encoder_hidden_states", (1, 77, 2048), "float16"),
            TensorContract("text_embeds", (1, 1280), "float16"),
            TensorContract("time_ids", (1, 6), "float16"),
        ),
        "outputs": (TensorContract("noise_pred", (1, 4, 64, 64), "float16"),),
    },
    "vae_decoder": {
        # The stock SDXL VAE declares force_upcast=True; matching Diffusers means
        # decoding in float32 rather than introducing known FP16 overflow risk.
        "inputs": (TensorContract("latents", (1, 4, 64, 64), "float32"),),
        "outputs": (TensorContract("images", (1, 3, 512, 512), "float32"),),
    },
}


def contract_as_dict(component: str) -> dict[str, list[dict[str, object]]]:
    if component not in COMPONENT_CONTRACTS:
        raise ValueError(f"Unknown component: {component}")
    return {
        group: [
            {"name": tensor.name, "shape": list(tensor.shape), "dtype": tensor.dtype}
            for tensor in tensors
        ]
        for group, tensors in COMPONENT_CONTRACTS[component].items()
    }
