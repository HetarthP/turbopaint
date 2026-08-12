"""Trace PyTorch and TensorRT SDXL-Turbo and locate numerical divergence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
from typing import Any, Sequence

from src.backends import GenerationConfig

STAGES = ("noise_pred", "scheduler_latents", "vae_output", "final_image_tensor")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-id", default="stabilityai/sdxl-turbo")
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=1e-1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/validation"))
    parser.add_argument(
        "--unet-engine", type=Path, default=Path("artifacts/tensorrt/unet/model.plan")
    )
    parser.add_argument(
        "--vae-engine",
        type=Path,
        default=Path("artifacts/tensorrt/vae_decoder/model.plan"),
    )
    return parser


def tensor_comparison(reference: Any, actual: Any, *, rtol: float, atol: float) -> dict[str, Any]:
    import numpy as np

    reference_array = reference.detach().float().cpu().numpy()
    actual_array = actual.detach().float().cpu().numpy()
    if reference_array.shape != actual_array.shape:
        raise RuntimeError(
            f"Tensor shape mismatch: PyTorch {reference_array.shape}, TensorRT {actual_array.shape}"
        )
    absolute = np.abs(reference_array - actual_array)
    return {
        "passed": bool(np.allclose(reference_array, actual_array, rtol=rtol, atol=atol)),
        "shape": list(reference_array.shape),
        "pytorch_dtype": str(reference.dtype),
        "tensorrt_dtype": str(actual.dtype),
        "rtol": rtol,
        "atol": atol,
        "mean_absolute_error": float(absolute.mean()),
        "max_absolute_error": float(absolute.max()),
    }


def exact_input_comparison(reference: Any, actual: Any) -> dict[str, Any]:
    import torch

    return {
        "identical": bool(torch.equal(reference.cpu(), actual.cpu())),
        "shape": list(reference.shape),
        "pytorch_dtype": str(reference.dtype),
        "tensorrt_dtype": str(actual.dtype),
    }


def scheduler_transition(
    *,
    scheduler_class: Any,
    scheduler_config: dict[str, Any],
    noise_pred: Any,
    latents: Any,
    generator_state: Any,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run an isolated one-step transition from a fresh scheduler instance."""
    import torch

    scheduler = scheduler_class.from_config(scheduler_config)
    scheduler.set_timesteps(1, device=device)
    timestep = scheduler.timesteps[0]
    generator = torch.Generator(device=device)
    generator.set_state(generator_state.cpu())
    model_output = noise_pred.to(device=device, dtype=latents.dtype).contiguous()
    sample = latents.to(device=device).contiguous()
    before = {
        "timestep": float(timestep.detach().cpu()),
        "timestep_dtype": str(timestep.dtype),
        "noise_pred_dtype": str(model_output.dtype),
        "latents_dtype": str(sample.dtype),
        "step_index": scheduler.step_index,
        "sigmas": [float(value) for value in scheduler.sigmas.detach().cpu()],
    }
    previous = scheduler.step(
        model_output,
        timestep,
        sample,
        generator=generator,
        return_dict=False,
    )[0]
    before["step_index_after"] = scheduler.step_index
    return previous.detach().cpu(), before


def pytorch_trace(backend: Any, config: GenerationConfig) -> dict[str, Any]:
    """Reproduce the relevant Diffusers __call__ stages with trace tensors."""
    torch = backend._torch
    pipe = backend.pipeline
    generator = torch.Generator(device=backend.device).manual_seed(config.seed)
    with torch.inference_mode():
        prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=config.prompt,
            device=backend.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        prompt_embeds = prompt_embeds.to(torch.float16).contiguous()
        pooled_prompt_embeds = pooled_prompt_embeds.to(torch.float16).contiguous()
        pipe.scheduler.set_timesteps(1, device=backend.device)
        latents = pipe.prepare_latents(
            1,
            4,
            512,
            512,
            prompt_embeds.dtype,
            backend.device,
            generator,
            None,
        )
        trace = {
            "prompt_embeds": prompt_embeds.detach().cpu(),
            "pooled_prompt_embeds": pooled_prompt_embeds.detach().cpu(),
            "initial_latents": latents.detach().cpu(),
            "generator_state": generator.get_state(),
            "timestep": pipe.scheduler.timesteps.detach().cpu(),
            "scheduler_config": dict(pipe.scheduler.config),
            "scheduler_sigmas": pipe.scheduler.sigmas.detach().cpu(),
            "scheduler_step_index_before": pipe.scheduler.step_index,
        }
        timestep = pipe.scheduler.timesteps[0]
        timestep_input = timestep.reshape(1).to(torch.float32).contiguous()
        time_ids = torch.tensor(
            [[512, 512, 0, 0, 512, 512]],
            dtype=torch.float16,
            device=backend.device,
        )
        trace["time_ids"] = time_ids.detach().cpu()
        latent_input = pipe.scheduler.scale_model_input(latents, timestep).contiguous()
        trace["unet_sample"] = latent_input.detach().cpu()
        noise_pred = pipe.unet(
            sample=latent_input,
            timestep=timestep_input,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "text_embeds": pooled_prompt_embeds,
                "time_ids": time_ids,
            },
            return_dict=False,
        )[0]
        trace["noise_pred"] = noise_pred.detach().cpu()
        latents = pipe.scheduler.step(
            noise_pred, timestep, latents, generator=generator, return_dict=False
        )[0]
        trace["scheduler_latents"] = latents.detach().cpu()

        needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
        if needs_upcasting:
            pipe.upcast_vae()
            vae_dtype = next(iter(pipe.vae.post_quant_conv.parameters())).dtype
            latents = latents.to(vae_dtype)
        if getattr(pipe.vae.config, "latents_mean", None) is not None:
            raise RuntimeError("TensorRT VAE export does not support VAE latents_mean/std")
        decoded = pipe.vae.decode(
            latents / pipe.vae.config.scaling_factor, return_dict=False
        )[0]
        trace["vae_output"] = decoded.detach().cpu()
        if getattr(pipe, "watermark", None) is not None:
            decoded = pipe.watermark.apply_watermark(decoded)
        final_tensor = pipe.image_processor.postprocess(decoded, output_type="pt")[0]
        trace["final_image_tensor"] = final_tensor.detach().cpu()
        trace["image"] = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        return trace


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = GenerationConfig(prompt=args.prompt, seed=args.seed)
    import torch
    from src.baseline.pipeline import SDXLTurboBaseline

    baseline = SDXLTurboBaseline(args.model_id)
    reference = pytorch_trace(baseline, config)
    torch.cuda.synchronize()
    del baseline
    gc.collect()
    torch.cuda.empty_cache()

    from src.tensorrt.pipeline import SDXLTurboTensorRT

    backend = SDXLTurboTensorRT(
        args.model_id, unet_engine=args.unet_engine, vae_engine=args.vae_engine
    )
    actual = backend.generate_trace(
        config,
        prompt_embeds=reference["prompt_embeds"],
        pooled_prompt_embeds=reference["pooled_prompt_embeds"],
        initial_latents=reference["initial_latents"],
        generator_state=reference["generator_state"],
    )
    torch.cuda.synchronize()

    inputs = {
        name: exact_input_comparison(reference[name], actual[name])
        for name in (
            "initial_latents",
            "timestep",
            "unet_sample",
            "prompt_embeds",
            "pooled_prompt_embeds",
            "time_ids",
        )
    }
    if not all(item["identical"] for item in inputs.values()):
        raise RuntimeError(f"PyTorch/TensorRT trace inputs are not identical: {inputs}")
    stages = {
        name: tensor_comparison(
            reference[name], actual[name], rtol=args.rtol, atol=args.atol
        )
        for name in STAGES
    }
    first_failure = next(
        (name for name in STAGES if not stages[name]["passed"]), None
    )

    scheduler_class = backend.pipeline.scheduler.__class__
    pytorch_noise_on_pytorch_scheduler, pytorch_scheduler_state = scheduler_transition(
        scheduler_class=scheduler_class,
        scheduler_config=reference["scheduler_config"],
        noise_pred=reference["noise_pred"],
        latents=reference["initial_latents"],
        generator_state=reference["generator_state"],
        device=backend.device,
    )
    pytorch_noise_on_tensorrt_scheduler, tensorrt_scheduler_state = scheduler_transition(
        scheduler_class=scheduler_class,
        scheduler_config=actual["scheduler_config"],
        noise_pred=reference["noise_pred"],
        latents=reference["initial_latents"],
        generator_state=reference["generator_state"],
        device=backend.device,
    )
    tensorrt_noise_transition, _ = scheduler_transition(
        scheduler_class=scheduler_class,
        scheduler_config=actual["scheduler_config"],
        noise_pred=actual["noise_pred"],
        latents=reference["initial_latents"],
        generator_state=reference["generator_state"],
        device=backend.device,
    )
    scheduler_audit = {
        "configs_identical": reference["scheduler_config"]
        == actual["scheduler_config"],
        "sigmas_identical": bool(
            torch.equal(
                reference["scheduler_sigmas"].cpu(),
                actual["scheduler_sigmas"].cpu(),
            )
        ),
        "pytorch_scheduler_state": pytorch_scheduler_state,
        "tensorrt_scheduler_state": tensorrt_scheduler_state,
        "same_noise_cross_scheduler": tensor_comparison(
            pytorch_noise_on_pytorch_scheduler,
            pytorch_noise_on_tensorrt_scheduler,
            rtol=0.0,
            atol=0.0,
        ),
        "tensorrt_noise_recomputed_transition": tensor_comparison(
            actual["scheduler_latents"],
            tensorrt_noise_transition,
            rtol=0.0,
            atol=0.0,
        ),
    }
    noise_mean = stages["noise_pred"]["mean_absolute_error"]
    noise_max = stages["noise_pred"]["max_absolute_error"]
    latent_mean = stages["scheduler_latents"]["mean_absolute_error"]
    latent_max = stages["scheduler_latents"]["max_absolute_error"]
    scheduler_audit["observed_error_amplification"] = {
        "mean_ratio": latent_mean / noise_mean if noise_mean else None,
        "max_ratio": latent_max / noise_max if noise_max else None,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = args.output_dir / "pytorch.png"
    actual_path = args.output_dir / "tensorrt.png"
    report_path = args.output_dir / "comparison.json"
    reference["image"].save(reference_path)
    actual["image"].save(actual_path)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_id,
        "prompt": args.prompt,
        "seed": args.seed,
        "resolution": [512, 512],
        "inputs": inputs,
        "stages": stages,
        "scheduler_audit": scheduler_audit,
        "first_failed_stage": first_failure,
        "pytorch_image": str(reference_path),
        "tensorrt_image": str(actual_path),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if first_failure is not None:
        raise RuntimeError(
            f"First meaningful divergence: {first_failure}; see {report_path}"
        )
    print(f"All TensorRT trace stages passed: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
