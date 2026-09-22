"""EraserDiT erase – denoising stage.

Port of the ``LTXVideoToVideoPipeline`` denoising loop.  The baseline expands the
CFG batch to two and slices ``[0:1]`` / ``[1:]`` before the forward; because both
halves are identical copies that is exactly two batch-1 forwards, which is what
this stage issues directly.
"""

from __future__ import annotations

import time

import torch

from cache.eraserdit import EraserDiTCacheWindow
from models.adapters.eraserdit.cfg import EraserDiTCFGWindow, validate_cfg_parallel
from models.adapters.eraserdit.mesh import EraserDiTMeshWindow, resolve_mesh

from config.server_args import ServerArgs
from memory.policies.component_offload import offload_component
from nodes.schedule_batch import Req
from nodes.stages.denoising import DenoisingStage
from pipelines.stages.eraserdit_erase._common import (
    field_summary,
    latent_frame_count,
)


def self_attention_backend_report(transformer) -> dict | None:
    """Effective self-attention backend, read off a live block processor."""
    blocks = getattr(transformer, "transformer_blocks", None)
    if not blocks:
        return None
    processor = getattr(blocks[0].attn1, "processor", None)
    report = getattr(processor, "attention_backend_report", None)
    return dict(report()) if callable(report) else None


class EraserDiTEraseDenoisingStage(DenoisingStage):
    """Denoising loop with optional torch.compile of the transformer.

    Extends the shared ``DenoisingStage`` so the compile wrapper, its warmup
    bookkeeping and the eager fallback all follow the framework's contract
    (plan §M3).
    """

    def __init__(self, transformer, scheduler, server_args=None):
        if server_args is None:
            from config.server_args import get_global_server_args

            server_args = get_global_server_args()
        validate_cfg_parallel(server_args)
        super().__init__(transformer, server_args)
        self._scheduler = scheduler

    @offload_component("transformer")
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        transformer = batch.modules.get("transformer") or self._transformer
        scheduler = batch.modules.get("scheduler") or self._scheduler

        latents = batch.latents
        cond_latents = batch.cond_latents
        mask_values = batch.mask_values
        timesteps = batch.timesteps
        if latents is None or cond_latents is None or timesteps is None:
            raise ValueError("denoising requires latent_preparation to run first")

        prompt_embeds = batch.prompt_embeds
        prompt_attention_mask = batch.prompt_attention_mask
        negative_prompt_embeds = batch.negative_prompt_embeds
        negative_attention_mask = batch.negative_attention_mask
        guidance_scale = float(batch.guidance_scale)
        rope_interpolation_scale = batch.rope_interpolation_scale

        model_frames = int(batch.padded_video.shape[2])
        temporal_ratio = int(getattr(batch.modules["vae"], "temporal_compression_ratio", 8))
        latent_num_frames = latent_frame_count(model_frames, temporal_ratio)
        latent_height = int(cond_latents.shape[-2])
        latent_width = int(cond_latents.shape[-1])

        model_dtype = prompt_embeds.dtype
        device = latents.device

        # One static shape per window for this model family, so a single compiled
        # graph covers every step; the signature is still recorded for reporting.
        transformer_for_forward = self.select_transformer_for_forward(
            transformer,
            batch=batch,
            local_shape=tuple(latents.shape),
            dynamic_cfg=False,
        )

        parallel_device = validate_cfg_parallel(server_args, batch)
        mesh_plan = resolve_mesh(server_args, batch)
        with EraserDiTMeshWindow(transformer, mesh_plan) as mesh, EraserDiTCFGWindow(transformer, parallel_device) as cfg_window, EraserDiTCacheWindow(
            batch, total_steps=len(timesteps),
            num_blocks=len(transformer.transformer_blocks),
            enable_torch_compile=server_args.enable_torch_compile,
        ) as cache_window, torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for step_index, timestep in enumerate(timesteps):
                step_start = time.perf_counter()
                latent_model_input = latents.to(model_dtype)
                cond_input = cond_latents.to(device=device)
                mask_input = mask_values.to(device=device)
                expanded_timestep = timestep.expand(1)

                negative_kwargs = dict(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=negative_prompt_embeds,
                    timestep=expanded_timestep,
                    encoder_attention_mask=negative_attention_mask,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    rope_interpolation_scale=rope_interpolation_scale,
                    attention_kwargs=None,
                    return_dict=False,
                    cond_latents=cond_input,
                    mask_values=mask_input,
                    **cache_window.kwargs("negative", step_index),
                )
                if mesh.active:
                    pass
                elif parallel_device is not None:
                    negative_future = cfg_window.submit(**negative_kwargs)
                else:
                    noise_pred_uncond = transformer_for_forward(**negative_kwargs)[0].float()

                positive_kwargs = dict(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=expanded_timestep,
                    encoder_attention_mask=prompt_attention_mask,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    rope_interpolation_scale=rope_interpolation_scale,
                    attention_kwargs=None,
                    return_dict=False,
                    cond_latents=cond_input,
                    mask_values=mask_input,
                    **cache_window.kwargs("positive", step_index),
                )
                if mesh.active:
                    noise_pred_uncond, noise_pred_text = mesh.predict(negative_kwargs, positive_kwargs)
                else:
                    noise_pred_text = transformer_for_forward(**positive_kwargs)[0].float()

                if parallel_device is not None:
                    noise_pred_uncond = negative_future.result().to(device)

                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
                batch.step_index = step_index
                if batch.metrics is not None:
                    batch.metrics.record_step(time.perf_counter() - step_start)

            if mesh.active:
                batch.extra["dit_parallel"] = mesh.report()

        if parallel_device is not None:
            batch.extra["cfg_parallel"] = {
                "enabled": True, "positive_device": str(device),
                "negative_device": str(parallel_device), "steps": cfg_window.steps,
                "replica_setup_seconds": cfg_window.setup_seconds,
                "secondary_peak_allocated_gib": torch.cuda.max_memory_allocated(parallel_device) / 1024**3,
                "transformer_cache_mode": "off",
            }
        if batch.metrics is not None:
            batch.metrics.record_operation("denoise")
        # Report the backend the forwards actually resolved to, not the request:
        # `auto` picks one at runtime and the matrix must record which.
        backend_report = self_attention_backend_report(self._transformer)
        if backend_report is not None:
            batch.extra["attention_backend"] = backend_report
        batch.noise_pred = noise_pred
        batch.latents = latents
        self.log_info(
            "%s | steps=%d",
            field_summary("latents", latents),
            len(timesteps),
        )
        return batch
