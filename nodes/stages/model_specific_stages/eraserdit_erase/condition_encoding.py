"""EraserDiT erase – condition encoding stage.

Port of the first half of ``LTXVideoToVideoPipeline.prepare_latents``: the VAE
encode of the masked window video.  Two details differ from the LTX095 path and
are load-bearing:

* The baseline keeps the VAE's ``latents_mean`` / ``latents_std`` normalisation
  with ``scaling_factor=1.0`` (the pipeline's own default), *not*
  ``vae.config.scaling_factor``.
* ``latent_dist.sample(generator)`` is used, **not** ``mode()`` -- the encode is
  a random draw and consumes the request generator.  ``vibe/plan.md`` §4.8 fixes
  the per-window draw order as encode sample -> init noise -> decode noise.
"""

from __future__ import annotations

import torch

from config.server_args import ServerArgs
from memory.policies.component_offload import offload_component
from memory.policies.memory_phase_controller import MemoryPhase
from nodes.schedule_batch import Req
from nodes.stages.base import PipelineStage
from nodes.stages.model_specific_stages.eraserdit_erase._common import (
    field_summary,
    get_task_state,
    normalize_latents,
)
from utils.resource_policy import module_device, module_dtype


class EraserDiTEraseConditionEncodingStage(PipelineStage):
    @offload_component("vae", phase=MemoryPhase.VAE_ENCODE)
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        vae = batch.modules["vae"]
        state = get_task_state(batch)

        # ``video_processor.preprocess`` normalises [0, 1] to [-1, 1].
        vae_input = (batch.padded_video * 2 - 1).to(
            device=module_device(vae), dtype=module_dtype(vae)
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            posterior = vae.encode(vae_input)
            if hasattr(posterior, "latent_dist"):
                cond_latents = posterior.latent_dist.sample(state.generator)
            else:
                cond_latents = posterior.latents
        cond_latents = cond_latents.to(torch.float32)
        batch.cond_latents = normalize_latents(
            cond_latents,
            vae.latents_mean,
            vae.latents_std,
            scaling_factor=1.0,
        )

        # Baseline: ``interpolate(masks, [h, w])`` with the default nearest mode,
        # then ``rearrange("b f c h w -> b c f h w")``.
        latent_height = int(batch.cond_latents.shape[-2])
        latent_width = int(batch.cond_latents.shape[-1])
        mask_latents = batch.padded_mask[0].permute(1, 0, 2, 3).to(torch.float32)
        mask_values = torch.nn.functional.interpolate(
            mask_latents, [latent_height, latent_width]
        )
        batch.mask_values = (
            mask_values.unsqueeze(0)
            .permute(0, 2, 1, 3, 4)
            .to(device=batch.cond_latents.device, dtype=batch.cond_latents.dtype)
        )
        batch.latent_shape = tuple(batch.cond_latents.shape)
        self.log_info(
            "%s | %s",
            field_summary("cond_latents", batch.cond_latents),
            field_summary("mask_values", batch.mask_values),
        )
        return batch
