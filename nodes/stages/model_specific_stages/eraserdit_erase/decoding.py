"""EraserDiT erase – decoding stage.

Port of the decode tail of ``LTXVideoToVideoPipeline.__call__``.  The decode noise
draw is numerically void (``decode_noise_scale == 0.0``) but still consumes the
request generator, so it must be issued to keep the random stream aligned with
the baseline (``vibe/plan.md`` §4.8).
"""

from __future__ import annotations

import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

from config.server_args import ServerArgs
from nodes.schedule_batch import Req
from nodes.stages.base import PipelineStage
from nodes.stages.model_specific_stages.eraserdit_erase._common import (
    denormalize_latents,
    field_summary,
    get_task_state,
)
from utils.resource_policy import module_device, module_dtype


class EraserDiTEraseDecodingStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        vae = batch.modules["vae"]
        state = get_task_state(batch)
        latents = batch.latents
        if latents is None:
            raise ValueError("decoding requires denoising to run first")

        device = module_device(vae)
        vae_dtype = module_dtype(vae)

        latents = denormalize_latents(
            latents,
            vae.latents_mean,
            vae.latents_std,
            scaling_factor=float(vae.config.scaling_factor),
        )
        latents = latents.to(vae_dtype)

        # Drawn but numerically inert at decode_noise_scale == 0.0.
        noise = randn_tensor(
            latents.shape,
            generator=state.generator,
            device=device,
            dtype=latents.dtype,
        )
        decode_timestep = torch.tensor(
            [float(batch.decode_timestep)], device=device, dtype=latents.dtype
        )
        decode_noise_scale = torch.tensor(
            [float(batch.decode_noise_scale)], device=device, dtype=latents.dtype
        )[:, None, None, None, None]
        latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            video = vae.decode(latents, decode_timestep, return_dict=False)[0]

        processor = VideoProcessor(
            vae_scale_factor=int(getattr(vae, "spatial_compression_ratio", 32))
        )
        # [B, C, F, H, W] -> [B, F, C, H, W], denormalised and clamped to [0, 1].
        decoded = processor.postprocess_video(video, output_type="pt")
        batch.decoded_video = decoded.permute(0, 2, 1, 3, 4).contiguous()
        batch.latents = None
        self.log_info("%s", field_summary("decoded_video", batch.decoded_video))
        return batch
