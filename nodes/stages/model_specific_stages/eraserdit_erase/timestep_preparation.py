"""EraserDiT erase – timestep preparation stage."""

from __future__ import annotations

from config.server_args import ServerArgs
from nodes.schedule_batch import Req
from nodes.stages.base import PipelineStage
from utils.latent import build_rope_interpolation_scale


class EraserDiTEraseTimestepPreparationStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        vae = batch.modules["vae"]
        # The baseline hard-codes ``frame_rate=25`` for the RoPE temporal scale;
        # it deliberately does not follow the input video frame rate.
        batch.rope_interpolation_scale = build_rope_interpolation_scale(
            temporal_ratio=int(vae.temporal_compression_ratio),
            frame_rate=int(batch.frame_rate),
            spatial_ratio=int(vae.spatial_compression_ratio),
        )
        self.log_info("rope_interpolation_scale=%s", batch.rope_interpolation_scale)
        return batch
