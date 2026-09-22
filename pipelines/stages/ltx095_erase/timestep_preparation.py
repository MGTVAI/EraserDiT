"""LTX095 erase – timestep preparation stage."""

from __future__ import annotations

from config.server_args import ServerArgs
from nodes.schedule_batch import Req
from nodes.stages.base import PipelineStage
from pipelines.stages.ltx095_erase._common import (
    _field_summary,
    _record_official_parallel_event,
    _should_skip_writer_only_stage,
)
from utils.latent import build_rope_interpolation_scale


class LTX095EraseTimestepPreparationStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if _should_skip_writer_only_stage(server_args, self.__class__.__name__):
            _record_official_parallel_event(
                batch,
                "stage_skip_non_writer",
                stage=self.__class__.__name__,
                reason="writer_only_stage",
            )
            return batch
        del server_args
        vae = batch.modules["vae"]
        batch.rope_interpolation_scale = build_rope_interpolation_scale(
            temporal_ratio=int(vae.temporal_compression_ratio),
            frame_rate=batch.fps,
            spatial_ratio=int(vae.spatial_compression_ratio),
        )
        self.log_info("rope_interpolation_scale=%s", batch.rope_interpolation_scale)
        return batch
