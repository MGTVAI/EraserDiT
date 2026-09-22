"""LTX095 erase – window validation stage."""

from __future__ import annotations

from config.server_args import ServerArgs
from nodes.schedule_batch import Req
from nodes.stages.base import PipelineStage
from nodes.stages.validators import StageValidators, VerificationResult
from pipelines.stages.ltx095_erase._common import (
    _field_summary,
    _official_vae_parallel_active,
    _is_writer_rank,
    record_ltx095_sp_writer_stage_operation,
    record_ltx095_sp_writer_stage_skip,
    should_skip_ltx095_sp_writer_stage,
)


class LTX095EraseWindowValidationStage(PipelineStage):
    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        if should_skip_ltx095_sp_writer_stage(batch, server_args):
            return result
        result.add_check("video", batch.video, StageValidators.with_dims(5))
        result.add_check("mask", batch.mask, StageValidators.with_dims(5))
        result.add_check("modules.vae", batch.modules.get("vae"), StageValidators.not_none)
        if not (_official_vae_parallel_active(server_args) and not _is_writer_rank(server_args)):
            result.add_check("modules.transformer", batch.modules.get("transformer"), StageValidators.not_none)
            result.add_check("modules.scheduler", batch.modules.get("scheduler"), StageValidators.not_none)
            result.add_check("modules.text_encoder", batch.modules.get("text_encoder"), StageValidators.not_none)
            result.add_check("modules.tokenizer", batch.modules.get("tokenizer"), StageValidators.not_none)
        return result

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if should_skip_ltx095_sp_writer_stage(batch, server_args):
            record_ltx095_sp_writer_stage_skip(
                batch,
                stage=self.__class__.__name__,
            )
            return batch
        record_ltx095_sp_writer_stage_operation(
            batch,
            server_args,
            operation="window_validation",
        )
        if batch.video.shape[0] != 1 or batch.mask.shape[0] != 1:
            raise ValueError("Phase 5 single-window stages only support batch size 1")
        self.log_info(
            "%s | %s | bbox=%s",
            _field_summary("video", batch.video),
            _field_summary("mask", batch.mask),
            batch.bbox,
        )
        return batch
