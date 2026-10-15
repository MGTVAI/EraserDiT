"""EraserDiT erase – window validation stage."""

from __future__ import annotations

import os

import torch

from config.server_args import ServerArgs
from nodes.schedule_batch import Req
from nodes.stages.base import PipelineStage
from pipelines.stages.eraserdit_erase._common import field_summary
from nodes.stages.validators import StageValidators, VerificationResult


def _validate_window_tensor(value) -> bool:
    if (os.environ.get("MGERASE_WINDOW_UINT8_VALIDATION_OPT") == "1"
            and isinstance(value, torch.Tensor) and value.dtype == torch.uint8):
        # uint8 cannot represent NaN; retain the structural check. Float and
        # complex inputs still use the existing NaN validation.
        return value.dim() == 5
    return StageValidators.tensor_with_dims(value, 5)


class EraserDiTEraseWindowValidationStage(PipelineStage):
    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("video", batch.video, _validate_window_tensor)
        result.add_check("mask", batch.mask, _validate_window_tensor)
        for name in ("vae", "transformer", "scheduler", "text_encoder", "tokenizer"):
            result.add_check(
                f"modules.{name}", batch.modules.get(name), StageValidators.not_none
            )
        return result

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        if batch.video.shape[0] != 1 or batch.mask.shape[0] != 1:
            raise ValueError("EraserDiT erase stages only support batch size 1")
        self.log_info(
            "%s | %s | bbox=%s",
            field_summary("video", batch.video),
            field_summary("mask", batch.mask),
            batch.bbox,
        )
        return batch
