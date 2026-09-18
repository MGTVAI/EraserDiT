"""LTX095 erase – preprocess stage."""

from __future__ import annotations

from config.server_args import ServerArgs
from nodes.schedule_batch import Req
from nodes.stages.base import PipelineStage
from nodes.stages.model_specific_stages.ltx095_erase._common import (
    _field_summary,
    record_ltx095_sp_writer_stage_operation,
    record_ltx095_sp_writer_stage_skip,
    should_skip_ltx095_sp_writer_stage,
)
from utils.erase_preprocess import preprocess_single_window
from utils.ltx095_origin_contract import (
    resolve_runtime_sp_degree,
    resolve_spatial_alignment,
)


class LTX095ErasePreprocessStage(PipelineStage):
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
            operation="window_preprocess",
        )
        align_w, align_h = resolve_spatial_alignment(
            resolve_runtime_sp_degree(server_args)
        )
        result = preprocess_single_window(
            video=batch.video,
            mask=batch.mask,
            bbox=batch.bbox,
            infer_len=batch.infer_len,
            align_h=align_h,
            align_w=align_w,
            mask_dilate_iter=batch.mask_dilate_iter,
            mask_dilate_kernel=batch.mask_dilate_kernel,
            scale_area_ratio=batch.scale_area_ratio,
            min_pixels=batch.min_pixels,
            max_pixels=batch.max_pixels,
            force_crop_align=batch.force_crop_align,
            use_dynamic_num_frames=batch.use_dynamic_num_frames,
            time_sample=batch.time_sample,
            time_shift=batch.time_shift,
            prealigned_crop_bbox=batch.extra.get("prealigned_crop_bbox"),
        )
        batch.crop_bbox = result.crop_bbox
        batch.crop_video = result.crop_video
        batch.crop_mask = result.crop_mask
        batch.padded_video = result.padded_video
        batch.padded_mask = result.padded_mask
        batch.masked_video = result.masked_video
        self.log_info(
            "%s | %s | crop_bbox=%s",
            _field_summary("masked_video", batch.masked_video),
            _field_summary("padded_mask", batch.padded_mask),
            batch.crop_bbox,
        )
        return batch
