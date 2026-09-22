"""LTX095 erase - window postprocess stage."""

from __future__ import annotations

from config.server_args import ServerArgs
from nodes.schedule_batch import Req
from nodes.stages.base import PipelineStage
from pipelines.stages.ltx095_erase._common import (
    _distributed_context,
    _field_summary,
    _official_vae_parallel_active,
    _record_official_parallel_event,
)
from pipelines.runtime.windowing.commit_sync import (
    resolve_active_ltx095_window_commit_context,
)
from models.adapters.ltx095.postprocess import postprocess_single_window


class LTX095EraseWindowPostprocessStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        active = resolve_active_ltx095_window_commit_context(server_args)
        if active is not None and not active.is_writer:
            batch.decoded_video = None
            batch.crop_video_modified = None
            batch.output_video = None
            batch.output = None
            return batch
        official_parallel_active = _official_vae_parallel_active(server_args)
        rank = int(getattr(_distributed_context(server_args), "rank", 0) or 0)
        if official_parallel_active:
            _record_official_parallel_event(
                batch,
                "stage_enter",
                stage=self.__class__.__name__,
                rank=rank,
            )
        del server_args
        materialize_full_output = not bool(batch.extra.get("patch_commit_only", False))
        if batch.metrics is not None:
            batch.metrics.record_operation("window_postprocess")
        result = postprocess_single_window(
            original_video=batch.video,
            masked_video=batch.masked_video,
            padded_mask=batch.padded_mask,
            generated_video=batch.decoded_video,
            crop_bbox=batch.crop_bbox,
            original_num_frames=batch.crop_video.shape[2],
            crop_height=batch.crop_video.shape[-2],
            crop_width=batch.crop_video.shape[-1],
            direct_out=batch.direct_out,
            enable_colorfix=batch.enable_colorfix,
            dialate_kernel_size=batch.postprocess_dilate_kernel_size,
            guss_dialate_iter=batch.guss_dialate_iter,
            guss_dialate_sigma=batch.guss_dialate_sigma,
            remain_distance=batch.remain_distance,
            colorfix_per_channel=batch.colorfix_per_channel,
            materialize_full_output=materialize_full_output,
        )
        batch.extra["paste_mask_shape"] = (
            tuple(result.paste_mask.shape) if result.paste_mask is not None else None
        )
        batch.extra["crop_video_modified_shape"] = tuple(
            result.crop_video_modified.shape
        )
        batch.extra["patch_commit_only"] = bool(
            batch.extra.get("patch_commit_only", False)
        )
        batch.crop_video_modified = result.crop_video_modified.to(
            device=batch.video.device,
            dtype=batch.video.dtype,
        )
        batch.output_video = (
            result.output_video.to(device=batch.video.device, dtype=batch.video.dtype)
            if result.output_video is not None
            else None
        )
        batch.output = (
            batch.output_video
            if batch.output_video is not None
            else batch.crop_video_modified
        )
        if active is not None:
            batch.decoded_video = None
        if official_parallel_active:
            _record_official_parallel_event(
                batch,
                "stage_complete",
                stage=self.__class__.__name__,
                rank=rank,
                output_shape=(
                    tuple(batch.output_video.shape)
                    if batch.output_video is not None
                    else tuple(batch.crop_video_modified.shape)
                ),
            )
        if batch.output_video is not None:
            self.log_info("%s", _field_summary("output_video", batch.output_video))
        else:
            self.log_info(
                "%s", _field_summary("crop_video_modified", batch.crop_video_modified)
            )
        return batch
