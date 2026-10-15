"""Mask ingress and lifecycle helpers for videoerase runtime paths."""

from __future__ import annotations

import os

import numpy as np
import torch

from pipelines.runtime.contracts import EraseRuntimeContext
from pipelines.runtime.contracts import _is_windowed_runtime_mode
from utils.video_io import ChunkedFrameCache, mask_uint8_to_tensor
from utils.windowing import WindowSpec


def materialize_window_mask(
    *,
    context: EraseRuntimeContext,
    spec: WindowSpec,
    crop_bbox: tuple[int, int, int, int] | None = None,
) -> torch.Tensor:
    if _is_windowed_runtime_mode(context.runtime_mode):
        if context.mask_frame_cache is None:
            raise ValueError(f"{context.runtime_mode} runtime missing mask frame cache")
        cache = context.mask_frame_cache
        if (os.environ.get("MGERASE_MASK_WINDOW_BUFFER_OPT") == "1"
            and isinstance(cache, ChunkedFrameCache) and cache.dtype == np.uint8):
            shape_tail = cache.shape_tail
            if crop_bbox is not None:
                x, y, width, height = crop_bbox
                shape_tail = np.empty((0, *shape_tail), dtype=np.uint8)[
                    :, y:y + height, x:x + width
                ].shape[1:]
            frames = np.empty((spec.input_len, *shape_tail), dtype=np.float32)
            frames[:spec.active_start_offset] = 0
            frames[spec.active_end_offset:] = 0
            cache.slice_into(
                spec.deal_start, spec.commit_end,
                frames[spec.active_start_offset:spec.active_end_offset],
                crop_bbox=crop_bbox, allow_cast=True,
            )
            return (torch.from_numpy(frames)[:, None].div_(255.0)
                    .permute(1, 0, 2, 3).unsqueeze(0))
        active_mask_frames = context.mask_frame_cache.slice(
            spec.deal_start, spec.commit_end
        )
        if crop_bbox is not None:
            x, y, width, height = crop_bbox
            active_mask_frames = active_mask_frames[
                :, y : y + height, x : x + width
            ]
        active_mask = (
            mask_uint8_to_tensor(active_mask_frames)
            .permute(1, 0, 2, 3)
            .unsqueeze(0)
        )
    else:
        if context.mask_cache is None:
            raise ValueError("full runtime missing mask cache")
        active_mask = context.mask_cache[:, :, spec.deal_start : spec.commit_end].clone()

    window_mask = active_mask.new_zeros(
        (
            active_mask.shape[0],
            active_mask.shape[1],
            spec.input_len,
            active_mask.shape[-2],
            active_mask.shape[-1],
        )
    )
    window_mask[:, :, spec.active_start_offset : spec.active_end_offset] = active_mask[
        :, :, : spec.deal_length
    ]
    return window_mask


def consume_window_mask(
    *,
    context: EraseRuntimeContext,
    spec: WindowSpec,
    crop_bbox: tuple[int, int, int, int],
) -> None:
    x, y, w, h = crop_bbox
    if spec.deal_length <= 0 or w <= 0 or h <= 0:
        return

    clear_start = int(spec.deal_start)
    clear_end = int(spec.commit_end)
    if _is_windowed_runtime_mode(context.runtime_mode):
        if context.mask_frame_cache is None:
            raise ValueError(f"{context.runtime_mode} runtime missing mask frame cache")
        clear_start = max(clear_start, int(context.mask_frame_cache.start_index))
        clear_end = min(clear_end, int(context.mask_frame_cache.end_index))
        if clear_end <= clear_start:
            return
        context.mask_frame_cache.fill_region(
            clear_start,
            clear_end,
            (x, y, w, h),
            0,
        )
        return

    if context.mask_cache is None:
        raise ValueError("full runtime missing mask cache")
    context.mask_cache[:, :, clear_start:clear_end, y : y + h, x : x + w] = 0
