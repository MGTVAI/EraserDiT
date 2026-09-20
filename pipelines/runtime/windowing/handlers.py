"""Composed-pipeline handlers for the windowed runtime.

Model-agnostic glue that adapts the ``pipelines.runtime`` runtime callbacks to the
pipeline-side helpers.  Shared by every windowed erase pipeline; nothing here is
specific to a particular diffusion model.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from nodes.schedule_batch import Req
from pipelines.runtime.contracts import (
    LTX095EraseRuntimeContext,
    ObjectRuntimeState,
)
from pipelines.runtime.events import (
    record_runtime_event as runtime_record_event,
    record_task_state_snapshot as runtime_record_task_state_snapshot,
    update_window_state as runtime_update_window_state,
)
from pipelines.runtime.io.streaming import (
    ensure_window_cache_loaded as runtime_ensure_window_cache_loaded,
    evict_cache_before as runtime_evict_cache_before,
    flush_windowed_frames as runtime_flush_windowed_frames,
    materialize_object_window_mask as runtime_materialize_object_window_mask,
    release_mask_frames as runtime_release_mask_frames,
)
from pipelines.runtime.windowing.cache_ops import (
    append_passthrough_gap as runtime_append_passthrough_gap,
    create_empty_cache_like as runtime_create_empty_cache_like,
    register_runtime_task_chain_hooks as runtime_register_task_chain_hooks,
    set_object_overlap_cache as runtime_set_object_overlap_cache,
)
from pipelines.runtime.windowing.commit_ops import (
    commit_ltx095_window_to_object_output,
    record_ltx095_skipped_object_window,
)
from pipelines.runtime.drivers.windowed import finalize_ltx095_object_window_step
from utils.video_io import (
    ArrayFrameCache,
    ChunkedFrameCache,
    TensorFrameCache,
    binarize_mask_tensor,
    ensure_nchw_video,
    read_mask_tensor,
    read_video_tensor,
)
from utils.windowing import WindowSpec

def _window_cache_impl_name(cache: ArrayFrameCache | ChunkedFrameCache | TensorFrameCache | None) -> str:
    if cache is None:
        return "none"
    return type(cache).__name__




def _ensure_5d_video(video: torch.Tensor, channels: int | None = None) -> torch.Tensor:
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(f"Expected 4D/5D video tensor, got {tuple(video.shape)}")
    if channels is not None and video.shape[1] != channels:
        raise ValueError(
            f"Expected video channel count {channels}, got {video.shape[1]}"
        )
    return video


def _module_device(module: Any) -> torch.device:
    if hasattr(module, "device"):
        return getattr(module, "device")
    return next(module.parameters()).device


def _synchronize_timing_device(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


def _select_sequence_item(value: Any, index: int) -> Any:
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return value[min(index, len(value) - 1)]
    return value


def _build_runtime_video(
    video_source: torch.Tensor | str,
) -> tuple[torch.Tensor, dict[str, object]]:
    if isinstance(video_source, str):
        video, metadata = read_video_tensor(video_source)
        video = video.permute(1, 0, 2, 3).unsqueeze(0)
    else:
        if video_source.ndim == 4:
            video = ensure_nchw_video(video_source).permute(1, 0, 2, 3).unsqueeze(0)
        elif video_source.ndim == 5:
            video = _ensure_5d_video(video_source)
            if video.shape[0] != 1:
                raise ValueError(
                    "Minimal LTX095 erase pipeline only supports batch size 1"
                )
        else:
            raise ValueError(
                f"Unsupported video tensor shape: {tuple(video_source.shape)}"
            )
        metadata = {
            "fps": None,
            "codec_name": None,
            "num_frames": int(video.shape[2]),
            "width": int(video.shape[-1]),
            "height": int(video.shape[-2]),
        }
    return _ensure_5d_video(video.float(), channels=3), metadata


def _build_runtime_mask(
    mask_source: torch.Tensor | str, num_frames: int
) -> torch.Tensor:
    if isinstance(mask_source, str):
        mask, _metadata = read_mask_tensor(mask_source)
    else:
        if mask_source.ndim == 5:
            mask = mask_source[0].permute(1, 0, 2, 3)
        elif mask_source.ndim == 4:
            mask = ensure_nchw_video(mask_source)
        else:
            raise ValueError(
                f"Unsupported mask tensor shape: {tuple(mask_source.shape)}"
            )
        mask = binarize_mask_tensor(mask)
    if mask.ndim == 4:
        mask = mask
    else:
        raise ValueError(f"Unsupported mask tensor shape: {tuple(mask.shape)}")
    if mask.shape[0] < num_frames:
        tail = mask[-1:, ...].repeat(num_frames - mask.shape[0], 1, 1, 1)
        mask = torch.cat([mask, tail], dim=0)
    return _ensure_5d_video(
        mask[:num_frames].permute(1, 0, 2, 3).unsqueeze(0).float(), channels=1
    )


def _record_runtime_event(
    context: LTX095EraseRuntimeContext,
    event: str,
    task_state: ObjectRuntimeState | None = None,
    **payload: Any,
) -> dict[str, Any]:
    return runtime_record_event(
        context,
        event,
        task_state=task_state,
        **payload,
    )


def _record_task_state_snapshot(
    context: LTX095EraseRuntimeContext,
    task_state: ObjectRuntimeState,
    phase: str,
    **payload: Any,
) -> dict[str, Any]:
    return runtime_record_task_state_snapshot(
        context,
        task_state,
        phase,
        **payload,
    )


def _update_runtime_window_state(
    context: LTX095EraseRuntimeContext,
    object_index: int,
    window_index: int,
    **payload: Any,
) -> None:
    runtime_update_window_state(
        context,
        object_index,
        window_index,
        record_runtime_event_fn=_record_runtime_event,
        record_task_state_snapshot_fn=_record_task_state_snapshot,
        **payload,
    )


def _ensure_runtime_window_cache_loaded(
    context: LTX095EraseRuntimeContext,
    spec: WindowSpec,
) -> None:
    runtime_ensure_window_cache_loaded(
        context=context,
        spec=spec,
        record_runtime_event=_record_runtime_event,
    )


def _evict_runtime_cache_before(
    context: LTX095EraseRuntimeContext,
    frame_index: int,
) -> None:
    runtime_evict_cache_before(
        context=context,
        frame_index=frame_index,
        record_runtime_event=_record_runtime_event,
    )


def _release_runtime_mask_frames(
    context: LTX095EraseRuntimeContext,
    release_end: int,
    source: str,
) -> None:
    runtime_release_mask_frames(
        context=context,
        release_end=release_end,
        source=source,
        record_runtime_event=_record_runtime_event,
    )


def _flush_runtime_windowed_frames(
    context: LTX095EraseRuntimeContext,
    flush_end: int,
) -> None:
    runtime_flush_windowed_frames(
        context=context,
        flush_end=flush_end,
        record_runtime_event=_record_runtime_event,
        evict_cache_before_fn=_evict_runtime_cache_before,
        release_mask_frames_fn=_release_runtime_mask_frames,
    )


def _register_runtime_task_chain_hooks(
    context: LTX095EraseRuntimeContext,
    object_states: list[ObjectRuntimeState],
) -> None:
    runtime_register_task_chain_hooks(
        context=context,
        object_states=object_states,
        record_runtime_event=_record_runtime_event,
        record_task_state_snapshot=_record_task_state_snapshot,
    )


def _materialize_runtime_object_window_mask(
    context: LTX095EraseRuntimeContext,
    object_state: ObjectRuntimeState,
    spec: WindowSpec,
    crop_bbox: tuple[int, int, int, int] | None = None,
) -> torch.Tensor:
    _ = object_state
    return runtime_materialize_object_window_mask(
        context=context,
        object_state=object_state,
        spec=spec,
        ensure_window_cache_loaded_fn=_ensure_runtime_window_cache_loaded,
        crop_bbox=crop_bbox,
    )


def _create_runtime_empty_cache_like(
    cache: ArrayFrameCache | ChunkedFrameCache | TensorFrameCache,
    start_index: int = 0,
) -> ArrayFrameCache | ChunkedFrameCache | TensorFrameCache:
    return runtime_create_empty_cache_like(cache, start_index=start_index)


def _commit_runtime_window_to_object_output(
    context: LTX095EraseRuntimeContext,
    object_state: ObjectRuntimeState,
    spec: WindowSpec,
    window_batch: Req,
) -> None:
    commit_started = time.perf_counter()
    try:
        commit_ltx095_window_to_object_output(
            context=context,
            object_state=object_state,
            spec=spec,
            window_batch=window_batch,
            append_passthrough_gap_fn=runtime_append_passthrough_gap,
            set_object_overlap_cache_fn=runtime_set_object_overlap_cache,
            record_runtime_event_fn=_record_runtime_event,
            record_task_state_snapshot_fn=_record_task_state_snapshot,
            update_window_state_fn=_update_runtime_window_state,
        )
    finally:
        context.record_runtime_timing(
            "cache_commit", time.perf_counter() - commit_started
        )


def _record_runtime_skipped_object_window(
    context: LTX095EraseRuntimeContext,
    object_state: ObjectRuntimeState,
    spec: WindowSpec,
    reason: str,
    prompt: Any,
    negative_prompt: Any,
) -> None:
    record_ltx095_skipped_object_window(
        context=context,
        object_state=object_state,
        spec=spec,
        reason=reason,
        prompt=prompt,
        negative_prompt=negative_prompt,
        append_passthrough_gap_fn=runtime_append_passthrough_gap,
        set_object_overlap_cache_fn=runtime_set_object_overlap_cache,
        record_runtime_event_fn=_record_runtime_event,
        record_task_state_snapshot_fn=_record_task_state_snapshot,
        update_window_state_fn=_update_runtime_window_state,
    )


def _finalize_runtime_object_window_step(
    context: LTX095EraseRuntimeContext,
    object_states: list[ObjectRuntimeState],
    object_state: ObjectRuntimeState,
    spec: WindowSpec,
) -> None:
    finalize_ltx095_object_window_step(
        context=context,
        object_states=object_states,
        object_state=object_state,
        spec=spec,
        set_object_overlap_cache_fn=runtime_set_object_overlap_cache,
        record_runtime_event_fn=_record_runtime_event,
        record_task_state_snapshot_fn=_record_task_state_snapshot,
    )

