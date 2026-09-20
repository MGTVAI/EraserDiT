"""Cache and forwarding helpers for the LTX095 windowed videoerase runtime."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch

from pipelines.runtime.contracts import (
    LTX095EraseRuntimeContext,
    ObjectRuntimeState,
)
from utils.video_io import ArrayFrameCache, ChunkedFrameCache, TensorFrameCache

FrameCache = ArrayFrameCache | ChunkedFrameCache | TensorFrameCache


def _append_owned(
    cache: FrameCache,
    frames: np.ndarray | torch.Tensor,
) -> int:
    append_owned = getattr(cache, "append_owned", None)
    if append_owned is None:
        cache.append(frames)
        return int(frames.nbytes)
    return int(append_owned(frames))


def create_empty_cache_like(
    cache: FrameCache,
    *,
    start_index: int = 0,
) -> FrameCache:
    if isinstance(cache, ArrayFrameCache):
        return ArrayFrameCache(
            start_index=start_index,
            shape_tail=cache.shape_tail,
            dtype=cache.dtype,
        )
    if isinstance(cache, ChunkedFrameCache):
        return ChunkedFrameCache(
            start_index=start_index,
            shape_tail=cache.shape_tail,
            dtype=cache.dtype,
        )
    if isinstance(cache, TensorFrameCache):
        return TensorFrameCache(
            start_index=start_index,
            shape_tail=cache.shape_tail,
            dtype=cache.dtype,
        )
    raise TypeError(f"Unsupported cache type: {type(cache).__name__}")


def clone_cache(
    cache: FrameCache,
) -> FrameCache:
    cloned = create_empty_cache_like(cache, start_index=cache.start_index)
    if cache.num_frames > 0:
        _append_owned(cloned, cache.slice(cache.start_index, cache.end_index))
    return cloned


def append_cache_range(
    target_cache: FrameCache,
    source_cache: FrameCache,
    start_frame: int,
    end_frame: int,
) -> tuple[int, int]:
    if end_frame <= start_frame:
        return 0, 0
    if target_cache.end_index != start_frame:
        raise ValueError(
            f"Object-chain append expects contiguous write at {start_frame}, "
            f"but target cache currently ends at {target_cache.end_index}"
        )
    frames = source_cache.slice(start_frame, end_frame)
    transferred_bytes = int(frames.nbytes)
    copied_bytes = transferred_bytes + _append_owned(target_cache, frames)
    return copied_bytes, transferred_bytes


def reset_cache_range(
    cache: FrameCache | None,
    source_cache: FrameCache,
    start_frame: int,
    end_frame: int,
) -> FrameCache:
    del cache
    refreshed = create_empty_cache_like(source_cache, start_index=start_frame)
    if end_frame > start_frame:
        _append_owned(refreshed, source_cache.slice(start_frame, end_frame))
    return refreshed


def reset_cache_frames(
    cache: FrameCache | None,
    cache_template: FrameCache,
    start_frame: int,
    frames: np.ndarray | torch.Tensor | None,
) -> FrameCache:
    refreshed = create_empty_cache_like(cache or cache_template, start_index=start_frame)
    if frames is not None and len(frames) > 0:
        refreshed.append(frames)
    return refreshed


def append_passthrough_gap(
    output_cache: FrameCache,
    input_cache: FrameCache,
    end_frame: int,
) -> tuple[int, int]:
    gap_start = output_cache.end_index
    gap_end = min(int(end_frame), input_cache.end_index)
    if gap_end <= gap_start:
        return gap_start, gap_start
    _append_owned(output_cache, input_cache.slice(gap_start, gap_end))
    return gap_start, gap_end


def set_object_overlap_cache(
    object_state: ObjectRuntimeState,
    start_frame: int,
    frames: np.ndarray | torch.Tensor | None,
) -> None:
    overlap_cache = object_state.overlap_cache
    if overlap_cache is None:
        overlap_cache = create_empty_cache_like(
            object_state.output_cache,
            start_index=start_frame,
        )
    object_state.overlap_cache = reset_cache_frames(
        overlap_cache,
        object_state.output_cache,
        start_frame,
        frames,
    )


def forward_task_channel_range(
    *,
    context: LTX095EraseRuntimeContext,
    source_state: ObjectRuntimeState,
    target_state: ObjectRuntimeState,
    start_index: int,
    length: int,
    channel: str,
    record_runtime_event: Callable[..., dict[str, Any]],
    record_task_state_snapshot: Callable[..., dict[str, Any]],
) -> None:
    if length <= 0:
        return
    channel_name = str(channel).strip().lower()
    if channel_name not in {"raw", "modified"}:
        raise ValueError(f"Unsupported forwarding channel: {channel}")

    end_index = int(start_index) + int(length)
    source_cache = (
        source_state.input_cache if channel_name == "raw" else source_state.output_cache
    )
    if target_state.input_cache.end_index != int(start_index):
        raise ValueError(
            f"Forwarding channel `{channel_name}` for object {source_state.object_index} "
            f"expects target object {target_state.object_index} input cache at {start_index}, "
            f"got {target_state.input_cache.end_index}"
        )

    transition_started = time.perf_counter()
    copied_bytes, transferred_bytes = append_cache_range(
        target_state.input_cache,
        source_cache,
        int(start_index),
        end_index,
    )
    context.record_runtime_transfer("object_chain_copy", copied_bytes)
    context.record_runtime_transfer("object_chain_transfer", transferred_bytes)
    target_state.input_frontier = max(target_state.input_frontier, end_index)
    transition_history_item = {
            "from_object_index": source_state.object_index,
            "to_object_index": target_state.object_index,
            "window_index": int(source_state.next_window_index),
            "scene_index": int(source_state.scene_index),
            "transition_start": int(start_index),
            "transition_end": end_index,
            "length": int(length),
            "channel": channel_name,
            "source_cache": (
                "raw_input_cache" if channel_name == "raw" else "stable_output_cache"
            ),
        }
    context.object_transition_history.append(transition_history_item)
    record_runtime_event(
        context,
        f"task_forward_{channel_name}",
        task_state=source_state,
        to_task_index=target_state.task_index,
        to_object_index=target_state.object_index,
        window_index=int(source_state.next_window_index),
        transition_start=int(start_index),
        transition_end=end_index,
        length=int(length),
        channel=channel_name,
    )
    record_task_state_snapshot(
        context,
        target_state,
        phase=f"forwarded_{channel_name}_input",
        from_task_index=source_state.task_index,
        from_object_index=source_state.object_index,
        transition_start=int(start_index),
        transition_end=end_index,
        channel=channel_name,
    )
    transition_duration_seconds = time.perf_counter() - transition_started
    transition_history_item["duration_seconds"] = transition_duration_seconds
    context.record_runtime_timing(
        "object_transition", transition_duration_seconds
    )


def register_runtime_task_chain_hooks(
    *,
    context: LTX095EraseRuntimeContext,
    object_states: list[ObjectRuntimeState],
    record_runtime_event: Callable[..., dict[str, Any]],
    record_task_state_snapshot: Callable[..., dict[str, Any]],
) -> None:
    for object_index, object_state in enumerate(object_states[:-1]):
        next_state = object_states[object_index + 1]
        object_state.register_pop_raw_input_hook(
            f"forward_raw_to_task_{next_state.task_index}",
            lambda start_index, length, src=object_state, dst=next_state: forward_task_channel_range(
                context=context,
                source_state=src,
                target_state=dst,
                start_index=start_index,
                length=length,
                channel="raw",
                record_runtime_event=record_runtime_event,
                record_task_state_snapshot=record_task_state_snapshot,
            ),
        )
        object_state.register_pop_modified_hook(
            f"forward_modified_to_task_{next_state.task_index}",
            lambda start_index, length, src=object_state, dst=next_state: forward_task_channel_range(
                context=context,
                source_state=src,
                target_state=dst,
                start_index=start_index,
                length=length,
                channel="modified",
                record_runtime_event=record_runtime_event,
                record_task_state_snapshot=record_task_state_snapshot,
            ),
        )
