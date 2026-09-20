"""Streaming runtime IO and lifecycle helpers for LTX095 pipelines.runtime."""

from __future__ import annotations

import time
from typing import Callable

import torch

from pipelines.runtime.contracts import LTX095EraseRuntimeContext
from pipelines.runtime.contracts import ObjectRuntimeState
from pipelines.runtime.contracts import _is_windowed_runtime_mode
from pipelines.runtime.io.masks import materialize_ltx095_window_mask
from utils.video_io import (
    TensorFrameCache,
    binarize_mask_array,
    frames_tensor_to_uint8,
)
from utils.windowing import WindowSpec


def _write_streaming_frames(
    *,
    context: LTX095EraseRuntimeContext,
    frames: torch.Tensor,
) -> None:
    """Convert BF16 frames once at the writer boundary."""

    writer = context.sequential_video_writer
    if frames.shape[0] == 0 or writer is None:
        return
    write_owned = getattr(writer, "write_frames_owned", writer.write_frames)
    frames_uint8 = frames_tensor_to_uint8(frames)
    context.record_runtime_transfer(
        "writer_bf16_to_uint8", int(frames_uint8.nbytes)
    )
    write_owned(frames_uint8)


def release_mask_frames(
    *,
    context: LTX095EraseRuntimeContext,
    release_end: int,
    source: str,
    record_runtime_event: Callable[..., dict],
) -> None:
    if context.mask_frame_cache is None:
        return
    release_start = int(context.mask_release_frontier)
    bounded_release_end = min(int(release_end), context.mask_frame_cache.end_index)
    if bounded_release_end <= release_start:
        return
    released_frames = context.mask_frame_cache.pop_before(bounded_release_end)
    context.mask_release_frontier = bounded_release_end
    history_item = {
        "release_start": release_start,
        "release_end": bounded_release_end,
        "released_frames": int(released_frames.shape[0]),
        "source": str(source),
        "mask_cache_start": int(context.mask_frame_cache.start_index),
        "mask_cache_end": int(context.mask_frame_cache.end_index),
    }
    context.mask_release_event_history.append(history_item)
    record_runtime_event(
        context,
        "mask_release",
        task_state=context.object_states[-1] if context.object_states else None,
        **history_item,
    )


def ensure_window_cache_loaded(
    *,
    context: LTX095EraseRuntimeContext,
    spec: WindowSpec,
    record_runtime_event: Callable[..., dict],
) -> None:
    if not _is_windowed_runtime_mode(context.runtime_mode):
        return
    if context.video_frame_cache is None or context.mask_frame_cache is None:
        raise ValueError(f"{context.runtime_mode} runtime missing frame caches")
    if spec.load_end <= context.video_frame_cache.end_index:
        return

    if context.window_runtime_mode == "preload":
        if context.mask_source_frames is None:
            raise ValueError("windowed preload runtime missing mask source frames")
        raise ValueError(
            f"window cache does not contain required frames [{spec.load_start}, {spec.load_end})"
        )

    if context.window_runtime_mode != "streaming":
        raise ValueError(f"Unsupported window runtime mode: {context.window_runtime_mode}")

    if spec.load_start < context.video_frame_cache.start_index:
        raise ValueError(
            f"windowed_streaming does not support backtracking cache loads: "
            f"requested start {spec.load_start}, cache starts at {context.video_frame_cache.start_index}"
        )
    if context.sequential_video_reader is None or context.sequential_mask_reader is None:
        raise ValueError("windowed_streaming runtime missing sequential readers")

    missing_frames = spec.load_end - context.video_frame_cache.end_index
    if missing_frames <= 0:
        return

    video_frames = context.sequential_video_reader.read_frames(missing_frames)
    mask_frames = context.sequential_mask_reader.read_frames(missing_frames)
    if video_frames.shape[0] != missing_frames or mask_frames.shape[0] != missing_frames:
        raise RuntimeError(
            f"windowed_streaming reader returned mismatched frame counts: "
            f"video={video_frames.shape[0]} mask={mask_frames.shape[0]} expected={missing_frames}"
        )
    mask_frames = binarize_mask_array(mask_frames)
    if isinstance(context.video_frame_cache, TensorFrameCache):
        context.video_frame_cache.append_uint8(video_frames)
        context.record_runtime_transfer(
            "decoder_uint8_to_bf16_once",
            int(video_frames.shape[0] * 3 * video_frames.shape[1] * video_frames.shape[2] * 2),
        )
    else:
        context.video_frame_cache.append(video_frames)
    context.mask_frame_cache.append(mask_frames)
    if context.object_states:
        context.object_states[0].input_frontier = context.video_frame_cache.end_index
    context.streaming_state.loaded_until_frame = context.video_frame_cache.end_index
    context.window_history.append(
        {
            "event": "streaming_cache_load",
            "load_start": spec.load_start,
            "load_end": spec.load_end,
            "cache_start": context.video_frame_cache.start_index,
            "cache_end": context.video_frame_cache.end_index,
            "loaded_frames": missing_frames,
        }
    )
    record_runtime_event(
        context,
        "load",
        task_state=context.object_states[0] if context.object_states else None,
        load_start=spec.load_start,
        load_end=spec.load_end,
        cache_start=context.video_frame_cache.start_index,
        cache_end=context.video_frame_cache.end_index,
        loaded_frames=missing_frames,
    )
    context.load_event_history.append(
        {
            "load_start": spec.load_start,
            "load_end": spec.load_end,
            "cache_start": context.video_frame_cache.start_index,
            "cache_end": context.video_frame_cache.end_index,
            "loaded_frames": missing_frames,
        }
    )
    if spec.load_end > context.video_frame_cache.end_index:
        raise ValueError(
            f"window cache does not contain required frames [{spec.load_start}, {spec.load_end})"
        )


def evict_cache_before(
    *,
    context: LTX095EraseRuntimeContext,
    frame_index: int,
    record_runtime_event: Callable[..., dict],
) -> None:
    if context.window_runtime_mode != "streaming":
        return

    stable_index = max(int(frame_index), 0)
    final_cache = context.final_window_output_cache
    if final_cache is None:
        return

    for state in context.object_states[:-1]:
        if state.next_window_index < state.window_count:
            target = min(
                stable_index,
                state.window_specs[state.next_window_index].load_start,
                state.output_cache.end_index,
            )
        else:
            target = min(stable_index, state.output_cache.end_index)
        if target > state.output_cache.start_index:
            state.output_cache.pop_before(target)

    for state in context.object_states[1:]:
        if state.next_window_index < state.window_count:
            target = min(
                stable_index,
                state.window_specs[state.next_window_index].load_start,
                state.input_cache.end_index,
            )
        else:
            target = min(stable_index, state.input_cache.end_index)
        if target > state.input_cache.start_index:
            state.input_cache.pop_before(target)

    if context.video_frame_cache is not None:
        if context.object_states and context.object_states[0].next_window_index < context.object_states[0].window_count:
            source_target = min(
                stable_index,
                context.object_states[0].window_specs[
                    context.object_states[0].next_window_index
                ].load_start,
                context.video_frame_cache.end_index,
            )
        else:
            source_target = min(stable_index, context.video_frame_cache.end_index)
        if source_target > context.video_frame_cache.start_index:
            context.video_frame_cache.pop_before(source_target)
        if context.object_states:
            context.object_states[0].input_frontier = max(
                context.object_states[0].input_frontier,
                context.video_frame_cache.start_index,
            )

    context.window_history.append(
        {
            "event": "streaming_cache_evict",
            "evict_before": stable_index,
            "source_cache_start": (
                context.video_frame_cache.start_index
                if context.video_frame_cache is not None
                else None
            ),
            "mask_cache_start": (
                context.mask_frame_cache.start_index
                if context.mask_frame_cache is not None
                else None
            ),
            "mask_release_frontier": int(context.mask_release_frontier),
            "final_cache_start": (
                final_cache.start_index
                if final_cache is not None
                else None
            ),
        }
    )
    context.evict_event_history.append(
        {
            "evict_before": stable_index,
            "source_cache_start": (
                context.video_frame_cache.start_index
                if context.video_frame_cache is not None
                else None
            ),
            "mask_cache_start": (
                context.mask_frame_cache.start_index
                if context.mask_frame_cache is not None
                else None
            ),
            "mask_release_frontier": int(context.mask_release_frontier),
            "final_cache_start": (
                final_cache.start_index
                if final_cache is not None
                else None
            ),
        }
    )
    record_runtime_event(
        context,
        "evict",
        task_state=context.object_states[-1] if context.object_states else None,
        evict_before=stable_index,
        source_cache_start=(
            context.video_frame_cache.start_index
            if context.video_frame_cache is not None
            else None
        ),
        mask_cache_start=(
            context.mask_frame_cache.start_index
            if context.mask_frame_cache is not None
            else None
        ),
        mask_release_frontier=int(context.mask_release_frontier),
        final_cache_start=(final_cache.start_index if final_cache is not None else None),
    )


def flush_windowed_frames(
    *,
    context: LTX095EraseRuntimeContext,
    flush_end: int,
    record_runtime_event: Callable[..., dict],
    evict_cache_before_fn: Callable[..., None],
    release_mask_frames_fn: Callable[..., None],
) -> None:
    if not _is_windowed_runtime_mode(context.runtime_mode):
        return
    final_cache = context.final_window_output_cache or context.video_frame_cache
    if final_cache is None:
        return
    flush_end = min(flush_end, final_cache.end_index)
    if flush_end <= context.next_write_index:
        return
    flush_started = time.perf_counter()
    flush_history_item = None
    runtime_event_item = None
    written_frames = max(0, flush_end - context.next_write_index)
    if context.window_runtime_mode == "streaming":
        frames_to_write = final_cache.pop_before(flush_end)
        writer_duration_seconds = 0.0
        if frames_to_write.shape[0] > 0 and context.sequential_video_writer is not None:
            writer_started = time.perf_counter()
            if isinstance(frames_to_write, torch.Tensor):
                _write_streaming_frames(context=context, frames=frames_to_write)
            else:
                write_owned = getattr(
                    context.sequential_video_writer,
                    "write_frames_owned",
                    context.sequential_video_writer.write_frames,
                )
                write_owned(frames_to_write)
            writer_duration_seconds = time.perf_counter() - writer_started
            context.record_runtime_timing(
                "video_encode_write", writer_duration_seconds
            )
        flush_history_item = {
            "flush_start": context.next_write_index,
            "flush_end": flush_end,
            "written_frames": int(frames_to_write.shape[0]),
            "writer_active": context.sequential_video_writer is not None,
            "writer_duration_seconds": writer_duration_seconds,
        }
        context.window_history.append(
            {"event": "streaming_tail_flush", **flush_history_item}
        )
        context.flush_event_history.append(flush_history_item)
        runtime_event_item = record_runtime_event(
            context,
            "flush",
            task_state=context.object_states[-1] if context.object_states else None,
            **flush_history_item,
        )
    context.next_write_index = flush_end
    evict_cache_before_fn(context=context, frame_index=context.next_write_index)
    if context.window_runtime_mode == "streaming":
        release_mask_frames_fn(
            context=context,
            release_end=flush_end,
            source="streaming_tail_flush_after_evict",
        )
    flush_duration_seconds = time.perf_counter() - flush_started
    context.record_runtime_timing("streaming_flush", flush_duration_seconds)
    if flush_history_item is not None:
        flush_history_item["duration_seconds"] = flush_duration_seconds
    if runtime_event_item is not None:
        runtime_event_item["duration_seconds"] = flush_duration_seconds


def materialize_object_window_mask(
    *,
    context: LTX095EraseRuntimeContext,
    object_state: ObjectRuntimeState,
    spec: WindowSpec,
    ensure_window_cache_loaded_fn: Callable[..., None],
    crop_bbox: tuple[int, int, int, int] | None = None,
):
    _ = object_state
    return materialize_ltx095_window_mask(
        context=context,
        spec=spec,
        ensure_window_cache_loaded_fn=ensure_window_cache_loaded_fn,
        crop_bbox=crop_bbox,
    )
