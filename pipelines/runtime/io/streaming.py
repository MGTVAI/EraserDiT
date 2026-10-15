"""Streaming runtime IO and lifecycle helpers for pipelines.runtime."""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from pipelines.runtime.events import record_runtime_event

from pipelines.runtime.contracts import EraseRuntimeContext
from pipelines.runtime.contracts import ObjectRuntimeState
from pipelines.runtime.contracts import _is_windowed_runtime_mode
from pipelines.runtime.io.masks import materialize_window_mask
from utils.video_io import (
    TensorFrameCache,
    ChunkedFrameCache,
    binarize_mask_array,
    threshold_mask_array,
    frames_tensor_to_uint8,
    read_paired_video_frames,
)
from utils.windowing import WindowSpec


def _write_streaming_frames(
    *,
    context: EraseRuntimeContext,
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


def _discard_cache_before(context, cache, end_frame: int) -> int:
    released_count = max(0, end_frame - cache.start_index)
    if os.environ.get("MGERASE_STREAMING_COPY_OPT") == "1" and isinstance(cache, ChunkedFrameCache):
        copied_bytes = cache.discard_before(end_frame)
        context.record_runtime_transfer("cache_eviction_internal_copy", copied_bytes)
    else:
        cache.pop_before(end_frame)
    return released_count


def release_mask_frames(
    context: EraseRuntimeContext,
    release_end: int,
    source: str,
) -> None:
    if context.mask_frame_cache is None:
        return
    release_start = int(context.mask_release_frontier)
    bounded_release_end = min(int(release_end), context.mask_frame_cache.end_index)
    if bounded_release_end <= release_start:
        return
    released_count = _discard_cache_before(context, context.mask_frame_cache, bounded_release_end)
    context.mask_release_frontier = bounded_release_end
    history_item = {
        "release_start": release_start,
        "release_end": bounded_release_end,
        "released_frames": released_count,
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
    context: EraseRuntimeContext,
    spec: WindowSpec,
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

    read_threads = os.environ.get("MGERASE_PIPE_READ_THREADS", "1")
    if read_threads not in {"1", "2"}:
        raise ValueError("MGERASE_PIPE_READ_THREADS must be 1 or 2")
    video_frames, mask_frames, read_stats = read_paired_video_frames(
        context.sequential_video_reader, context.sequential_mask_reader,
        missing_frames, parallel=read_threads == "2",
    )
    context.record_runtime_timing("reader_pair_wall", read_stats["wall_seconds"])
    context.record_runtime_timing("video_reader_wall", read_stats["video_seconds"])
    context.record_runtime_timing("mask_reader_wall", read_stats["mask_seconds"])
    context.record_runtime_transfer("video_decoder_pipe", read_stats["video_pipe_bytes"])
    context.record_runtime_transfer("mask_decoder_pipe", read_stats["mask_pipe_bytes"])
    context.window_history.append({"event": "reader_pair", **read_stats})
    if video_frames.shape[0] != missing_frames or mask_frames.shape[0] != missing_frames:
        raise RuntimeError(
            f"windowed_streaming reader returned mismatched frame counts: "
            f"video={video_frames.shape[0]} mask={mask_frames.shape[0]} expected={missing_frames}"
        )
    # Exact uint8 mode uses the same RGB decode and first-channel mask
    # contract as read_mask_array. Gray conversion can shift threshold edges.
    if mask_frames.ndim == 4:
        mask_frames = mask_frames[..., 0]
    threshold_inplace_requested = os.environ.get("MGERASE_MASK_THRESHOLD_INPLACE") == "1"
    threshold_inplace_applied = False
    if context.streaming_mask_threshold is None:
        mask_frames = binarize_mask_array(mask_frames)
    else:
        # This freshly read buffer has not yet been published to either cache.
        threshold_inplace_applied = (
            threshold_inplace_requested and mask_frames.dtype == np.uint8
            and mask_frames.flags.writeable
        )
        mask_frames = threshold_mask_array(
            mask_frames, context.streaming_mask_threshold,
            inplace=threshold_inplace_applied,
        )
    context.window_history.append({
        "event": "mask_threshold",
        "window_index": spec.window_index,
        "requested": threshold_inplace_requested,
        "applied": threshold_inplace_applied,
        "threshold": context.streaming_mask_threshold,
        "frames": int(mask_frames.shape[0]),
        "modeled_full_size_temporary_array_bytes": (
            None if context.streaming_mask_threshold is None else
            0 if threshold_inplace_applied else 2 * int(mask_frames.size)
        ),
    })
    if isinstance(context.video_frame_cache, TensorFrameCache):
        context.video_frame_cache.append_uint8(video_frames)
        context.record_runtime_transfer(
            "decoder_uint8_to_bf16_once",
            int(video_frames.shape[0] * 3 * video_frames.shape[1] * video_frames.shape[2] * 2),
        )
    else:
        if os.environ.get("MGERASE_STREAMING_COPY_OPT") == "1" and isinstance(context.video_frame_cache, ChunkedFrameCache):
            copied = context.video_frame_cache.append_owned(video_frames)
        else:
            context.video_frame_cache.append(video_frames)
            copied = int(video_frames.nbytes)
        context.record_runtime_transfer("video_decoder_cache_copy", copied)
    if os.environ.get("MGERASE_STREAMING_COPY_OPT") == "1" and isinstance(context.mask_frame_cache, ChunkedFrameCache):
        copied = context.mask_frame_cache.append_owned(mask_frames)
    else:
        context.mask_frame_cache.append(mask_frames)
        copied = int(mask_frames.nbytes)
    context.record_runtime_transfer("mask_decoder_cache_copy", copied)
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
    context: EraseRuntimeContext,
    frame_index: int,
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
            _discard_cache_before(context, state.output_cache, target)

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
            _discard_cache_before(context, state.input_cache, target)

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
            _discard_cache_before(context, context.video_frame_cache, source_target)
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
    context: EraseRuntimeContext,
    flush_end: int,
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
        writer = context.sequential_video_writer
        handoff_optimized = (
            os.environ.get("MGERASE_WRITER_HANDOFF_OPT") == "1"
            and isinstance(final_cache, ChunkedFrameCache)
            and final_cache.dtype == np.dtype(np.uint8)
            and (writer is None or hasattr(writer, "write_frame_blocks_owned"))
        )
        writer_duration_seconds = 0.0
        writer_block_count = 0
        handoff_internal_copy_bytes = None
        if handoff_optimized:
            blocks, handoff_internal_copy_bytes = final_cache.pop_before_owned(flush_end)
            written_frames = sum(int(block.shape[0]) for block in blocks)
            writer_block_count = len(blocks)
            context.record_runtime_transfer("writer_handoff_internal_copy", handoff_internal_copy_bytes)
            if blocks and writer is not None:
                writer_started = time.perf_counter()
                writer.write_frame_blocks_owned(blocks)
                writer_duration_seconds = time.perf_counter() - writer_started
                context.record_runtime_timing("video_encode_write", writer_duration_seconds)
        else:
            frames_to_write = final_cache.pop_before(flush_end)
            written_frames = int(frames_to_write.shape[0])
            writer_block_count = int(written_frames > 0)
        if not handoff_optimized and written_frames > 0 and writer is not None:
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
            "written_frames": written_frames,
            "writer_active": context.sequential_video_writer is not None,
            "writer_handoff_optimized": handoff_optimized,
            "writer_block_count": writer_block_count,
            "writer_handoff_internal_copy_bytes": handoff_internal_copy_bytes,
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
    evict_cache_before(context=context, frame_index=context.next_write_index)
    if context.window_runtime_mode == "streaming":
        release_mask_frames(
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
    context: EraseRuntimeContext,
    object_state: ObjectRuntimeState,
    spec: WindowSpec,
    crop_bbox: tuple[int, int, int, int] | None = None,
):
    ensure_window_cache_loaded(context, spec)
    return materialize_window_mask(
        context=context,
        spec=spec,
        crop_bbox=crop_bbox,
    )
