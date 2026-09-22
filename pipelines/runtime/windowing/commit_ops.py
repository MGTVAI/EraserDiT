"""Window commit and skip kernel helpers for LTX095 pipelines.runtime."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch

from nodes.schedule_batch import Req
from pipelines.runtime.io.masks import consume_ltx095_window_mask
from pipelines.runtime.contracts import (
    LTX095EraseRuntimeContext,
    ObjectRuntimeState,
    _is_windowed_runtime_mode,
)
from media.video_io import (
    TensorFrameCache,
    frames_tensor_to_uint8,
)
from utils.windowing import WindowSpec

_WINDOW_COMMIT_CONVERSION_CHUNK_SIZE = 4


def splice_uint8_patch(
    source_frames: np.ndarray,
    patch_frames: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    source_is_exclusive: bool = False,
) -> tuple[np.ndarray, int]:
    """Splice a THWC uint8 patch without ever materializing a GPU source window."""

    if source_frames.ndim != 4 or source_frames.shape[-1] != 3:
        raise ValueError("source_frames must have shape [F,H,W,3]")
    if patch_frames.ndim != 4 or patch_frames.shape[-1] != 3:
        raise ValueError("patch_frames must have shape [F,H,W,3]")
    x, y, width, height = (int(value) for value in bbox)
    if (
        patch_frames.shape[0] != source_frames.shape[0]
        or patch_frames.shape[1:3] != (height, width)
    ):
        raise ValueError(
            "patch shape does not match bbox/source frames: "
            f"source={source_frames.shape} patch={patch_frames.shape} bbox={bbox}"
        )
    if x < 0 or y < 0 or x + width > source_frames.shape[2] or y + height > source_frames.shape[1]:
        raise ValueError(f"bbox {bbox} exceeds source shape {source_frames.shape}")
    if x == 0 and y == 0 and width == source_frames.shape[2] and height == source_frames.shape[1]:
        return patch_frames, 0
    result = source_frames if source_is_exclusive else source_frames.copy()
    result[:, y : y + height, x : x + width] = patch_frames
    return result, 0 if source_is_exclusive else int(source_frames.nbytes)


def splice_bf16_patch(
    source_frames: torch.Tensor,
    patch_frames: torch.Tensor,
    bbox: tuple[int, int, int, int],
) -> torch.Tensor:
    """Splice an FCHW patch into an exclusive CPU BF16 source buffer."""

    if source_frames.ndim != 4 or source_frames.shape[1] != 3:
        raise ValueError("source_frames must have shape [F,3,H,W]")
    if patch_frames.ndim != 4 or patch_frames.shape[1] != 3:
        raise ValueError("patch_frames must have shape [F,3,H,W]")
    if source_frames.dtype is not torch.bfloat16 or patch_frames.dtype is not torch.bfloat16:
        raise ValueError("BF16 object-chain splice requires bfloat16 tensors")
    x, y, width, height = (int(value) for value in bbox)
    if patch_frames.shape[0] != source_frames.shape[0] or tuple(
        patch_frames.shape[-2:]
    ) != (height, width):
        raise ValueError("patch shape does not match bbox/source frames")
    if x < 0 or y < 0 or x + width > source_frames.shape[-1] or y + height > source_frames.shape[-2]:
        raise ValueError(f"bbox {bbox} exceeds source shape {tuple(source_frames.shape)}")
    if x == 0 and y == 0 and width == source_frames.shape[-1] and height == source_frames.shape[-2]:
        return patch_frames
    source_frames[:, :, y : y + height, x : x + width].copy_(patch_frames)
    return source_frames


def blend_bf16_temporal_overlap(
    before: torch.Tensor,
    after: torch.Tensor,
    mode: str = "before",
) -> torch.Tensor:
    """Blend equal FCHW BF16 overlap tensors without crossing the uint8 boundary."""

    if before.shape != after.shape:
        raise ValueError("overlap tensors must share shape")
    if before.dtype is not torch.bfloat16 or after.dtype is not torch.bfloat16:
        raise ValueError("BF16 overlap blend requires bfloat16 tensors")
    normalized_mode = (mode or "before").strip().lower()
    if normalized_mode == "before":
        return before.clone()
    if normalized_mode == "after":
        return after.clone()
    overlap_len = int(before.shape[0])
    if overlap_len <= 1:
        return after.clone()
    if normalized_mode == "cosine":
        weights = torch.cos(torch.linspace(0, math.pi / 2, overlap_len))
    elif normalized_mode == "linear":
        weights = torch.linspace(1, 0, overlap_len)
    elif normalized_mode == "sigmoid":
        weights = torch.sigmoid(torch.linspace(6, -6, overlap_len))
    elif normalized_mode.startswith("value:"):
        ratio = float(normalized_mode.split(":", 1)[1].strip())
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("value fuse mode ratio must be within [0,1]")
        weights = torch.full((overlap_len,), ratio)
    else:
        raise ValueError(f"Unsupported overlap fuse mode: {mode}")
    weights = weights.to(dtype=torch.float32).view(-1, 1, 1, 1)
    return (
        weights * before.float() + (1.0 - weights) * after.float()
    ).to(dtype=torch.bfloat16)


def blend_uint8_temporal_overlap(
    before: np.ndarray,
    after: np.ndarray,
    mode: str = "before",
) -> np.ndarray:
    """Blend equal THWC uint8 overlap arrays with the established fuse weights."""

    if before.shape != after.shape:
        raise ValueError(
            f"overlap arrays must share shape, got {before.shape} and {after.shape}"
        )
    if before.dtype != np.uint8 or after.dtype != np.uint8:
        raise ValueError("overlap arrays must use uint8")
    normalized_mode = (mode or "before").strip().lower()
    if normalized_mode == "before":
        return before.copy()
    if normalized_mode == "after":
        return after.copy()
    overlap_len = int(before.shape[0])
    if overlap_len <= 1:
        return after.copy()
    if normalized_mode == "cosine":
        weights = np.cos(np.linspace(0, math.pi / 2, overlap_len, dtype=np.float32))
    elif normalized_mode == "linear":
        weights = np.linspace(1, 0, overlap_len, dtype=np.float32)
    elif normalized_mode == "sigmoid":
        values = np.linspace(6, -6, overlap_len, dtype=np.float32)
        weights = 1.0 / (1.0 + np.exp(-values))
    elif normalized_mode.startswith("value:"):
        ratio = float(normalized_mode.split(":", 1)[1].strip())
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(
                f"value fuse mode ratio must be within [0,1], got {ratio}"
            )
        weights = np.full((overlap_len,), ratio, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported overlap fuse mode: {mode}")
    weights = weights.reshape(-1, 1, 1, 1)
    blended = (
        weights * before.astype(np.float32)
        + (1.0 - weights) * after.astype(np.float32)
    )
    return np.rint(np.clip(blended, 0.0, 255.0)).astype(np.uint8)


def append_frames_tensor_to_cache(
    output_cache: Any,
    frames: torch.Tensor,
    *,
    chunk_size: int = _WINDOW_COMMIT_CONVERSION_CHUNK_SIZE,
) -> None:
    """Convert and append frames without materializing a full-window GPU temporary."""

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    for start in range(0, int(frames.shape[0]), chunk_size):
        output_cache.append(frames_tensor_to_uint8(frames[start : start + chunk_size]))


def blend_ltx095_temporal_overlap(
    before: torch.Tensor,
    after: torch.Tensor,
    mode: str = "before",
) -> torch.Tensor:
    if before.shape != after.shape:
        raise ValueError(
            f"overlap tensors must share shape, got {tuple(before.shape)} and {tuple(after.shape)}"
        )
    mode = (mode or "before").strip().lower()
    if mode == "before":
        return before
    if mode == "after":
        return after

    overlap_len = before.shape[2]
    if overlap_len <= 1:
        return after if mode != "before" else before

    if mode == "cosine":
        weights = torch.cos(
            torch.linspace(
                0, math.pi / 2, overlap_len, device=before.device, dtype=before.dtype
            )
        )
    elif mode == "linear":
        weights = torch.linspace(
            1, 0, overlap_len, device=before.device, dtype=before.dtype
        )
    elif mode == "sigmoid":
        weights = torch.sigmoid(
            torch.linspace(6, -6, overlap_len, device=before.device, dtype=before.dtype)
        )
    elif mode.startswith("value:"):
        ratio = float(mode.split(":", 1)[1].strip())
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(f"value fuse mode ratio must be within [0,1], got {ratio}")
        weights = torch.full(
            (overlap_len,), ratio, device=before.device, dtype=before.dtype
        )
    else:
        raise ValueError(f"Unsupported overlap fuse mode: {mode}")

    weights = weights.view(1, 1, -1, 1, 1)
    return weights * before + (1.0 - weights) * after


def commit_ltx095_window_to_object_output(
    *,
    context: LTX095EraseRuntimeContext,
    object_state: ObjectRuntimeState,
    spec: WindowSpec,
    window_batch: Req,
    append_passthrough_gap_fn,
    set_object_overlap_cache_fn,
    record_runtime_event_fn: Callable[..., dict[str, Any]],
    record_task_state_snapshot_fn: Callable[..., dict[str, Any]],
    update_window_state_fn,
) -> None:
    crop_patch_video = getattr(window_batch, "crop_video_modified", None)
    if crop_patch_video is None:
        raise ValueError("window crop patch is missing")

    crop_bbox = window_batch.crop_bbox
    if crop_bbox is None:
        raise ValueError("window crop bbox is missing")
    x, y, w, h = crop_bbox
    overlap_fuse_mode = str(window_batch.overlap_fuse_mode or "before")

    commit_start = spec.commit_start
    commit_end = spec.commit_end

    input_cache = object_state.input_cache
    output_cache = object_state.output_cache
    stable_start = spec.load_start
    stable_end = commit_end - spec.overlap_right
    stable_length = stable_end - stable_start

    crop_patch = crop_patch_video[:, :, : spec.input_len]
    crop_patch_4d = crop_patch[0].permute(1, 0, 2, 3).contiguous()
    bf16_object_chain = isinstance(input_cache, TensorFrameCache)
    if bf16_object_chain:
        patch_frames = crop_patch_4d.detach().to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous()
        context.record_runtime_transfer(
            "generated_patch_d2h",
            int(patch_frames.numel() * patch_frames.element_size()),
        )
        window_source_frames = None
        source_is_exclusive = True
    else:
        patch_frames = frames_tensor_to_uint8(crop_patch_4d)
        context.record_runtime_transfer("generated_patch_d2h", patch_frames.nbytes)
        window_source_frames = window_batch.extra.pop(
            "window_source_frames_uint8",
            None,
        )
        source_is_exclusive = window_source_frames is not None
    if window_source_frames is None:
        if (
            spec.overlap_left > 0
            and object_state.overlap_cache is not None
            and object_state.overlap_cache.start_index == spec.load_start
            and object_state.overlap_cache.end_index == spec.load_start + spec.overlap_left
        ):
            overlap_frames = object_state.overlap_cache.slice(
                spec.load_start,
                spec.load_start + spec.overlap_left,
            )
            if spec.load_start + spec.overlap_left < spec.load_end:
                new_frames = input_cache.slice(
                    spec.load_start + spec.overlap_left,
                    spec.load_end,
                )
                window_source_frames = (
                    torch.cat([overlap_frames, new_frames], dim=0)
                    if bf16_object_chain
                    else np.concatenate([overlap_frames, new_frames], axis=0)
                )
            else:
                window_source_frames = overlap_frames
        else:
            window_source_frames = input_cache.slice(spec.load_start, spec.load_end)
    if bf16_object_chain:
        valid_source = (
            isinstance(window_source_frames, torch.Tensor)
            and window_source_frames.dtype is torch.bfloat16
            and window_source_frames.ndim == 4
            and window_source_frames.shape[1] == 3
            and window_source_frames.shape[0] == spec.input_len
        )
    else:
        valid_source = (
            isinstance(window_source_frames, np.ndarray)
            and window_source_frames.dtype == np.uint8
            and window_source_frames.ndim == 4
            and window_source_frames.shape[0] == spec.input_len
        )
    if not valid_source:
        raise ValueError(
            "window source buffer does not match the active object-chain contract: "
            f"shape={getattr(window_source_frames, 'shape', None)} "
            f"dtype={getattr(window_source_frames, 'dtype', None)} "
            f"input_len={spec.input_len}"
        )

    original_overlap_frames = None
    if spec.overlap_left > 0:
        original_overlap_frames = (
            window_source_frames[: spec.overlap_left].clone()
            if bf16_object_chain
            else window_source_frames[: spec.overlap_left].copy()
        )

    if bf16_object_chain:
        window_modified_frames = splice_bf16_patch(
            window_source_frames, patch_frames, (x, y, w, h)
        )
        context.record_runtime_transfer("cpu_source_splice", 0)
    else:
        window_modified_frames, splice_bytes = splice_uint8_patch(
            window_source_frames,
            patch_frames,
            (x, y, w, h),
            source_is_exclusive=source_is_exclusive,
        )
        context.record_runtime_transfer("cpu_source_splice", splice_bytes)
    if source_is_exclusive:
        context.record_runtime_transfer(
            "exclusive_buffer_transfer",
            int(window_source_frames.numel() * window_source_frames.element_size())
            if bf16_object_chain
            else int(window_source_frames.nbytes),
        )

    if spec.overlap_left > 0:
        assert original_overlap_frames is not None
        window_modified_frames[: spec.overlap_left] = (
            blend_bf16_temporal_overlap(
                before=original_overlap_frames,
                after=window_modified_frames[: spec.overlap_left],
                mode=overlap_fuse_mode,
            )
            if bf16_object_chain
            else blend_uint8_temporal_overlap(
                before=original_overlap_frames,
                after=window_modified_frames[: spec.overlap_left],
                mode=overlap_fuse_mode,
            )
        )
        context.record_runtime_transfer(
            "overlap_blend",
            int(
                window_modified_frames[: spec.overlap_left].numel()
                * window_modified_frames.element_size()
            )
            if bf16_object_chain
            else int(window_modified_frames[: spec.overlap_left].nbytes),
        )

    passthrough_gap_start = stable_start
    passthrough_gap_end = stable_start
    if output_cache.end_index > stable_start:
        raise ValueError(
            f"Object output cache for object {object_state.object_index} "
            f"advanced beyond stable start {stable_start}: {output_cache.end_index}"
        )
    if output_cache.end_index < stable_start:
        passthrough_gap_start, passthrough_gap_end = append_passthrough_gap_fn(
            output_cache,
            input_cache,
            stable_start,
        )
    passthrough_gap_length = max(0, passthrough_gap_end - passthrough_gap_start)
    if passthrough_gap_length > 0:
        object_state.emit_pop_raw_input(passthrough_gap_start, passthrough_gap_length)
        record_runtime_event_fn(
            context,
            "raw_pop",
            task_state=object_state,
            window_index=spec.window_index,
            start_index=passthrough_gap_start,
            length=passthrough_gap_length,
            source="passthrough_gap",
        )
    if output_cache.end_index != stable_start:
        raise ValueError(
            f"Object output cache for object {object_state.object_index} "
            f"expected stable append index {stable_start}, got {output_cache.end_index}"
        )

    if stable_length > 0:
        stable_frames = window_modified_frames[:stable_length]
        if bf16_object_chain:
            stable_frames = stable_frames.clone()
        append_owned = getattr(output_cache, "append_owned", None)
        if append_owned is None:
            output_cache.append(stable_frames)
            cache_append_bytes = int(stable_frames.nbytes)
        else:
            cache_append_bytes = int(append_owned(stable_frames))
        context.record_runtime_transfer(
            "cache_append",
            cache_append_bytes,
        )
        context.record_runtime_transfer(
            "cache_append_transfer",
            int(stable_frames.numel() * stable_frames.element_size())
            if bf16_object_chain
            else int(stable_frames.nbytes),
        )
        object_state.emit_pop_modified(stable_start, stable_length)
        record_runtime_event_fn(
            context,
            "modified_pop",
            task_state=object_state,
            window_index=spec.window_index,
            start_index=stable_start,
            length=stable_length,
            source="window_commit",
        )

    overlap_frames = None
    if spec.overlap_right > 0:
        overlap_start_offset = stable_end - spec.load_start
        overlap_end_offset = commit_end - spec.load_start
        overlap_frames = window_modified_frames[
            overlap_start_offset:overlap_end_offset
        ]
        overlap_frames = (
            overlap_frames.clone() if bf16_object_chain else overlap_frames.copy()
        )
    set_object_overlap_cache_fn(object_state, stable_end, overlap_frames)
    consume_ltx095_window_mask(
        context=context,
        spec=spec,
        crop_bbox=crop_bbox,
    )

    object_state.scene_index = spec.scene_index
    object_state.flush_frontier = stable_end
    update_window_state_fn(
        context,
        object_state.object_index,
        spec.window_index,
        status="committed",
        scene_index=spec.scene_index,
        commit_start=commit_start,
        commit_end=commit_end,
        stable_end=stable_end,
        skip=False,
    )
    context.window_history.append(
        {
            "object_index": object_state.object_index,
            "window_index": spec.window_index,
            "scene_index": spec.scene_index,
            "start_index": spec.start_index,
            "end_index": spec.end_index,
            "load_start": spec.load_start,
            "load_end": spec.load_end,
            "deal_start": spec.deal_start,
            "deal_length": spec.deal_length,
            "load_length": spec.load_length,
            "future_length": spec.future_length,
            "stable_start": stable_start,
            "commit_start": commit_start,
            "commit_end": commit_end,
            "stable_end": stable_end,
            "overlap_left": spec.overlap_left,
            "overlap_right": spec.overlap_right,
            "overlap_fuse_mode": overlap_fuse_mode,
            "skip": False,
            "bbox": crop_bbox,
            "prompt": window_batch.prompt,
            "negative_prompt": window_batch.negative_prompt,
            "mask_policy": "consumed_shared_input_stream",
            "output_shape": tuple(crop_patch_video.shape),
            "materialized_output_shape": None,
            "passthrough_gap_start": passthrough_gap_start,
            "passthrough_gap_end": passthrough_gap_end,
            "forward_policy": "stable_output_plus_overlap_cache",
            "input_role": object_state.input_role,
            "output_cache_end": output_cache.end_index,
        }
    )
    record_runtime_event_fn(
        context,
        "window_commit",
        task_state=object_state,
        window_index=spec.window_index,
        stable_start=stable_start,
        stable_end=stable_end,
        passthrough_gap_start=passthrough_gap_start,
        passthrough_gap_end=passthrough_gap_end,
        overlap_right=spec.overlap_right,
        overlap_fuse_mode=overlap_fuse_mode,
    )
    record_task_state_snapshot_fn(
        context,
        object_state,
        phase="committed",
        window_index=spec.window_index,
        stable_end=stable_end,
    )


def record_ltx095_skipped_object_window(
    *,
    context: LTX095EraseRuntimeContext,
    object_state: ObjectRuntimeState,
    spec: WindowSpec,
    reason: str,
    prompt: Any,
    negative_prompt: Any,
    append_passthrough_gap_fn,
    set_object_overlap_cache_fn,
    record_runtime_event_fn: Callable[..., dict[str, Any]],
    record_task_state_snapshot_fn: Callable[..., dict[str, Any]],
    update_window_state_fn,
) -> None:
    object_state.skip_count += 1
    stable_start = spec.load_start
    stable_end = spec.commit_end - spec.overlap_right
    passthrough_gap_start = stable_start
    passthrough_gap_end = stable_start

    if object_state.output_cache.end_index > stable_start:
        raise ValueError(
            f"Object output cache for object {object_state.object_index} "
            f"advanced beyond stable start {stable_start}: {object_state.output_cache.end_index}"
        )
    if object_state.output_cache.end_index < stable_start:
        passthrough_gap_start, passthrough_gap_end = append_passthrough_gap_fn(
            object_state.output_cache,
            object_state.input_cache,
            stable_start,
        )
    passthrough_gap_length = max(0, passthrough_gap_end - passthrough_gap_start)
    if passthrough_gap_length > 0:
        object_state.emit_pop_raw_input(passthrough_gap_start, passthrough_gap_length)
        record_runtime_event_fn(
            context,
            "raw_pop",
            task_state=object_state,
            window_index=spec.window_index,
            start_index=passthrough_gap_start,
            length=passthrough_gap_length,
            source="skip_passthrough_gap",
        )
    if object_state.output_cache.end_index != stable_start:
        raise ValueError(
            f"Object output cache for object {object_state.object_index} "
            f"expected stable append index {stable_start}, got {object_state.output_cache.end_index}"
        )
    if stable_end > stable_start:
        object_state.output_cache.append(
            object_state.input_cache.slice(stable_start, stable_end)
        )
        object_state.emit_pop_raw_input(stable_start, stable_end - stable_start)
        record_runtime_event_fn(
            context,
            "raw_pop",
            task_state=object_state,
            window_index=spec.window_index,
            start_index=stable_start,
            length=stable_end - stable_start,
            source="window_skip",
        )
    overlap_frames = None
    if spec.overlap_right > 0:
        overlap_frames = object_state.input_cache.slice(stable_end, spec.commit_end)
    set_object_overlap_cache_fn(object_state, stable_end, overlap_frames)
    object_state.scene_index = spec.scene_index
    object_state.flush_frontier = stable_end
    update_window_state_fn(
        context,
        object_state.object_index,
        spec.window_index,
        status="skipped",
        scene_index=spec.scene_index,
        commit_start=spec.commit_start,
        commit_end=spec.commit_end,
        stable_end=stable_end,
        skip=True,
        skip_reason=reason,
        passthrough=True,
    )
    context.window_history.append(
        {
            "object_index": object_state.object_index,
            "window_index": spec.window_index,
            "scene_index": spec.scene_index,
            "start_index": spec.start_index,
            "end_index": spec.end_index,
            "load_start": spec.load_start,
            "load_end": spec.load_end,
            "deal_start": spec.deal_start,
            "deal_length": spec.deal_length,
            "load_length": spec.load_length,
            "future_length": spec.future_length,
            "stable_start": stable_start,
            "commit_start": spec.commit_start,
            "commit_end": spec.commit_end,
            "stable_end": stable_end,
            "overlap_left": spec.overlap_left,
            "overlap_right": spec.overlap_right,
            "overlap_fuse_mode": "before",
            "skip": True,
            "skip_reason": reason,
            "bbox": None,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "mask_policy": "immutable_shared_input_stream",
            "output_shape": None,
            "passthrough_gap_start": passthrough_gap_start,
            "passthrough_gap_end": passthrough_gap_end,
            "forward_policy": "stable_passthrough_plus_overlap_cache",
            "input_role": object_state.input_role,
            "output_cache_end": object_state.output_cache.end_index,
        }
    )
    record_runtime_event_fn(
        context,
        "window_skip",
        task_state=object_state,
        window_index=spec.window_index,
        stable_start=stable_start,
        stable_end=stable_end,
        skip_reason=reason,
    )
    record_task_state_snapshot_fn(
        context,
        object_state,
        phase="skipped",
        window_index=spec.window_index,
        stable_end=stable_end,
        skip_reason=reason,
    )


def commit_ltx095_window(
    *,
    context: LTX095EraseRuntimeContext,
    spec: WindowSpec,
    window_batch: Req,
    object_index: int,
) -> None:
    crop_patch_video = getattr(window_batch, "crop_video_modified", None)
    if crop_patch_video is None:
        raise ValueError("window crop patch is missing")

    commit_start = spec.commit_start
    commit_end = spec.commit_end
    local_start = spec.active_start_offset
    local_end = min(spec.active_end_offset, spec.input_len)
    overlap_fuse_mode = str(window_batch.overlap_fuse_mode or "before")
    crop_bbox = window_batch.crop_bbox
    if crop_bbox is None:
        raise ValueError("window crop bbox is missing")
    x, y, w, h = crop_bbox
    crop_patch = crop_patch_video[:, :, : spec.input_len]
    crop_patch_4d = crop_patch[0].permute(1, 0, 2, 3).contiguous()

    if spec.overlap_left > 0:
        if _is_windowed_runtime_mode(context.runtime_mode):
            if context.video_frame_cache is None:
                raise ValueError(f"{context.runtime_mode} runtime missing frame cache")
            before_overlap_4d = frames_uint8_to_tensor(
                context.video_frame_cache.slice(
                    spec.load_start,
                    spec.load_start + spec.overlap_left,
                )
            ).contiguous()
        else:
            assert context.working_video is not None
            before_overlap = context.working_video[
                :, :, spec.load_start : spec.load_start + spec.overlap_left
            ]
            before_overlap_4d = before_overlap[0].permute(1, 0, 2, 3).contiguous()
        after_overlap_4d = before_overlap_4d.clone()
        overlap_patch_4d = crop_patch_4d[: spec.overlap_left]
        splice_video_inplace(after_overlap_4d, overlap_patch_4d, (x, y))
        fused_overlap_4d = blend_ltx095_temporal_overlap(
            before=before_overlap_4d,
            after=after_overlap_4d,
            mode=overlap_fuse_mode,
        )
        fused_overlap = fused_overlap_4d.permute(1, 0, 2, 3).unsqueeze(0)
    if _is_windowed_runtime_mode(context.runtime_mode):
        if context.video_frame_cache is None or context.mask_frame_cache is None:
            raise ValueError(f"{context.runtime_mode} runtime missing frame caches")
        if spec.overlap_left > 0:
            context.video_frame_cache.overwrite(
                spec.load_start,
                frames_tensor_to_uint8(fused_overlap_4d),
            )
        commit_base = window_batch.video[:, :, local_start:local_end].clone()
        commit_base_4d = commit_base[0].permute(1, 0, 2, 3).contiguous()
        commit_patch_4d = crop_patch_4d[local_start:local_end]
        splice_video_inplace(commit_base_4d, commit_patch_4d, (x, y))
        context.video_frame_cache.overwrite(
            commit_start,
            frames_tensor_to_uint8(commit_base_4d),
        )
    else:
        assert context.working_video is not None
        assert context.final_video is not None
        if spec.overlap_left > 0:
            context.working_video[
                :, :, spec.load_start : spec.load_start + spec.overlap_left
            ] = fused_overlap.to(
                device=context.working_video.device,
                dtype=context.working_video.dtype,
            )
            context.final_video[
                :, :, spec.load_start : spec.load_start + spec.overlap_left
            ] = fused_overlap.to(
                device=context.final_video.device,
                dtype=context.final_video.dtype,
            )
        commit_patch = crop_patch[:, :, local_start:local_end]
        context.working_video[:, :, commit_start:commit_end, y : y + h, x : x + w] = (
            commit_patch.to(
                device=context.working_video.device,
                dtype=context.working_video.dtype,
            )
        )
        context.final_video[:, :, commit_start:commit_end, y : y + h, x : x + w] = (
            commit_patch.to(
                device=context.final_video.device,
                dtype=context.final_video.dtype,
            )
        )
    consume_ltx095_window_mask(
        context=context,
        spec=spec,
        crop_bbox=crop_bbox,
    )

    context.window_history.append(
        {
            "object_index": object_index,
            "window_index": spec.window_index,
            "scene_index": spec.scene_index,
            "start_index": spec.start_index,
            "end_index": spec.end_index,
            "load_start": spec.load_start,
            "load_end": spec.load_end,
            "deal_start": spec.deal_start,
            "deal_length": spec.deal_length,
            "load_length": spec.load_length,
            "future_length": spec.future_length,
            "commit_start": commit_start,
            "commit_end": commit_end,
            "overlap_left": spec.overlap_left,
            "overlap_right": spec.overlap_right,
            "overlap_fuse_mode": overlap_fuse_mode,
            "skip": False,
            "bbox": crop_bbox,
            "prompt": window_batch.prompt,
            "negative_prompt": window_batch.negative_prompt,
            "mask_policy": "consumed_shared_input_stream",
            "output_shape": tuple(crop_patch_video.shape),
            "materialized_output_shape": None,
        }
    )


def record_ltx095_skipped_window(
    *,
    context: LTX095EraseRuntimeContext,
    spec: WindowSpec,
    object_index: int,
    reason: str,
    prompt: Any,
    negative_prompt: Any,
) -> None:
    context.window_history.append(
        {
            "object_index": object_index,
            "window_index": spec.window_index,
            "scene_index": spec.scene_index,
            "start_index": spec.start_index,
            "end_index": spec.end_index,
            "load_start": spec.load_start,
            "load_end": spec.load_end,
            "deal_start": spec.deal_start,
            "deal_length": spec.deal_length,
            "load_length": spec.load_length,
            "future_length": spec.future_length,
            "commit_start": spec.commit_start,
            "commit_end": spec.commit_end,
            "overlap_left": spec.overlap_left,
            "overlap_right": spec.overlap_right,
            "overlap_fuse_mode": "before",
            "skip": True,
            "skip_reason": reason,
            "bbox": None,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "output_shape": None,
        }
    )
