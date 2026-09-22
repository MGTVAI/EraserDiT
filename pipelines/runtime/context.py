"""Runtime context bootstrap for the LTX095 videoerase pipeline."""

from __future__ import annotations

import os
import struct
import time
from dataclasses import asdict
from typing import Any, Callable

import numpy as np
import torch

from config.ltx095 import LTX095EraseSamplingParams
from config.server_args import ServerArgs
from parallel.runtime import resolve_ltx095_native_vae_parallel_status
from nodes.schedule_batch import Req
from distributed.group_coordinator import GroupCoordinator
from pipelines.runtime.windowing.sp_dispatch import (
    LTX095ActiveSPWindowContext,
    LTX095SPWindowControlError,
    resolve_active_ltx095_window_commit_context,
)
from pipelines.runtime.scheduler import RuntimeTaskScheduler
from pipelines.runtime.tracks import _load_bbox_tracks
from pipelines.runtime.contracts import (
    LTX095EraseRuntimeContext,
    WindowedPreloadRuntimeState,
    WindowedStreamingRuntimeState,
    _resolve_runtime_mode,
)
from memory.policies.memory_phase_controller import MemoryPhaseController
from nodes.control import service_checkpoint
from utils.runtime_progress import create_runtime_progress
from media.encoding import VideoEncodingProfile
from media.video_io import (
    ArrayFrameCache,
    ChunkedFrameCache,
    SequentialVideoReader,
    SequentialVideoWriter,
    TensorFrameCache,
    WindowedVideoStore,
)

_VIDEO_METADATA_PROTOCOL_VERSION = 1
_VIDEO_METADATA_STATUS_OK = 1
_VIDEO_METADATA_STATUS_ERROR = 2
_VIDEO_METADATA_HEADER_SIZE = 6


def _float64_to_int64_bits(value: float) -> int:
    return struct.unpack("!q", struct.pack("!d", value))[0]


def _int64_bits_to_float64(value: int) -> float:
    return struct.unpack("!d", struct.pack("!q", value))[0]


def _validate_sp_video_metadata(
    metadata: dict[str, Any],
    *,
    fallback_fps: float,
) -> tuple[int, int, int, float]:
    normalized: list[int] = []
    for name in ("height", "width", "num_frames"):
        value = metadata.get(name)
        if type(value) is not int:
            try:
                value = int(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"video metadata {name} must be a positive int"
                ) from error
        if value <= 0 or value > 2**31 - 1:
            raise ValueError(f"video metadata {name} must be a positive int")
        normalized.append(value)
    fps = float(metadata.get("fps") or fallback_fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("video metadata fps must be a positive finite float")
    return normalized[0], normalized[1], normalized[2], fps


def _encode_sp_video_metadata(
    metadata: dict[str, Any],
    *,
    fallback_fps: float,
    device: torch.device,
) -> torch.Tensor:
    height, width, num_frames, fps = _validate_sp_video_metadata(
        metadata,
        fallback_fps=fallback_fps,
    )
    return torch.tensor(
        [
            _VIDEO_METADATA_PROTOCOL_VERSION,
            _VIDEO_METADATA_STATUS_OK,
            height,
            width,
            num_frames,
            _float64_to_int64_bits(fps),
        ],
        dtype=torch.int64,
        device=device,
    )


def _decode_sp_video_metadata(
    header: torch.Tensor,
) -> tuple[int, int, int, float] | None:
    if not isinstance(header, torch.Tensor):
        raise TypeError("video metadata header must be a tensor")
    if header.dtype is not torch.int64 or header.ndim != 1:
        raise TypeError("video metadata header must be a one-dimensional int64 tensor")
    if header.numel() != _VIDEO_METADATA_HEADER_SIZE:
        raise ValueError("video metadata header has an invalid fixed width")
    values = header.detach().to(device="cpu").tolist()
    if values[0] != _VIDEO_METADATA_PROTOCOL_VERSION:
        raise ValueError("unsupported video metadata protocol version")
    if values[1] == _VIDEO_METADATA_STATUS_ERROR:
        return None
    if values[1] != _VIDEO_METADATA_STATUS_OK:
        raise ValueError("video metadata status code is invalid")
    metadata = {
        "height": values[2],
        "width": values[3],
        "num_frames": values[4],
        "fps": _int64_bits_to_float64(values[5]),
    }
    return _validate_sp_video_metadata(metadata, fallback_fps=25.0)


def _synchronize_sp_metadata_error(
    error: BaseException | None,
    server_args: ServerArgs,
) -> None:
    from pipelines.runtime.windowing.commit_sync import (
        synchronize_ltx095_window_runtime_boundary,
    )

    synchronize_ltx095_window_runtime_boundary(error, server_args)


def _apply_baseline_output_contract(
    profile: "VideoEncodingProfile",
    video_metadata: dict[str, Any],
) -> "VideoEncodingProfile":
    """Rewrite the writer profile to the EraserDiT baseline's output contract.

    ``inference.py`` picks ``libx264`` / ``yuv420p`` and a bit rate of
    ``bit_rate // 1000000`` M; the shared profile builder instead reuses the raw
    input bit rate string and the source pixel format, which changes the encoder
    enough to move the decoded-pixel comparison by ~0.4 dB (plan §4.10).
    """
    from dataclasses import replace

    raw_bit_rate = video_metadata.get("bit_rate")
    try:
        megabits = max(1, int(raw_bit_rate) // 1000000)
    except (TypeError, ValueError):
        megabits = 10
    return replace(
        profile,
        encoder="libx264",
        output_pix_fmt="yuv420p",
        video_bitrate=f"{megabits}M",
        color_space=None,
        color_transfer=None,
        color_primaries=None,
        color_range=None,
        field_order="progressive",
    )


def _read_and_broadcast_sp_video_metadata(
    *,
    active: LTX095ActiveSPWindowContext,
    video_path: str,
    fallback_fps: float,
    server_args: ServerArgs,
    read_video_metadata: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    writer_metadata: dict[str, Any] | None = None
    local_error: BaseException | None = None
    if active.is_writer:
        try:
            writer_metadata = read_video_metadata(video_path)
            header = _encode_sp_video_metadata(
                writer_metadata,
                fallback_fps=fallback_fps,
                device=active.control_device,
            )
        except BaseException as error:
            local_error = error
            header = torch.tensor(
                [
                    _VIDEO_METADATA_PROTOCOL_VERSION,
                    _VIDEO_METADATA_STATUS_ERROR,
                    0,
                    0,
                    0,
                    0,
                ],
                dtype=torch.int64,
                device=active.control_device,
            )
    else:
        header = torch.zeros(
            _VIDEO_METADATA_HEADER_SIZE,
            dtype=torch.int64,
            device=active.control_device,
        )

    try:
        GroupCoordinator(active.group).broadcast(
            header,
            src=active.contract.writer_rank,
        )
    except BaseException as error:
        raise LTX095SPWindowControlError(
            rank=active.parallel_context.global_rank,
            phase="video metadata broadcast",
            original=error,
        ) from error

    decode_error: BaseException | None = None
    decoded = None
    try:
        decoded = _decode_sp_video_metadata(header)
    except BaseException as error:
        decode_error = error
    if decode_error is not None:
        _synchronize_sp_metadata_error(decode_error, server_args)
        raise decode_error
    if decoded is None:
        _synchronize_sp_metadata_error(local_error, server_args)
        raise LTX095SPWindowControlError(
            rank=active.parallel_context.global_rank,
            phase="video metadata preparation",
            original=RuntimeError("writer failed to prepare video metadata"),
        )

    height, width, num_frames, fps = decoded
    metadata = dict(writer_metadata or {})
    metadata.update(
        height=height,
        width=width,
        num_frames=num_frames,
        fps=fps,
    )
    return metadata


def _window_cache_impl_name(cache: ArrayFrameCache | ChunkedFrameCache | TensorFrameCache | None) -> str:
    if cache is None:
        return "none"
    return type(cache).__name__


def _resolve_ffmpeg_thread_count(batch: Req) -> object:
    cpu_resources = batch.extra.get("cpu_resources")
    if isinstance(cpu_resources, dict) and "ffmpeg_threads" in cpu_resources:
        return cpu_resources["ffmpeg_threads"]
    environment_value = os.environ.get("MGERASE_FFMPEG_THREADS", "4")
    if environment_value.isdecimal():
        return int(environment_value)
    return environment_value


def prepare_ltx095_runtime_context(
    *,
    batch: Req,
    params: LTX095EraseSamplingParams,
    server_args: ServerArgs,
    resource_policy: Any,
    build_runtime_video: Callable[
        [torch.Tensor | str], tuple[torch.Tensor, dict[str, object]]
    ],
    build_runtime_mask: Callable[[torch.Tensor | str, int], torch.Tensor],
    read_video_metadata: Callable[[str], dict[str, Any]],
    read_video_array: Callable[[str], tuple[np.ndarray, Any] | np.ndarray],
    read_mask_array: Callable[[str], tuple[np.ndarray, Any] | np.ndarray],
    window_store_builder: Callable[[str], WindowedVideoStore],
    memory_adapter: Any | None = None,
) -> LTX095EraseRuntimeContext:
    service_checkpoint(batch, server_args, phase="runtime_context_start")
    distributed_context = getattr(server_args, "distributed_context", None)
    distributed_metadata = (
        distributed_context.as_dict() if distributed_context is not None else {}
    )
    official_parallel_context = getattr(server_args, "official_parallel_context", None)
    official_parallel_metadata = (
        official_parallel_context.as_dict()
        if official_parallel_context is not None
        else {}
    )
    (
        native_vae_parallel_enabled,
        native_vae_parallel_degree,
        native_vae_parallel_mode,
    ) = resolve_ltx095_native_vae_parallel_status(server_args)
    official_parallel_metadata.update(
        {
            "ltx095_native_vae_parallel_enabled": native_vae_parallel_enabled,
            "ltx095_native_vae_parallel_degree": native_vae_parallel_degree,
            "ltx095_native_vae_parallel_mode": native_vae_parallel_mode,
        }
    )
    resource_policy_dict = resource_policy.as_dict()

    active_sp_context = None
    if str(params.runtime_mode or "").strip().lower() == "windowed_streaming":
        active_sp_context = resolve_active_ltx095_window_commit_context(server_args)
    sp_writer_owned_runtime = active_sp_context is not None
    if active_sp_context is not None:
        distributed_writer = bool(distributed_metadata.get("is_writer_rank", True))
        if distributed_writer != active_sp_context.is_writer:
            raise RuntimeError(
                "distributed writer rank does not match frozen LTX095 SP writer"
            )
        distributed_metadata["is_writer_rank"] = active_sp_context.is_writer
        distributed_metadata["ltx095_sp_writer_owned_runtime"] = True
    batch.extra["ltx095_sp_writer_owned_runtime"] = sp_writer_owned_runtime

    batch.extra["runtime_distributed_metadata"] = dict(distributed_metadata)
    batch.extra["runtime_official_parallel_metadata"] = dict(official_parallel_metadata)
    if not batch.extra.get("_runtime_resource_policy_selected_logged", False):
        batch.extra["runtime_resource_policy"] = resource_policy_dict
        batch.extra.setdefault("runtime_resource_event_history", [])
        batch.extra["runtime_resource_event_history"].append(
            {
                "event": "resource_policy_selected",
                **resource_policy_dict,
            }
        )
        for reason in getattr(resource_policy, "fallback_reasons", ()):
            batch.extra["runtime_resource_event_history"].append(
                {
                    "event": "resource_policy_fallback",
                    "requested_policy": getattr(
                        resource_policy, "requested_policy", None
                    ),
                    "selected_policy": getattr(
                        resource_policy, "selected_policy", None
                    ),
                    "reason": reason,
                }
            )
        batch.extra["_runtime_resource_policy_selected_logged"] = True

    if not isinstance(params.runtime_state, dict):
        params.runtime_state = {}
    params.runtime_state["metrics"] = batch.metrics

    output_file_path = None

    video_source = batch.video
    mask_source = batch.mask
    video_path = params.video_input_path
    mask_path = params.mask_input_path

    if params.save_output and bool(distributed_metadata.get("is_writer_rank", True)):
        if params.output_path and not params.output_file_name:
            params.output_file_name = f"{params.request_id}.mp4"
        output_file_path = batch.output_file_path()

    if batch.video is None:
        if not params.video_input_path:
            raise ValueError("video tensor or video_input_path is required")
        if active_sp_context is not None:
            video_metadata = _read_and_broadcast_sp_video_metadata(
                active=active_sp_context,
                video_path=params.video_input_path,
                fallback_fps=float(params.fps or 25),
                server_args=server_args,
                read_video_metadata=read_video_metadata,
            )
        else:
            video_metadata = read_video_metadata(params.video_input_path)
        original_video = None
    else:
        original_video, video_metadata = build_runtime_video(batch.video)
        if params.video_input_path:
            video_metadata = read_video_metadata(params.video_input_path)

    height = int(
        video_metadata.get("height")
        or (original_video.shape[-2] if original_video is not None else 0)
    )
    width = int(
        video_metadata.get("width")
        or (original_video.shape[-1] if original_video is not None else 0)
    )
    num_frames = int(
        video_metadata.get("num_frames")
        or (original_video.shape[2] if original_video is not None else 0)
    )

    runtime_mode_selection = _resolve_runtime_mode(
        params,
        height=height,
        width=width,
    )
    runtime_mode = runtime_mode_selection.effective_mode

    preload_state = WindowedPreloadRuntimeState(
        active=runtime_mode_selection.window_runtime_mode == "preload",
    )
    streaming_state = WindowedStreamingRuntimeState(
        active=runtime_mode_selection.window_runtime_mode == "streaming",
        status=(
            "ready_for_incremental_load"
            if runtime_mode_selection.window_runtime_mode == "streaming"
            else "inactive"
        ),
    )

    fps = float(video_metadata.get("fps") or params.fps or 25)
    codec_name = video_metadata.get("codec_name")
    writer_enabled = bool(distributed_metadata.get("is_writer_rank", True))
    encoding_profile = (
        VideoEncodingProfile.from_metadata(video_metadata)
        if writer_enabled and params.video_input_path
        else None
    )
    if encoding_profile is not None and bool(
        getattr(params, "output_encode_baseline_contract", False)
    ):
        encoding_profile = _apply_baseline_output_contract(
            encoding_profile, video_metadata
        )

    if runtime_mode == "full":
        if original_video is None:
            original_video, _video_metadata = build_runtime_video(
                params.video_input_path
            )
            video_metadata = {**video_metadata, **_video_metadata}
        working_video = original_video.clone()
        mask_cache = (
            build_runtime_mask(params.mask_input_path, original_video.shape[2])
            if batch.mask is None
            else build_runtime_mask(batch.mask, original_video.shape[2])
        )
        if batch.mask is None:
            # Keep old input-probe attribution for file-backed mask load.
            # build_runtime_mask internally performs the actual read.
            pass
        final_video = working_video.clone()
        video_store = None
        sequential_video_reader = None
        sequential_mask_reader = None
        sequential_video_writer = None
        video_frame_cache = None
        mask_frame_cache = None
        mask_source_frames = None
    elif runtime_mode == "windowed_preload":
        if batch.video is not None or batch.mask is not None:
            raise ValueError(
                "windowed_preload runtime requires file-backed video and mask inputs"
            )
        if not params.video_input_path or not params.mask_input_path:
            raise ValueError(
                "windowed_preload runtime requires video_input_path and mask_input_path"
            )
        # The current preload runtime does not consume WindowedVideoStore.
        # Avoid materializing large mmap-backed stores here because they add
        # substantial cleanup overhead without contributing to the active path.
        video_store = None
        working_video = None
        final_video = None
        mask_cache = None
        sequential_video_reader = None
        sequential_mask_reader = None
        sequential_video_writer = None
        video_array_result = read_video_array(video_path)
        mask_array_result = read_mask_array(mask_path)
        video_frames = (
            video_array_result[0]
            if isinstance(video_array_result, tuple)
            else video_array_result
        )
        mask_frames = (
            mask_array_result[0]
            if isinstance(mask_array_result, tuple)
            else mask_array_result
        )
        video_frame_cache = ArrayFrameCache(
            start_index=0,
            shape_tail=(height, width, 3),
            dtype=np.uint8,
        )
        mask_frame_cache = ArrayFrameCache(
            start_index=0,
            shape_tail=(height, width),
            dtype=np.uint8,
        )
        video_frame_cache.append(video_frames)
        mask_frame_cache.append(mask_frames)
        mask_source_frames = mask_frames
    else:
        if batch.video is not None or batch.mask is not None:
            raise ValueError(
                "windowed_streaming runtime requires file-backed video and mask inputs"
            )
        if not params.video_input_path or not params.mask_input_path:
            raise ValueError(
                "windowed_streaming runtime requires video_input_path and mask_input_path"
            )
        # The streaming runtime uses SequentialVideoReader/Writer plus chunked
        # caches. A WindowedVideoStore instance is not part of the active data
        # path, so avoid constructing its large mmap files during normal runs.
        video_store = None
        working_video = None
        final_video = None
        mask_cache = None
        if active_sp_context is None or active_sp_context.is_writer:
            ffmpeg_thread_count = _resolve_ffmpeg_thread_count(batch)
            sequential_video_reader = SequentialVideoReader(
                video_path,
                width=width,
                height=height,
                pix_fmt="rgb24",
                thread_count=ffmpeg_thread_count,
            )
            sequential_mask_reader = SequentialVideoReader(
                mask_path,
                width=width,
                height=height,
                pix_fmt="gray",
                squeeze_single_channel=True,
                thread_count=ffmpeg_thread_count,
            )
            sequential_video_writer = (
                SequentialVideoWriter(
                    output_file_path,
                    thread_count=ffmpeg_thread_count,
                    encoding_profile=encoding_profile,
                    async_queue_depth=1,
                )
                if output_file_path
                else None
            )
            video_frame_cache = TensorFrameCache(
                start_index=0,
                shape_tail=(3, height, width),
                dtype=torch.bfloat16,
            )
            mask_frame_cache = ChunkedFrameCache(
                start_index=0,
                shape_tail=(height, width),
                dtype=np.uint8,
            )
        else:
            sequential_video_reader = None
            sequential_mask_reader = None
            sequential_video_writer = None
            video_frame_cache = None
            mask_frame_cache = None
        mask_source_frames = None

    preload_state.video_cache_impl = _window_cache_impl_name(video_frame_cache)
    preload_state.mask_cache_impl = _window_cache_impl_name(mask_frame_cache)
    streaming_state.video_cache_impl = _window_cache_impl_name(video_frame_cache)
    streaming_state.mask_cache_impl = _window_cache_impl_name(mask_frame_cache)

    bbox_tracks = (
        _load_bbox_tracks(
            params.bbox_path if params.bbox_path is not None else batch.bbox,
            num_frames=num_frames,
        )
        if active_sp_context is None or active_sp_context.is_writer
        else []
    )

    params.height = height
    params.width = width
    params.num_frames = num_frames
    params.fps = int(round(fps))
    batch.height = params.height
    batch.width = params.width
    batch.num_frames = params.num_frames
    batch.fps = params.fps

    controller = (
        MemoryPhaseController(
            memory_adapter,
            rank=int(
                getattr(
                    getattr(server_args, "parallel_context", None),
                    "global_rank",
                    getattr(distributed_context, "rank", 0),
                )
            ),
            device=getattr(server_args, "device", "cpu"),
        )
        if memory_adapter is not None
        else None
    )
    context = LTX095EraseRuntimeContext(
        request_batch=batch,
        original_video=original_video.clone() if original_video is not None else None,
        working_video=working_video,
        final_video=final_video,
        mask_cache=mask_cache,
        fps=fps,
        codec_name=codec_name or None,
        encoding_profile=encoding_profile,
        video_path=video_path,
        mask_path=mask_path,
        output_file_path=output_file_path,
        runtime_mode=runtime_mode,
        requested_runtime_mode=runtime_mode_selection.requested_mode,
        effective_runtime_mode=runtime_mode_selection.effective_mode,
        window_runtime_mode=runtime_mode_selection.window_runtime_mode,
        window_input_policy=runtime_mode_selection.window_input_policy,
        window_output_policy=runtime_mode_selection.window_output_policy,
        window_cache_policy=runtime_mode_selection.window_cache_policy,
        runtime_window_backend=runtime_mode_selection.runtime_window_backend,
        runtime_window_io_policy=runtime_mode_selection.runtime_window_io_policy,
        runtime_flush_policy=runtime_mode_selection.runtime_flush_policy,
        video_store=video_store,
        bbox_tracks=bbox_tracks,
        object_count=len(bbox_tracks) if bbox_tracks else 1,
        pipeline_start_time=time.time(),
        progress_state=(
            batch.extra.get("service_progress_state")
            if batch.extra.get("service_progress_state") is not None
            else (
                create_runtime_progress()
                if bool(distributed_metadata.get("is_progress_rank", True))
                and not batch.suppress_logs
                else None
            )
        ),
        scheduler=(
            RuntimeTaskScheduler()
            if active_sp_context is None or active_sp_context.is_writer
            else None
        ),
        sequential_video_reader=sequential_video_reader,
        sequential_mask_reader=sequential_mask_reader,
        sequential_video_writer=sequential_video_writer,
        video_frame_cache=video_frame_cache,
        mask_frame_cache=mask_frame_cache,
        mask_source_frames=mask_source_frames,
        preload_state=preload_state,
        streaming_state=streaming_state,
        sp_writer_owned_runtime=sp_writer_owned_runtime,
        distributed_metadata=dict(distributed_metadata),
        official_parallel_metadata=dict(official_parallel_metadata),
        memory_phase_controller=controller,
    )

    batch.extra["runtime_mode_selection"] = asdict(runtime_mode_selection)
    batch.extra["runtime_mode_requested"] = context.requested_runtime_mode
    batch.extra["runtime_mode_effective"] = context.effective_runtime_mode
    batch.extra["runtime_window_backend"] = context.runtime_window_backend
    batch.extra["runtime_window_io_policy"] = context.runtime_window_io_policy
    batch.extra["runtime_flush_policy"] = context.runtime_flush_policy
    batch.extra["runtime_resource_policy"] = resource_policy_dict
    batch.extra["runtime_distributed_metadata"] = dict(distributed_metadata)
    batch.extra["runtime_official_parallel_metadata"] = dict(official_parallel_metadata)
    batch.extra["runtime_context"] = context
    if controller is not None:
        batch.extra["memory_phase_controller"] = controller

    return context
