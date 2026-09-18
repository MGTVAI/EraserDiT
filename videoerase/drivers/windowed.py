"""Windowed runtime driver for the LTX095 videoerase pipeline."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import torch

from config.ltx095 import LTX095EraseSamplingParams
from config.server_args import ServerArgs
from nodes.schedule_batch import Req
from videoerase.windowing.commit_sync import (
    release_ltx095_window_commit_payload,
    synchronize_ltx095_window_runtime_boundary,
)
from videoerase.objects import (
    build_ltx095_object_runtime_states,
)
from videoerase.metadata import (
    collect_ltx095_window_torch_compile_status,
    collect_ltx095_window_transformer_cache_status,
    collect_ltx095_window_vae_parallel_history,
    compute_runtime_final_video_shape,
    write_runtime_history_batch_extra,
    write_runtime_mode_batch_extra,
    write_runtime_result_batch_extra,
)
from videoerase.scheduler import RuntimeTaskScheduler
from videoerase.contracts import (
    LTX095EraseRuntimeContext,
    ObjectRuntimeState,
)
from videoerase.windowing.sp_dispatch import (
    LTX095SPWindowCommand,
    LTX095SPWindowCommandCode,
    dispatch_ltx095_sp_window_command,
    resolve_active_ltx095_window_commit_context,
)
from videoerase.windowing.materializer import (
    build_ltx095_sp_peer_window_batch,
    cache_ltx095_window_text_embeddings,
    materialize_ltx095_window_batch,
)
from videoerase.windowing.planner import infer_ltx095_window_bbox
from videoerase.windowing.reclaim import (
    reclaim_completed_ltx095_window,
    transfer_completed_ltx095_window,
)
from memory.policies.memory_phase_controller import MemoryPhase
from service.control import service_checkpoint
from utils.inference_timing import record_diagnostic_stage
from utils.erase_preprocess import resolve_single_window_crop_bbox
from utils.ltx095_origin_contract import (
    resolve_runtime_sp_degree,
    resolve_spatial_alignment,
)
from utils.video_io import TensorFrameCache, frames_uint8_to_tensor
from utils.windowing import WindowSpec


def object_input_ready(
    object_state: ObjectRuntimeState,
    spec: WindowSpec,
) -> bool:
    return object_state.input_frontier >= spec.load_end


def prime_windowed_object_chain_inputs(
    *,
    context: LTX095EraseRuntimeContext,
    object_states: list[ObjectRuntimeState],
    ensure_window_cache_loaded_fn,
) -> bool:
    if context.window_runtime_mode != "streaming":
        return False
    if context.video_frame_cache is None:
        return False

    for object_state in object_states:
        if object_state.finished:
            continue
        if object_state.next_window_index >= object_state.window_count:
            continue
        if object_state.object_index != 0:
            continue
        spec = object_state.window_specs[object_state.next_window_index]
        cache_end_before = context.video_frame_cache.end_index
        if cache_end_before >= spec.load_end:
            return False
        ensure_window_cache_loaded_fn(context, spec)
        return context.video_frame_cache.end_index > cache_end_before
    return False


def finalize_ltx095_object_window_step(
    *,
    context: LTX095EraseRuntimeContext,
    object_states: list[ObjectRuntimeState],
    object_state: ObjectRuntimeState,
    spec: WindowSpec,
    set_object_overlap_cache_fn,
    record_runtime_event_fn,
    record_task_state_snapshot_fn,
) -> None:
    del object_states
    record_runtime_event_fn(
        context,
        "stage_end_begin",
        task_state=object_state,
        window_index=spec.window_index,
        scene_index=spec.scene_index,
        flush_frontier=int(object_state.flush_frontier),
    )
    current_scene_index = object_state.scene_index
    object_state.next_window_index += 1
    if object_state.next_window_index >= object_state.window_count:
        object_state.finished = True
        set_object_overlap_cache_fn(
            object_state,
            start_frame=object_state.flush_frontier,
            frames=None,
        )
        record_runtime_event_fn(
            context,
            "task_finished",
            task_state=object_state,
            window_index=spec.window_index,
        )
    else:
        next_spec = object_state.window_specs[object_state.next_window_index]
        if next_spec.scene_index != current_scene_index:
            set_object_overlap_cache_fn(
                object_state,
                start_frame=next_spec.load_start,
                frames=None,
            )
            next_scene = (
                int(next_spec.scene_start),
                int(next_spec.scene_end - next_spec.scene_start),
            )
            object_state.emit_pop_scene(
                next_spec.scene_index,
                len(object_state.scenes or []),
                next_scene,
            )
            record_runtime_event_fn(
                context,
                "scene_switch",
                task_state=object_state,
                window_index=spec.window_index,
                previous_scene_index=current_scene_index,
                new_scene_index=next_spec.scene_index,
                new_scene=next_scene,
                overlap_reset=True,
            )
        object_state.scene_index = next_spec.scene_index
    record_runtime_event_fn(
        context,
        "stage_end_complete",
        task_state=object_state,
        window_index=spec.window_index,
        next_window_index=int(object_state.next_window_index),
        finished=bool(object_state.finished),
        flush_frontier=int(object_state.flush_frontier),
    )
    record_task_state_snapshot_fn(
        context,
        object_state,
        phase="window_step_finalized",
        window_index=spec.window_index,
    )


def compute_ltx095_streaming_stable_end(
    context: LTX095EraseRuntimeContext,
) -> int:
    if context.scheduler is None:
        return int(context.next_write_index)
    return context.scheduler.compute_tail_flush_end(context)


def maybe_flush_ltx095_streaming_runtime(
    *,
    context: LTX095EraseRuntimeContext,
    evict_cache_before_fn,
    release_mask_frames_fn,
    flush_windowed_frames_fn,
) -> None:
    if context.window_runtime_mode != "streaming":
        return
    stable_end = compute_ltx095_streaming_stable_end(context)
    if context.sequential_video_writer is None:
        if stable_end > context.next_write_index:
            context.next_write_index = stable_end
        if context.object_states:
            evict_cache_before_fn(context, frame_index=stable_end)
            release_mask_frames_fn(
                context,
                release_end=stable_end,
                source="streaming_logical_flush_after_evict",
            )
        return
    flush_windowed_frames_fn(context, flush_end=stable_end)


class _LTX095SPWriterCommandController:
    def __init__(
        self,
        *,
        server_args: ServerArgs,
        runtime_mode: str,
    ) -> None:
        self.server_args = server_args
        self.runtime_mode = runtime_mode
        self.command_dispatched = False

    def _dispatch(self, command: LTX095SPWindowCommand, *, phase: str) -> None:
        self.command_dispatched = True
        dispatched = dispatch_ltx095_sp_window_command(
            self.server_args,
            runtime_mode=self.runtime_mode,
            prepare_writer_command=lambda: command,
            phase=phase,
        )
        if dispatched != command:
            raise RuntimeError("writer received a mismatched SP window command")

    def dispatch_run(
        self,
        *,
        object_index: int,
        window_index: int,
        scene_index: int,
    ) -> None:
        self._dispatch(
            LTX095SPWindowCommand(
                code=LTX095SPWindowCommandCode.RUN,
                object_index=object_index,
                window_index=window_index,
                scene_index=scene_index,
                reason_code=0,
            ),
            phase="writer RUN command",
        )

    def dispatch_skip(
        self,
        *,
        object_index: int,
        window_index: int,
        scene_index: int,
        reason_code: int,
    ) -> None:
        self._dispatch(
            LTX095SPWindowCommand(
                code=LTX095SPWindowCommandCode.SKIP,
                object_index=object_index,
                window_index=window_index,
                scene_index=scene_index,
                reason_code=reason_code,
            ),
            phase="writer SKIP command",
        )

    def dispatch_end(self) -> None:
        self._dispatch(
            LTX095SPWindowCommand(
                code=LTX095SPWindowCommandCode.END,
                object_index=-1,
                window_index=-1,
                scene_index=-1,
                reason_code=0,
            ),
            phase="writer END command",
        )

    def dispatch_preparation_error(self, error: BaseException) -> None:
        def raise_preparation_error() -> LTX095SPWindowCommand:
            raise error

        dispatch_ltx095_sp_window_command(
            self.server_args,
            runtime_mode=self.runtime_mode,
            prepare_writer_command=raise_preparation_error,
            phase="writer window preparation",
        )

    def complete_boundary(self) -> None:
        self.command_dispatched = False


def _reclaim_pending_ltx095_window(
    *,
    context: LTX095EraseRuntimeContext,
    server_args: ServerArgs,
) -> None:
    pending = context.pending_window_reclaim
    window_key = pending.window_key if pending is not None else None
    reclaim_error = None
    try:
        if pending is not None:
            controller = context.memory_phase_controller
            snapshot = (
                (lambda _device: controller.record_point("window_reclaim"))
                if controller is not None
                else None
            )
            reclaim_completed_ltx095_window(
                pending,
                device=getattr(server_args, "device", "cpu"),
                snapshot=snapshot,
            )
            context.pending_window_reclaim = None
    except BaseException as error:
        reclaim_error = error
    synchronize_ltx095_window_runtime_boundary(reclaim_error, server_args)
    context.window_reclaim_events.append(
        {
            "event": "window_reclaim",
            "monotonic_ns": time.monotonic_ns(),
            "window_key": window_key,
            "status": "complete" if reclaim_error is None else "error",
            "error": (
                None
                if reclaim_error is None
                else f"{type(reclaim_error).__name__}: {reclaim_error}"
            ),
        }
    )


def _drop_pending_ltx095_window_refs(
    context: LTX095EraseRuntimeContext,
) -> None:
    pending = context.pending_window_reclaim
    if pending is not None:
        context.window_reclaim_events.append(
            {
                "event": "window_end_drop_refs",
                "monotonic_ns": time.monotonic_ns(),
                "window_key": pending.window_key,
                "status": "complete",
            }
        )
        pending.batch = None
        context.pending_window_reclaim = None


def _close_ltx095_memory_phase_controller(
    context: LTX095EraseRuntimeContext,
) -> None:
    controller = context.memory_phase_controller
    if controller is not None:
        controller.close()


def _run_ltx095_writer_window_runtime(
    *,
    executor,
    stages,
    batch: Req,
    context: LTX095EraseRuntimeContext,
    params: LTX095EraseSamplingParams,
    server_args: ServerArgs,
    logger,
    handlers,
    command_controller: _LTX095SPWriterCommandController | None,
) -> Req:
    runtime_setup_start = time.perf_counter()
    object_states, total_windows = build_ltx095_object_runtime_states(
        context=context,
        params=params,
        create_empty_cache_like_fn=handlers.create_empty_cache_like,
        register_runtime_task_chain_hooks_fn=handlers.register_runtime_task_chain_hooks,
        record_runtime_event_fn=handlers.record_runtime_event,
        record_task_state_snapshot_fn=handlers.record_task_state_snapshot,
        window_cache_impl_name_fn=handlers.window_cache_impl_name,
    )
    batch.extra["runtime_object_state_count"] = context.object_count
    batch.extra["runtime_total_windows"] = total_windows
    object_count = len(object_states)
    flat_window_specs: list[dict[str, Any]] = []

    write_runtime_mode_batch_extra(
        batch=batch,
        context=context,
        params=params,
        object_count=object_count,
        runtime_window_cache_impl=handlers.window_cache_impl_name(
            context.video_frame_cache
        ),
    )

    if not batch.suppress_logs:
        logger.info(
            "LTX095 erase pipeline objects=%d infer_len=%d overlap=%d time_sample=%d time_shift=%d fps=%.3f runtime_requested=%s runtime_effective=%s window_backend=%s cache_impl=%s flush_policy=%s distributed_compute_mode=%s native_vae_parallel=%s degree=%d mode=%s",
            object_count,
            params.infer_len,
            params.overlap,
            params.time_sample,
            params.time_shift,
            context.fps,
            context.requested_runtime_mode,
            context.effective_runtime_mode,
            context.runtime_window_backend,
            handlers.window_cache_impl_name(context.video_frame_cache),
            context.runtime_flush_policy,
            context.official_parallel_metadata.get(
                "distributed_compute_mode", "entry_only"
            ),
            context.official_parallel_metadata.get(
                "ltx095_native_vae_parallel_enabled", False
            ),
            int(
                context.official_parallel_metadata.get(
                    "ltx095_native_vae_parallel_degree", 1
                )
            ),
            context.official_parallel_metadata.get(
                "ltx095_native_vae_parallel_mode", "disabled"
            ),
        )

    if context.progress_state is not None:
        context.progress_state.add_pipeline_task(total_windows)

    for state in object_states:
        context.window_specs = state.window_specs
        flat_window_specs.extend(
            [
                {**asdict(spec), "object_index": state.object_index}
                for spec in state.window_specs
            ]
        )
        for spec in state.window_specs:
            handlers.update_window_state(
                context,
                state.object_index,
                spec.window_index,
                status="ready",
                scene_index=spec.scene_index,
                load_start=spec.load_start,
                load_end=spec.load_end,
                commit_start=spec.commit_start,
                commit_end=spec.commit_end,
                stable_end=spec.commit_end - spec.overlap_right,
            )
        if not batch.suppress_logs:
            logger.info(
                "object %d/%d windows=%d scenes=%s input_role=%s",
                state.object_index + 1,
                object_count,
                state.window_count,
                state.scenes,
                state.input_role,
            )

    record_diagnostic_stage(
        batch.metrics,
        "diagnostic.runtime.setup",
        time.perf_counter() - runtime_setup_start,
    )
    completed_windows = 0
    window_prepare_seconds = 0.0
    executor_seconds = 0.0
    window_commit_seconds = 0.0
    while completed_windows < total_windows:
        window_prepare_start = time.perf_counter()
        scheduler = context.scheduler or RuntimeTaskScheduler()
        context.scheduler = scheduler
        ready_state = scheduler.select_next_task(object_states, object_input_ready)
        if ready_state is None:
            primed = prime_windowed_object_chain_inputs(
                context=context,
                object_states=object_states,
                ensure_window_cache_loaded_fn=handlers.ensure_window_cache_loaded,
            )
            scheduler.record_event(
                "scheduler_prime_source",
                primed=bool(primed),
            )
            if primed:
                ready_state = scheduler.select_next_task(
                    object_states, object_input_ready
                )
        if ready_state is None:
            state_debug = scheduler.describe_states(object_states)
            raise RuntimeError(
                f"No ready object found in windowed object-chain scheduler. state={state_debug}"
            )

        spec = ready_state.window_specs[ready_state.next_window_index]
        handlers.update_window_state(
            context,
            ready_state.object_index,
            spec.window_index,
            status="running",
            scene_index=spec.scene_index,
            load_start=spec.load_start,
            load_end=spec.load_end,
            commit_start=spec.commit_start,
            commit_end=spec.commit_end,
            input_role=ready_state.input_role,
        )
        object_bbox_frames = (
            context.bbox_tracks[ready_state.object_index]
            if context.bbox_tracks
            and ready_state.object_index < len(context.bbox_tracks)
            else None
        )
        if object_bbox_frames is not None:
            window_bbox = infer_ltx095_window_bbox(
                bbox_frames=object_bbox_frames,
                spec=spec,
                window_mask=None,
            )
            window_mask = None
        else:
            window_mask = handlers.materialize_object_window_mask(
                context, ready_state, spec
            )
            window_bbox = infer_ltx095_window_bbox(
                bbox_frames=None,
                spec=spec,
                window_mask=window_mask,
            )
        if window_bbox is None:
            if command_controller is not None:
                command_controller.dispatch_skip(
                    object_index=ready_state.object_index,
                    window_index=spec.window_index,
                    scene_index=spec.scene_index,
                    reason_code=2,
                )
            if not batch.suppress_logs:
                logger.info(
                    "object %d window %d skip: load=%d:%d deal=%d:%d reason=no_valid_bbox",
                    ready_state.object_index,
                    spec.window_index,
                    spec.load_start,
                    spec.load_end,
                    spec.deal_start,
                    spec.commit_end,
                )
            lifecycle_error = None
            try:
                handlers.record_skipped_object_window(
                    context=context,
                    object_state=ready_state,
                    spec=spec,
                    reason="no_valid_bbox",
                    prompt=handlers.select_sequence_item(
                        ready_state.prompt_source, spec.scene_index
                    ),
                    negative_prompt=handlers.select_sequence_item(
                        ready_state.negative_prompt_source,
                        spec.scene_index,
                    ),
                )
                finalize_ltx095_object_window_step(
                    context=context,
                    object_states=object_states,
                    object_state=ready_state,
                    spec=spec,
                    set_object_overlap_cache_fn=handlers.set_object_overlap_cache,
                    record_runtime_event_fn=handlers.record_runtime_event,
                    record_task_state_snapshot_fn=handlers.record_task_state_snapshot,
                )
                maybe_flush_ltx095_streaming_runtime(
                    context=context,
                    evict_cache_before_fn=handlers.evict_cache_before,
                    release_mask_frames_fn=handlers.release_mask_frames,
                    flush_windowed_frames_fn=handlers.flush_windowed_frames,
                )
            except BaseException as error:
                lifecycle_error = error
            synchronize_ltx095_window_runtime_boundary(lifecycle_error, server_args)
            service_checkpoint(
                batch,
                server_args,
                phase="window_skip_complete",
            )
            if command_controller is not None:
                command_controller.complete_boundary()
            completed_windows += 1
            if not batch.suppress_logs:
                handlers.log_runtime_progress(
                    context=context,
                    completed_windows=completed_windows,
                    total_windows=total_windows,
                    current_object_index=ready_state.object_index,
                    current_window_index=spec.window_index,
                )
            elif context.progress_state is not None:
                context.progress_state.update_pipeline(
                    completed=completed_windows,
                    total=total_windows,
                    object_index=ready_state.object_index,
                    object_count=context.object_count,
                    window_index=spec.window_index,
                    window_count=len(context.window_specs),
                )
            window_prepare_seconds += time.perf_counter() - window_prepare_start
            continue

        if command_controller is not None:
            command_controller.dispatch_run(
                object_index=ready_state.object_index,
                window_index=spec.window_index,
                scene_index=spec.scene_index,
            )
        align_w, align_h = resolve_spatial_alignment(
            resolve_runtime_sp_degree(server_args)
        )
        aligned_crop_bbox = resolve_single_window_crop_bbox(
            bbox=window_bbox,
            video_width=int(params.width),
            video_height=int(params.height),
            align_h=align_h,
            align_w=align_w,
            scale_area_ratio=float(params.scale_area_ratio),
            min_pixels=int(params.min_pixels),
            max_pixels=int(params.max_pixels),
            force_crop_align=bool(params.force_crop_align),
        )
        if window_mask is None:
            window_mask = handlers.materialize_object_window_mask(
                context, ready_state, spec, crop_bbox=aligned_crop_bbox
            )
        elif isinstance(ready_state.input_cache, TensorFrameCache):
            x, y, width, height = aligned_crop_bbox
            window_mask = window_mask[..., y : y + height, x : x + width].contiguous()
        _reclaim_pending_ltx095_window(
            context=context,
            server_args=server_args,
        )
        window_batch = materialize_ltx095_window_batch(
            batch=batch,
            context=context,
            params=params,
            spec=spec,
            object_index=ready_state.object_index,
            bbox_frames=object_bbox_frames,
            ensure_window_cache_loaded_fn=handlers.ensure_window_cache_loaded,
            record_runtime_event_fn=handlers.record_runtime_event,
            object_state=ready_state,
            video_cache=ready_state.input_cache,
            window_mask=window_mask,
            window_bbox=window_bbox,
            aligned_crop_bbox=(
                aligned_crop_bbox
                if isinstance(ready_state.input_cache, TensorFrameCache)
                else None
            ),
        )
        window_batch.mask = window_mask

        if not batch.suppress_logs:
            logger.info(
                "object %d window %d run: load=%d:%d deal=%d:%d commit=%d:%d overlap_left=%d overlap_right=%d future=%d bbox=%s fuse=%s input_role=%s",
                ready_state.object_index,
                spec.window_index,
                spec.load_start,
                spec.load_end,
                spec.deal_start,
                spec.commit_end,
                spec.commit_start,
                spec.commit_end,
                spec.overlap_left,
                spec.overlap_right,
                spec.future_length,
                window_batch.bbox,
                window_batch.overlap_fuse_mode,
                ready_state.input_role,
            )
        if context.progress_state is not None:
            context.progress_state.reset_denoise_task(
                object_index=ready_state.object_index,
                object_count=object_count,
                window_index=spec.window_index,
                window_count=ready_state.window_count,
                total_steps=(
                    len(window_batch.timesteps)
                    if window_batch.timesteps is not None
                    else int(window_batch.num_inference_steps)
                ),
            )
        window_batch.extra["runtime_progress_state"] = context.progress_state
        window_batch.extra["runtime_progress_object_index"] = ready_state.object_index
        window_batch.extra["runtime_progress_object_count"] = object_count
        window_batch.extra["runtime_progress_window_index"] = spec.window_index
        window_batch.extra["runtime_progress_window_count"] = ready_state.window_count
        window_batch.extra["runtime_object_input_role"] = ready_state.input_role
        window_prepare_seconds += time.perf_counter() - window_prepare_start
        executor_start = time.perf_counter()
        window_batch = executor.execute_with_profiling(
            stages,
            window_batch,
            server_args,
        )
        collect_ltx095_window_vae_parallel_history(
            context=context,
            window_batch=window_batch,
        )
        collect_ltx095_window_torch_compile_status(
            batch=batch,
            window_batch=window_batch,
        )
        collect_ltx095_window_transformer_cache_status(
            batch=batch,
            window_batch=window_batch,
        )
        executor_seconds += time.perf_counter() - executor_start
        window_commit_start = time.perf_counter()
        lifecycle_error = None
        controller = context.memory_phase_controller
        try:
            if controller is not None:
                controller.enter(
                    MemoryPhase.COMMIT,
                    component_name=None,
                    window_key=(
                        int(ready_state.object_index),
                        int(spec.window_index),
                    ),
                )
            if (
                getattr(window_batch, "prompt_embeds", None) is not None
                and getattr(window_batch, "prompt_attention_mask", None) is not None
                and getattr(window_batch, "negative_prompt_embeds", None) is not None
                and getattr(window_batch, "negative_attention_mask", None) is not None
                and not isinstance(
                    window_batch.extra.get("cached_text_embeddings"), dict
                )
            ):
                cache_ltx095_window_text_embeddings(
                    context=context,
                    object_index=ready_state.object_index,
                    scene_index=spec.scene_index,
                    prompt=window_batch.prompt,
                    negative_prompt=window_batch.negative_prompt,
                    prompt_embeds=window_batch.prompt_embeds,
                    prompt_attention_mask=window_batch.prompt_attention_mask,
                    negative_prompt_embeds=window_batch.negative_prompt_embeds,
                    negative_attention_mask=window_batch.negative_attention_mask,
                    record_runtime_event_fn=handlers.record_runtime_event,
                )
            try:
                handlers.commit_window_to_object_output(
                    context=context,
                    object_state=ready_state,
                    spec=spec,
                    window_batch=window_batch,
                )
            finally:
                release_ltx095_window_commit_payload(window_batch, server_args)
            finalize_ltx095_object_window_step(
                context=context,
                object_states=object_states,
                object_state=ready_state,
                spec=spec,
                set_object_overlap_cache_fn=handlers.set_object_overlap_cache,
                record_runtime_event_fn=handlers.record_runtime_event,
                record_task_state_snapshot_fn=handlers.record_task_state_snapshot,
            )
            maybe_flush_ltx095_streaming_runtime(
                context=context,
                evict_cache_before_fn=handlers.evict_cache_before,
                release_mask_frames_fn=handlers.release_mask_frames,
                flush_windowed_frames_fn=handlers.flush_windowed_frames,
            )
        except BaseException as error:
            lifecycle_error = error
        finally:
            if (
                controller is not None
                and controller.active_phase is MemoryPhase.COMMIT
            ):
                try:
                    controller.exit(MemoryPhase.COMMIT)
                except BaseException as error:
                    if lifecycle_error is None:
                        lifecycle_error = error
        if lifecycle_error is None:
            context.pending_window_reclaim = transfer_completed_ltx095_window(
                window_batch
            )
            window_batch = None
        synchronize_ltx095_window_runtime_boundary(lifecycle_error, server_args)
        service_checkpoint(
            batch,
            server_args,
            phase="window_commit_complete",
        )
        if command_controller is not None:
            command_controller.complete_boundary()
        completed_windows += 1
        if not batch.suppress_logs:
            handlers.log_runtime_progress(
                context=context,
                completed_windows=completed_windows,
                total_windows=total_windows,
                current_object_index=ready_state.object_index,
                current_window_index=spec.window_index,
            )
        elif context.progress_state is not None:
            context.progress_state.update_pipeline(
                completed=completed_windows,
                total=total_windows,
                object_index=ready_state.object_index,
                object_count=context.object_count,
                window_index=spec.window_index,
                window_count=len(context.window_specs),
            )
        if context.progress_state is not None:
            context.progress_state.hide_denoise()
        window_commit_seconds += time.perf_counter() - window_commit_start

    runtime_finalize_start = time.perf_counter()
    context.object_history = [
        {
            "object_index": state.object_index,
            "scenes": state.scenes,
            "prompt_source": state.prompt_source,
            "negative_prompt_source": state.negative_prompt_source,
            "window_count": state.window_count,
            "skip_count": state.skip_count,
            "input_role": state.input_role,
            "flush_frontier": state.flush_frontier,
            "finished": state.finished,
            "input_cache_end": state.input_cache.end_index,
            "output_cache_end": state.output_cache.end_index,
        }
        for state in object_states
    ]

    if context.window_runtime_mode == "streaming":
        maybe_flush_ltx095_streaming_runtime(
            context=context,
            evict_cache_before_fn=handlers.evict_cache_before,
            release_mask_frames_fn=handlers.release_mask_frames,
            flush_windowed_frames_fn=handlers.flush_windowed_frames,
        )

    if object_states:
        context.final_window_output_cache = object_states[-1].output_cache
        if context.window_runtime_mode != "streaming" or not params.save_output:
            context.video_frame_cache = object_states[-1].output_cache
            cached_final_frames = context.final_window_output_cache.slice(
                context.final_window_output_cache.start_index,
                context.final_window_output_cache.end_index,
            )
            final_frames = (
                cached_final_frames
                if isinstance(cached_final_frames, torch.Tensor)
                else frames_uint8_to_tensor(cached_final_frames)
            )
            context.final_video = final_frames.permute(1, 0, 2, 3).unsqueeze(0)
        else:
            context.final_video = None

    write_runtime_result_batch_extra(
        batch=batch,
        context=context,
        flat_window_specs=flat_window_specs,
        final_video_shape=compute_runtime_final_video_shape(
            context=context,
            num_frames=int(params.num_frames),
            height=int(params.height),
            width=int(params.width),
        ),
        include_window_state_history=True,
        include_text_embedding_cache_keys=True,
        object_chain_final_cache_impl=handlers.window_cache_impl_name(
            context.final_window_output_cache
        ),
    )
    write_runtime_history_batch_extra(
        batch=batch,
        context=context,
        include_text_embedding_history=True,
        include_load_event_history=True,
        include_flush_event_history=True,
        include_evict_event_history=True,
        include_object_transition_history=True,
        include_runtime_role_metadata=False,
    )
    if command_controller is not None:
        command_controller.dispatch_end()
    finalize_error = None
    try:
        _drop_pending_ltx095_window_refs(context)
        _close_ltx095_memory_phase_controller(context)
    except BaseException as error:
        finalize_error = error
    if command_controller is not None:
        synchronize_ltx095_window_runtime_boundary(finalize_error, server_args)
    elif finalize_error is not None:
        raise finalize_error
    batch.extra["runtime_window_reclaim_events"] = list(
        context.window_reclaim_events
    )
    if context.memory_phase_controller is not None:
        batch.extra["runtime_phase_events"] = [
            asdict(event) for event in context.memory_phase_controller.events
        ]
    record_diagnostic_stage(
        batch.metrics,
        "diagnostic.runtime.window_prepare",
        window_prepare_seconds,
    )
    record_diagnostic_stage(
        batch.metrics,
        "diagnostic.runtime.executor",
        executor_seconds,
    )
    record_diagnostic_stage(
        batch.metrics,
        "diagnostic.runtime.window_commit",
        window_commit_seconds,
    )
    record_diagnostic_stage(
        batch.metrics,
        "diagnostic.runtime.finalize",
        time.perf_counter() - runtime_finalize_start,
    )
    return batch


def run_ltx095_legacy_window_runtime(
    *,
    executor,
    stages,
    batch: Req,
    context: LTX095EraseRuntimeContext,
    params: LTX095EraseSamplingParams,
    server_args: ServerArgs,
    logger,
    handlers,
) -> Req:
    return _run_ltx095_writer_window_runtime(
        executor=executor,
        stages=stages,
        batch=batch,
        context=context,
        params=params,
        server_args=server_args,
        logger=logger,
        handlers=handlers,
        command_controller=None,
    )


def run_ltx095_sp_writer_window_runtime(
    *,
    executor,
    stages,
    batch: Req,
    context: LTX095EraseRuntimeContext,
    params: LTX095EraseSamplingParams,
    server_args: ServerArgs,
    logger,
    handlers,
) -> Req:
    controller = _LTX095SPWriterCommandController(
        server_args=server_args,
        runtime_mode=context.effective_runtime_mode,
    )
    try:
        return _run_ltx095_writer_window_runtime(
            executor=executor,
            stages=stages,
            batch=batch,
            context=context,
            params=params,
            server_args=server_args,
            logger=logger,
            handlers=handlers,
            command_controller=controller,
        )
    except BaseException as error:
        if not controller.command_dispatched:
            controller.dispatch_preparation_error(error)
        raise


def run_ltx095_sp_peer_window_runtime(
    *,
    executor,
    stages,
    batch: Req,
    context: LTX095EraseRuntimeContext,
    params: LTX095EraseSamplingParams,
    server_args: ServerArgs,
    logger,
    handlers,
) -> Req:
    del logger, handlers
    while True:
        command = dispatch_ltx095_sp_window_command(
            server_args,
            runtime_mode=context.effective_runtime_mode,
            prepare_writer_command=None,
            phase="peer window command receive",
        )
        if command is None:
            raise RuntimeError("active SP peer did not receive a window command")
        if command.code is LTX095SPWindowCommandCode.END:
            finalize_error = None
            try:
                _drop_pending_ltx095_window_refs(context)
                _close_ltx095_memory_phase_controller(context)
            except BaseException as error:
                finalize_error = error
            synchronize_ltx095_window_runtime_boundary(finalize_error, server_args)
            return batch
        if command.code is LTX095SPWindowCommandCode.SKIP:
            synchronize_ltx095_window_runtime_boundary(None, server_args)
            service_checkpoint(
                batch,
                server_args,
                phase="window_skip_complete",
            )
            continue
        if command.code is not LTX095SPWindowCommandCode.RUN:
            raise RuntimeError(f"unsupported SP peer window command: {command.code}")
        _reclaim_pending_ltx095_window(
            context=context,
            server_args=server_args,
        )
        window_batch = build_ltx095_sp_peer_window_batch(
            batch=batch,
            params=params,
            object_index=command.object_index,
            window_index=command.window_index,
            scene_index=command.scene_index,
        )
        window_batch = executor.execute_with_profiling(
            stages,
            window_batch,
            server_args,
        )
        collect_ltx095_window_vae_parallel_history(
            context=context,
            window_batch=window_batch,
        )
        collect_ltx095_window_torch_compile_status(
            batch=batch,
            window_batch=window_batch,
        )
        collect_ltx095_window_transformer_cache_status(
            batch=batch,
            window_batch=window_batch,
        )
        context.pending_window_reclaim = transfer_completed_ltx095_window(
            window_batch
        )
        window_batch = None
        synchronize_ltx095_window_runtime_boundary(None, server_args)
        service_checkpoint(
            batch,
            server_args,
            phase="window_commit_complete",
        )


def is_ltx095_sp_writer_owned_window_runtime(
    context: LTX095EraseRuntimeContext,
    server_args: ServerArgs,
) -> bool:
    return bool(
        context.sp_writer_owned_runtime
        and context.effective_runtime_mode == "windowed_streaming"
        and resolve_active_ltx095_window_commit_context(server_args) is not None
    )


def run_ltx095_windowed_runtime(
    *,
    executor,
    stages,
    batch: Req,
    context: LTX095EraseRuntimeContext,
    params: LTX095EraseSamplingParams,
    server_args: ServerArgs,
    logger,
    handlers,
) -> Req:
    if is_ltx095_sp_writer_owned_window_runtime(context, server_args):
        active = resolve_active_ltx095_window_commit_context(server_args)
        assert active is not None
        runtime_fn = (
            run_ltx095_sp_writer_window_runtime
            if active.is_writer
            else run_ltx095_sp_peer_window_runtime
        )
        return runtime_fn(
            executor=executor,
            stages=stages,
            batch=batch,
            context=context,
            params=params,
            server_args=server_args,
            logger=logger,
            handlers=handlers,
        )
    return run_ltx095_legacy_window_runtime(
        executor=executor,
        stages=stages,
        batch=batch,
        context=context,
        params=params,
        server_args=server_args,
        logger=logger,
        handlers=handlers,
    )
