"""Full runtime driver for the LTX095 videoerase pipeline."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from config.ltx095 import LTX095EraseSamplingParams
from config.server_args import ServerArgs
from nodes.schedule_batch import Req
from videoerase.windowing.commit_sync import (
    release_ltx095_window_commit_payload,
    synchronize_ltx095_window_runtime_boundary,
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
from videoerase.tracks import (
    _resolve_object_scenes,
    _resolve_object_value,
)
from videoerase.contracts import (
    LTX095EraseRuntimeContext,
    _is_windowed_runtime_mode,
)
from videoerase.io.masks import materialize_ltx095_window_mask
from service.control import service_checkpoint
from videoerase.windowing.materializer import (
    cache_ltx095_window_text_embeddings,
    materialize_ltx095_window_batch,
)
from videoerase.windowing.planner import (
    build_ltx095_window_specs,
    infer_ltx095_window_bbox,
)


def run_ltx095_full_runtime(
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
    object_count = context.object_count
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
            "LTX095 erase pipeline objects=%d infer_len=%d overlap=%d time_sample=%d time_shift=%d fps=%.3f runtime_requested=%s runtime_effective=%s window_backend=%s cache_impl=%s flush_policy=%s",
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
        )

    total_windows = 0
    for object_index in range(object_count):
        object_scenes = _resolve_object_scenes(
            params.scenes,
            object_index=object_index,
            object_count=object_count,
        )
        total_windows += len(build_ltx095_window_specs(params, scenes=object_scenes))
    completed_windows = 0
    if context.progress_state is not None:
        context.progress_state.add_pipeline_task(total_windows)

    for object_index in range(object_count):
        object_bbox_frames = (
            context.bbox_tracks[object_index]
            if context.bbox_tracks and object_index < len(context.bbox_tracks)
            else None
        )
        object_scenes = _resolve_object_scenes(
            params.scenes,
            object_index=object_index,
            object_count=object_count,
        )
        window_specs = build_ltx095_window_specs(params, scenes=object_scenes)
        context.window_specs = window_specs
        flat_window_specs.extend(
            [{**asdict(spec), "object_index": object_index} for spec in window_specs]
        )

        object_prompt_source = _resolve_object_value(
            params.prompt,
            object_index=object_index,
            object_count=object_count,
        )
        object_negative_prompt_source = _resolve_object_value(
            params.negative_prompt,
            object_index=object_index,
            object_count=object_count,
        )

        object_summary = {
            "object_index": object_index,
            "scenes": object_scenes,
            "prompt_source": object_prompt_source,
            "negative_prompt_source": object_negative_prompt_source,
            "window_count": len(window_specs),
            "skip_count": 0,
        }

        if not batch.suppress_logs:
            logger.info(
                "object %d/%d windows=%d scenes=%s",
                object_index + 1,
                object_count,
                len(window_specs),
                object_scenes,
            )

        for spec in window_specs:
            service_checkpoint(
                batch,
                server_args,
                phase="full_window_start",
            )
            window_mask = materialize_ltx095_window_mask(
                context=context,
                spec=spec,
                ensure_window_cache_loaded_fn=handlers.ensure_window_cache_loaded,
            )
            window_bbox = infer_ltx095_window_bbox(
                bbox_frames=object_bbox_frames,
                spec=spec,
                window_mask=window_mask,
            )
            if window_bbox is None:
                if not batch.suppress_logs:
                    logger.info(
                        "object %d window %d skip: load=%d:%d deal=%d:%d reason=no_valid_bbox",
                        object_index,
                        spec.window_index,
                        spec.load_start,
                        spec.load_end,
                        spec.deal_start,
                        spec.commit_end,
                    )
                object_summary["skip_count"] += 1
                lifecycle_error = None
                try:
                    handlers.record_skipped_window(
                        context=context,
                        spec=spec,
                        object_index=object_index,
                        reason="no_valid_bbox",
                        prompt=handlers.select_sequence_item(
                            object_prompt_source, spec.scene_index
                        ),
                        negative_prompt=handlers.select_sequence_item(
                            object_negative_prompt_source,
                            spec.scene_index,
                        ),
                    )
                except BaseException as error:
                    lifecycle_error = error
                synchronize_ltx095_window_runtime_boundary(lifecycle_error, server_args)
                service_checkpoint(
                    batch,
                    server_args,
                    phase="full_window_skip_complete",
                )
                completed_windows += 1
                if not batch.suppress_logs:
                    handlers.log_runtime_progress(
                        context=context,
                        completed_windows=completed_windows,
                        total_windows=total_windows,
                        current_object_index=object_index,
                        current_window_index=spec.window_index,
                    )
                elif context.progress_state is not None:
                    context.progress_state.update_pipeline(
                        completed=completed_windows,
                        total=total_windows,
                        object_index=object_index,
                        object_count=context.object_count,
                        window_index=spec.window_index,
                        window_count=len(context.window_specs),
                    )
                continue

            window_batch = materialize_ltx095_window_batch(
                batch=batch,
                context=context,
                params=params,
                spec=spec,
                object_index=object_index,
                bbox_frames=object_bbox_frames,
                ensure_window_cache_loaded_fn=handlers.ensure_window_cache_loaded,
                record_runtime_event_fn=handlers.record_runtime_event,
                window_mask=window_mask,
                window_bbox=window_bbox,
            )

            if not batch.suppress_logs:
                logger.info(
                    "object %d window %d run: load=%d:%d deal=%d:%d commit=%d:%d overlap_left=%d overlap_right=%d future=%d bbox=%s fuse=%s",
                    object_index,
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
                )
            if context.progress_state is not None:
                context.progress_state.reset_denoise_task(
                    object_index=object_index,
                    object_count=object_count,
                    window_index=spec.window_index,
                    window_count=len(window_specs),
                    total_steps=(
                        len(window_batch.timesteps)
                        if window_batch.timesteps is not None
                        else int(window_batch.num_inference_steps)
                    ),
                )
            window_batch.extra["runtime_progress_state"] = context.progress_state
            window_batch.extra["runtime_progress_object_index"] = object_index
            window_batch.extra["runtime_progress_object_count"] = object_count
            window_batch.extra["runtime_progress_window_index"] = spec.window_index
            window_batch.extra["runtime_progress_window_count"] = len(window_specs)
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
            lifecycle_error = None
            try:
                if (
                    getattr(window_batch, "prompt_embeds", None) is not None
                    and getattr(window_batch, "prompt_attention_mask", None) is not None
                    and getattr(window_batch, "negative_prompt_embeds", None)
                    is not None
                    and getattr(window_batch, "negative_attention_mask", None)
                    is not None
                    and not isinstance(
                        window_batch.extra.get("cached_text_embeddings"), dict
                    )
                ):
                    cache_ltx095_window_text_embeddings(
                        context=context,
                        object_index=object_index,
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
                    handlers.commit_window(
                        context=context,
                        spec=spec,
                        window_batch=window_batch,
                        object_index=object_index,
                    )
                finally:
                    release_ltx095_window_commit_payload(window_batch, server_args)

                if _is_windowed_runtime_mode(context.runtime_mode):
                    handlers.evict_cache_before(context, frame_index=spec.commit_end)
            except BaseException as error:
                lifecycle_error = error
            synchronize_ltx095_window_runtime_boundary(lifecycle_error, server_args)
            service_checkpoint(
                batch,
                server_args,
                phase="full_window_commit_complete",
            )
            completed_windows += 1
            if not batch.suppress_logs:
                handlers.log_runtime_progress(
                    context=context,
                    completed_windows=completed_windows,
                    total_windows=total_windows,
                    current_object_index=object_index,
                    current_window_index=spec.window_index,
                )
            elif context.progress_state is not None:
                context.progress_state.update_pipeline(
                    completed=completed_windows,
                    total=total_windows,
                    object_index=object_index,
                    object_count=context.object_count,
                    window_index=spec.window_index,
                    window_count=len(context.window_specs),
                )
            if context.progress_state is not None:
                context.progress_state.hide_denoise()

        context.object_history.append(object_summary)

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
        include_window_state_history=False,
        include_text_embedding_cache_keys=False,
    )
    write_runtime_history_batch_extra(
        batch=batch,
        context=context,
        include_text_embedding_history=False,
        include_load_event_history=False,
        include_flush_event_history=False,
        include_evict_event_history=False,
        include_object_transition_history=False,
        include_runtime_role_metadata=False,
    )
    return batch
