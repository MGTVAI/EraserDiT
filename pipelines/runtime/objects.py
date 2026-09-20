"""Object runtime state builder helpers for LTX095 pipelines.runtime."""

from __future__ import annotations

from typing import Any, Callable

from config.ltx095 import LTX095EraseSamplingParams
from pipelines.runtime.tracks import (
    _resolve_object_scenes,
    _resolve_object_value,
)
from pipelines.runtime.contracts import (
    LTX095EraseRuntimeContext,
    ObjectRuntimeState,
)
from pipelines.runtime.windowing.planner import build_ltx095_window_specs
from utils.video_io import ArrayFrameCache, ChunkedFrameCache, TensorFrameCache

FrameCache = ArrayFrameCache | ChunkedFrameCache | TensorFrameCache


def build_ltx095_object_runtime_states(
    *,
    context: LTX095EraseRuntimeContext,
    params: LTX095EraseSamplingParams,
    create_empty_cache_like_fn: Callable[[FrameCache, int], FrameCache],
    register_runtime_task_chain_hooks_fn: Callable[
        [LTX095EraseRuntimeContext, list[ObjectRuntimeState]], None
    ],
    record_runtime_event_fn: Callable[..., dict[str, Any]],
    record_task_state_snapshot_fn: Callable[..., dict[str, Any]],
    window_cache_impl_name_fn: Callable[[FrameCache | None], str],
) -> tuple[list[ObjectRuntimeState], int]:
    if context.video_frame_cache is None:
        raise ValueError("windowed runtime missing source video cache")

    object_count = context.object_count
    object_states: list[ObjectRuntimeState] = []
    shared_window_specs = []
    prompt_sources = []
    negative_prompt_sources = []
    scene_sources = []

    total_windows = 0
    for object_index in range(object_count):
        object_scenes = _resolve_object_scenes(
            params.scenes,
            object_index=object_index,
            object_count=object_count,
        )
        window_specs = build_ltx095_window_specs(params, scenes=object_scenes)
        shared_window_specs.append(window_specs)
        total_windows += len(window_specs)
        scene_sources.append(object_scenes)
        prompt_sources.append(
            _resolve_object_value(
                params.prompt,
                object_index=object_index,
                object_count=object_count,
            )
        )
        negative_prompt_sources.append(
            _resolve_object_value(
                params.negative_prompt,
                object_index=object_index,
                object_count=object_count,
            )
        )

    for object_index in range(object_count):
        if object_index == 0:
            input_cache = context.video_frame_cache
            input_role = "source_video_cache"
            input_frontier = input_cache.end_index
        else:
            input_cache = create_empty_cache_like_fn(context.video_frame_cache, 0)
            input_role = f"object_{object_index - 1}_output"
            input_frontier = input_cache.end_index

        output_cache = create_empty_cache_like_fn(context.video_frame_cache, 0)
        overlap_cache = create_empty_cache_like_fn(context.video_frame_cache, 0)
        object_states.append(
            ObjectRuntimeState(
                task_index=object_index,
                object_index=object_index,
                window_specs=shared_window_specs[object_index],
                raw_input_cache=input_cache,
                modified_output_cache=output_cache,
                input_role=input_role,
                next_window_index=0,
                finished=False,
                flush_frontier=0,
                input_frontier=input_frontier,
                overlap_output_cache=overlap_cache,
                scene_index=(
                    shared_window_specs[object_index][0].scene_index
                    if shared_window_specs[object_index]
                    else 0
                ),
                skip_count=0,
                scenes=scene_sources[object_index],
                prompt_source=prompt_sources[object_index],
                negative_prompt_source=negative_prompt_sources[object_index],
            )
        )

    context.object_states = object_states
    context.final_window_output_cache = (
        object_states[-1].output_cache if object_states else None
    )
    context.task_state_history = []
    register_runtime_task_chain_hooks_fn(context, object_states)
    for state in object_states:
        record_runtime_event_fn(
            context,
            "task_state_initialized",
            task_state=state,
            input_role=state.input_role,
            scene_index=state.scene_index,
            window_count=state.window_count,
            raw_input_cache_impl=window_cache_impl_name_fn(state.raw_input_cache),
            modified_output_cache_impl=window_cache_impl_name_fn(
                state.modified_output_cache
            ),
            overlap_output_cache_impl=window_cache_impl_name_fn(
                state.overlap_output_cache
            ),
        )
        record_task_state_snapshot_fn(context, state, phase="initialized")
    return object_states, total_windows
