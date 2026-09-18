"""EraserDiT erase pipeline with full-video window commit.

Serial, single-GPU implementation of phase 1: the whole model layer is the ported
EraserDiT algorithm, the windowing / commit / IO layer is the shared
``videoerase`` runtime.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Any

import torch

from config.eraserdit import EraserDiTEraseSamplingParams, EraserDiTPipelineConfig
from config.server_args import ServerArgs
from memory.adapters.model_memory_adapter import ModelMemoryAdapter
from nodes.composed_pipeline_base import ComposedPipelineBase
from nodes.schedule_batch import Req
from nodes.stages.model_specific_stages.eraserdit_erase import (
    EraserDiTEraseConditionEncodingStage,
    EraserDiTEraseDecodingStage,
    EraserDiTEraseDenoisingStage,
    EraserDiTEraseLatentPreparationStage,
    EraserDiTErasePreprocessStage,
    EraserDiTEraseTextEncodingStage,
    EraserDiTEraseTimestepPreparationStage,
    EraserDiTEraseWindowCommitSyncStage,
    EraserDiTEraseWindowPostprocessStage,
    EraserDiTEraseWindowValidationStage,
)
from nodes.stages.model_specific_stages.eraserdit_erase._common import (
    TASK_STATE_KEY,
    EraserDiTTaskState,
)
from utils.inference_timing import record_diagnostic_stage
from utils.logging_utils import init_logger
from utils.video_io import (
    WindowedVideoStore,
    read_mask_rgb_array as _read_mask_rgb_array,
    read_video_array,
    read_video_metadata,
)
from videoerase.context import prepare_ltx095_runtime_context
from videoerase.contracts import (
    LTX095EraseRuntimeContext,
    _is_windowed_runtime_mode,
)
from videoerase.drivers.windowed import run_ltx095_windowed_runtime
from videoerase.io.output import (
    close_ltx095_runtime_resources,
    finalize_ltx095_output,
)
from videoerase.windowing.cache_ops import (
    append_passthrough_gap as runtime_append_passthrough_gap,
    create_empty_cache_like as runtime_create_empty_cache_like,
    register_runtime_task_chain_hooks as runtime_register_task_chain_hooks,
    set_object_overlap_cache as runtime_set_object_overlap_cache,
)
from videoerase.windowing.commit_ops import (
    commit_ltx095_window_to_object_output,
    record_ltx095_skipped_object_window,
)
from videoerase.windowing.handlers import (
    _build_runtime_mask,
    _build_runtime_video,
    _create_runtime_empty_cache_like,
    _ensure_runtime_window_cache_loaded,
    _evict_runtime_cache_before,
    _flush_runtime_windowed_frames,
    _record_runtime_event,
    _record_runtime_skipped_object_window,
    _record_task_state_snapshot,
    _register_runtime_task_chain_hooks,
    _release_runtime_mask_frames,
    _select_sequence_item,
    _update_runtime_window_state,
    _window_cache_impl_name,
)

logger = init_logger(__name__)


def _as_eraserdit_params(batch: Req) -> EraserDiTEraseSamplingParams:
    params = batch.sampling_params
    if not isinstance(params, EraserDiTEraseSamplingParams):
        raise TypeError(
            "EraserDiTErasePipeline requires EraserDiTEraseSamplingParams, "
            f"got {type(params).__name__}"
        )
    return params


class EraserDiTErasePipeline(ComposedPipelineBase):
    pipeline_name = "EraserDiTErasePipeline"
    pipeline_config_cls = EraserDiTPipelineConfig
    sampling_params_cls = EraserDiTEraseSamplingParams

    _required_config_modules = [
        "tokenizer",
        "text_encoder",
        "vae",
        "transformer",
        "scheduler",
    ]

    # The EraserDiT snapshot ships the five component folders but no root
    # ``model_index.json``, so the component map is declared here (plan §4.1).
    # The architecture names are informational; the actual classes come from
    # ``EraserDiTPipelineConfig.component_architectures``.
    _declared_model_index = {
        "tokenizer": ("transformers", "T5Tokenizer"),
        "text_encoder": ("transformers", "T5EncoderModel"),
        "vae": ("diffusers", "AutoencoderKLLTXVideo"),
        "transformer": ("diffusers", "LTXVideoTransformer3DModel"),
        "scheduler": ("diffusers", "FlowMatchEulerDiscreteScheduler"),
    }

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stages(
            [
                EraserDiTEraseWindowValidationStage(),
                EraserDiTEraseTextEncodingStage(
                    text_encoder=self.get_module("text_encoder"),
                    tokenizer=self.get_module("tokenizer"),
                ),
                EraserDiTErasePreprocessStage(),
                EraserDiTEraseConditionEncodingStage(),
                EraserDiTEraseLatentPreparationStage(
                    scheduler=self.get_module("scheduler"),
                ),
                EraserDiTEraseTimestepPreparationStage(),
                EraserDiTEraseDenoisingStage(
                    transformer=self.get_module("transformer"),
                    scheduler=self.get_module("scheduler"),
                ),
                EraserDiTEraseDecodingStage(),
                EraserDiTEraseWindowPostprocessStage(),
                EraserDiTEraseWindowCommitSyncStage(),
            ]
        )

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        del server_args
        self._closed = False
        self._memory_adapter = self._build_memory_adapter()
        self.memory_registration_summary = self._memory_adapter.snapshot()

    def _build_memory_adapter(self) -> ModelMemoryAdapter:
        return ModelMemoryAdapter()

    def close(self, *, terminal: bool = False) -> dict[str, object]:
        if getattr(self, "_closed", False):
            return dict(getattr(self, "_memory_adapter_shutdown_snapshot", None) or {})
        memory_adapter = getattr(self, "_memory_adapter", None)
        snapshot = (
            memory_adapter.shutdown(terminal=terminal) if memory_adapter else {}
        )
        self._memory_adapter_shutdown_snapshot = snapshot
        self._closed = True
        return dict(snapshot)

    def _prepare_global_context(
        self, batch: Req, server_args: ServerArgs
    ) -> LTX095EraseRuntimeContext:
        params = _as_eraserdit_params(batch)
        return prepare_ltx095_runtime_context(
            batch=batch,
            params=params,
            server_args=server_args,
            resource_policy=server_args.resolve_resource_policy(),
            build_runtime_video=_build_runtime_video,
            build_runtime_mask=_build_runtime_mask,
            read_video_metadata=read_video_metadata,
            read_video_array=read_video_array,
            # The baseline thresholds each RGB channel of the raw mask stream at
            # ``255/2 * mask_threshold`` (utils/pre.py:257).  The shared readers
            # default to 0.3*max and to a luma decode, either of which flips a
            # small number of pixels.
            read_mask_array=partial(
                _read_mask_rgb_array,
                threshold_ratio=float(params.mask_threshold) / 2.0,
            ),
            window_store_builder=WindowedVideoStore,
            memory_adapter=self._memory_adapter,
        )

    def _commit_window_to_object_output(
        self,
        context: LTX095EraseRuntimeContext,
        object_state: Any,
        spec: Any,
        window_batch: Req,
    ) -> None:
        started = time.perf_counter()
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
                "cache_commit", time.perf_counter() - started
            )

    def _run_windowed_object_chain(
        self,
        batch: Req,
        context: LTX095EraseRuntimeContext,
        params: EraserDiTEraseSamplingParams,
        server_args: ServerArgs,
    ) -> Req:
        from dataclasses import dataclass, field
        from typing import Callable

        @dataclass
        class _WindowedHandlers:
            window_cache_impl_name: Callable = field(default=lambda _: "none")
            select_sequence_item: Callable = field(default=lambda v, i: v)
            create_empty_cache_like: Callable = field(default=lambda *a, **kw: None)
            register_runtime_task_chain_hooks: Callable = field(
                default=lambda *a, **kw: None
            )
            record_runtime_event: Callable = field(default=lambda *a, **kw: None)
            record_task_state_snapshot: Callable = field(default=lambda *a, **kw: None)
            update_window_state: Callable = field(default=lambda *a, **kw: None)
            ensure_window_cache_loaded: Callable = field(default=lambda *a, **kw: None)
            materialize_object_window_mask: Callable = field(
                default=lambda *a, **kw: None
            )
            record_skipped_object_window: Callable = field(
                default=lambda *a, **kw: None
            )
            commit_window_to_object_output: Callable = field(
                default=lambda *a, **kw: None
            )
            set_object_overlap_cache: Callable = field(default=lambda *a, **kw: None)
            evict_cache_before: Callable = field(default=lambda *a, **kw: None)
            release_mask_frames: Callable = field(default=lambda *a, **kw: None)
            flush_windowed_frames: Callable = field(default=lambda *a, **kw: None)
            log_runtime_progress: Callable = field(default=lambda *a, **kw: None)

        handlers = _WindowedHandlers(
            window_cache_impl_name=_window_cache_impl_name,
            select_sequence_item=_select_sequence_item,
            create_empty_cache_like=_create_runtime_empty_cache_like,
            register_runtime_task_chain_hooks=_register_runtime_task_chain_hooks,
            record_runtime_event=_record_runtime_event,
            record_task_state_snapshot=_record_task_state_snapshot,
            update_window_state=_update_runtime_window_state,
            ensure_window_cache_loaded=_ensure_runtime_window_cache_loaded,
            materialize_object_window_mask=self._materialize_object_window_mask,
            record_skipped_object_window=_record_runtime_skipped_object_window,
            commit_window_to_object_output=self._commit_window_to_object_output,
            set_object_overlap_cache=runtime_set_object_overlap_cache,
            evict_cache_before=_evict_runtime_cache_before,
            release_mask_frames=_release_runtime_mask_frames,
            flush_windowed_frames=_flush_runtime_windowed_frames,
            log_runtime_progress=self._log_runtime_progress,
        )
        return run_ltx095_windowed_runtime(
            executor=self.executor,
            stages=self.stages,
            batch=batch,
            context=context,
            params=params,
            server_args=server_args,
            logger=logger,
            handlers=handlers,
        )

    def _materialize_object_window_mask(
        self,
        context: LTX095EraseRuntimeContext,
        object_state: Any,
        spec: Any,
        crop_bbox: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        from videoerase.io.streaming import (
            materialize_object_window_mask as runtime_materialize,
        )

        return runtime_materialize(
            context=context,
            object_state=object_state,
            spec=spec,
            ensure_window_cache_loaded_fn=_ensure_runtime_window_cache_loaded,
            crop_bbox=crop_bbox,
        )

    def _log_runtime_progress(
        self,
        context: LTX095EraseRuntimeContext,
        completed_windows: int,
        total_windows: int,
        current_object_index: int,
        current_window_index: int,
    ) -> None:
        if total_windows <= 0:
            return
        elapsed = time.time() - context.pipeline_start_time
        rate = elapsed / max(completed_windows, 1)
        remaining = max(total_windows - completed_windows, 0) * rate
        logger.info(
            "Progress %d/%d %.1f%% object=%d window=%d elapsed=%.1fs eta=%.1fs",
            completed_windows,
            total_windows,
            100.0 * completed_windows / max(total_windows, 1),
            current_object_index + 1,
            current_window_index + 1,
            elapsed,
            remaining,
        )
        if context.progress_state is not None:
            context.progress_state.update_pipeline(
                completed=completed_windows,
                total=total_windows,
                object_index=current_object_index,
                object_count=context.object_count,
                window_index=current_window_index,
                window_count=len(context.window_specs),
            )

    def _maybe_save_output(
        self, batch: Req, context: LTX095EraseRuntimeContext
    ) -> None:
        params = _as_eraserdit_params(batch)
        finalize_ltx095_output(
            batch=batch,
            context=context,
            params=params,
            flush_windowed_frames_fn=_flush_runtime_windowed_frames,
            release_mask_frames_fn=_release_runtime_mask_frames,
        )

    def _install_task_state(
        self, batch: Req, server_args: ServerArgs
    ) -> EraserDiTTaskState:
        params = _as_eraserdit_params(batch)
        device = torch.device(server_args.device)
        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(params.seed))
        state = EraserDiTTaskState(generator=generator)
        batch.extra[TASK_STATE_KEY] = state
        batch.generator = generator
        return state

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if getattr(self, "_closed", False):
            raise RuntimeError("EraserDiTErasePipeline is closed")
        params = _as_eraserdit_params(batch)
        batch.modules = self.modules if not batch.modules else batch.modules
        batch.extra["memory_registration_summary"] = dict(
            getattr(self, "memory_registration_summary", {})
        )
        self._install_task_state(batch, server_args)

        context_prepare_start = time.perf_counter()
        context = self._prepare_global_context(batch, server_args)
        record_diagnostic_stage(
            batch.metrics,
            "diagnostic.pipeline.context_prepare",
            time.perf_counter() - context_prepare_start,
        )
        try:
            if not _is_windowed_runtime_mode(context.runtime_mode):
                raise NotImplementedError(
                    "EraserDiT phase 1 only supports the windowed runtime; "
                    f"got runtime_mode={context.runtime_mode!r}. Set "
                    "runtime_mode='windowed_streaming'."
                )
            runtime_driver_start = time.perf_counter()
            result = self._run_windowed_object_chain(
                batch, context, params, server_args
            )
            record_diagnostic_stage(
                batch.metrics,
                "diagnostic.pipeline.runtime_driver",
                time.perf_counter() - runtime_driver_start,
            )
            output_finalize_start = time.perf_counter()
            self._maybe_save_output(result, context)
            record_diagnostic_stage(
                batch.metrics,
                "diagnostic.pipeline.output_finalize",
                time.perf_counter() - output_finalize_start,
            )
            return result
        finally:
            try:
                close_ltx095_runtime_resources(context)
            finally:
                batch.extra[TASK_STATE_KEY] = None


EntryClass = EraserDiTErasePipeline
