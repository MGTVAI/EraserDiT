"""EraserDiT erase pipeline with full-video window commit.

The whole model layer is the ported
EraserDiT algorithm, the windowing / commit / IO layer is the shared
``pipelines.runtime`` runtime.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Any

import torch

from config.eraserdit import EraserDiTEraseSamplingParams, EraserDiTPipelineConfig
from config.server_args import ServerArgs
from memory.adapters.model_memory_adapter import ModelMemoryAdapter
from memory.adapters.eraserdit_memory_adapter import EraserDiTMemoryAdapter
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
from pipelines.runtime.context import prepare_ltx095_runtime_context
from pipelines.runtime.contracts import (
    LTX095EraseRuntimeContext,
    _is_windowed_runtime_mode,
)
from pipelines.runtime.drivers.windowed import run_ltx095_windowed_runtime
from pipelines.runtime.io.output import (
    close_ltx095_runtime_resources,
    finalize_ltx095_output,
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
from pipelines.runtime.windowing.handlers import (
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

    # Adapter-provided service contract (schema, sampling builder, capability).
    from config.service_contracts.eraserdit import ERASERDIT_SERVICE_CONTRACT as service_contract
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

    def load_modules(self, server_args, loaded_modules=None):
        from models.dits.eraserdit_quantization import validate_quantization
        validate_quantization(server_args)
        from parallel.eraserdit_cfg import validate_cfg_parallel
        validate_cfg_parallel(server_args)
        from parallel.eraserdit_mesh import resolve_mesh
        resolve_mesh(server_args)
        policy = server_args.resolve_resource_policy()
        if policy.requested_dynamic_offload and (
            torch.device(server_args.device).type != "cuda" or not torch.cuda.is_available()
        ):
            raise ValueError("EraserDiT dynamic_offload requires an available CUDA device")
        if policy.dit_cpu_offload and server_args.enable_torch_compile:
            raise ValueError(
                "EraserDiT transformer CPU offload with torch.compile is not yet "
                "validated; disable torch.compile for CPU offload"
            )
        modules = super().load_modules(server_args, loaded_modules)
        # Preloaded components bypass the component loaders' target-device logic.
        for name, enabled in (
            ("text_encoder", policy.text_encoder_cpu_offload),
            ("transformer", policy.dit_cpu_offload),
            ("vae", policy.vae_cpu_offload),
        ):
            if enabled:
                modules[name].to(device="cpu")
        if server_args.transformer_quantization == "int8_w8a8_native":
            from models.dits.eraserdit_quantization import quantize_transformer
            report = quantize_transformer(modules["transformer"], server_args.pipeline_config.quantization_scope)
            server_args.effective_transformer_quantization = report["mode"]
            server_args.transformer_quantization_report = report
        return modules

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
                    server_args=server_args,
                ),
                EraserDiTEraseDecodingStage(),
                EraserDiTEraseWindowPostprocessStage(),
                EraserDiTEraseWindowCommitSyncStage(),
            ]
        )

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        self._closed = False
        self._memory_adapter = self._build_memory_adapter()
        policy = server_args.resolve_resource_policy()
        if policy.dynamic_offload or policy.pin_memory:
            self._memory_adapter.register(
                modules=self.modules,
                device=torch.device(server_args.device),
                dynamic_offload=policy.dynamic_offload,
                pin_memory=policy.pin_memory,
                max_weight_usage=policy.max_weight_usage,
                rank=0,
            )
        self.memory_registration_summary = self._memory_adapter.snapshot()
        self.memory_registration_summary["resource_policy"] = (
            server_args.resolve_resource_policy().as_dict()
        )
        self._report_attention_backend(server_args)

    def _report_attention_backend(self, server_args: ServerArgs) -> None:
        """Resolve the self-attention backend once, at startup.

        ``auto`` may fall back (Sage -> Flash -> SDPA); the resolved value and
        the reasons are what ``/server_info`` must report (plan §M2/§M3).
        """
        transformer = self.get_module("transformer")
        block = getattr(transformer, "transformer_blocks", None)
        processor = None
        if block:
            processor = getattr(getattr(block[0], "attn1", None), "processor", None)
        if processor is None or not hasattr(
            processor, "preflight_self_attention_backend"
        ):
            self.attention_backend_report = {
                "requested": str(getattr(server_args, "attention_backend", "sdpa")),
                "effective": None,
                "fallback_reasons": ["transformer has no backend-aware processor"],
                "fallback_count": 1,
            }
        else:
            self.attention_backend_report = dict(
                processor.preflight_self_attention_backend(
                    device=torch.device(server_args.device),
                    dtype=server_args.resolve_component_dtype("transformer")
                    or torch.bfloat16,
                    head_size=64,
                    num_heads=32,
                )
            )
        server_args.attention_backend_report = dict(self.attention_backend_report)
        logger.info("Attention backend preflight: %s", self.attention_backend_report)

    def _build_memory_adapter(self) -> ModelMemoryAdapter:
        return EraserDiTMemoryAdapter()

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
        from pipelines.runtime.io.streaming import (
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
        from models.dits.eraserdit_quantization import validate_quantization, runtime_report
        validate_quantization(server_args, batch)
        from parallel.eraserdit_cfg import validate_cfg_parallel
        validate_cfg_parallel(server_args, batch)
        from parallel.eraserdit_mesh import resolve_mesh
        resolve_mesh(server_args, batch)
        if getattr(self, "_closed", False):
            raise RuntimeError("EraserDiTErasePipeline is closed")
        params = _as_eraserdit_params(batch)
        batch.modules = self.modules if not batch.modules else batch.modules
        from config.eraserdit_cache import resolve_eraserdit_cache_params
        resolve_eraserdit_cache_params(
            params, enable_torch_compile=server_args.enable_torch_compile,
            num_blocks=len(batch.modules["transformer"].transformer_blocks),
        )
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
            result.extra["quantization"] = runtime_report(batch.modules["transformer"])
            return result
        finally:
            try:
                close_ltx095_runtime_resources(context)
            finally:
                batch.extra["memory_runtime"] = self._memory_adapter.snapshot()
                batch.extra[TASK_STATE_KEY] = None


EntryClass = EraserDiTErasePipeline
