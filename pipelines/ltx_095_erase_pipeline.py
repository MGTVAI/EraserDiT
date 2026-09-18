"""LTX0.9.5 erase pipeline with full-video window commit."""

from __future__ import annotations

import os
import time
from typing import Any

import torch

from config.ltx095 import LTX095EraseSamplingParams, LTX095PipelineConfig
from config.server_args import ServerArgs
from memory.adapters.model_memory_adapter import ModelMemoryAdapter
from nodes.composed_pipeline_base import ComposedPipelineBase
from nodes.schedule_batch import Req
from nodes.stages.model_specific_stages.ltx095_erase import (
    LTX095EraseConditionEncodingStage,
    LTX095EraseDecodingStage,
    LTX095EraseDenoisingStage,
    LTX095EraseLatentPreparationStage,
    LTX095ErasePreprocessStage,
    LTX095EraseSequenceParallelPrepareSyncStage,
    LTX095EraseTextEncodingStage,
    LTX095EraseTimestepPreparationStage,
    LTX095EraseWindowPostprocessStage,
    LTX095EraseWindowCommitSyncStage,
    LTX095EraseWindowValidationStage,
)
from nodes.stages.model_specific_stages.ltx095_erase._common import (
    _offload_module,
    _onload_module,
    _should_skip_writer_only_stage,
)
from videoerase.drivers.full import run_ltx095_full_runtime
from videoerase.io.output import (
    close_ltx095_runtime_resources,
    finalize_ltx095_output,
)
from videoerase.context import prepare_ltx095_runtime_context
from videoerase.events import (
    record_runtime_event as runtime_record_event,
    record_task_state_snapshot as runtime_record_task_state_snapshot,
    update_window_state as runtime_update_window_state,
)
from videoerase.scheduler import RuntimeTaskScheduler
from videoerase.tracks import (
    _resolve_object_scenes,
    _resolve_object_value,
)
from videoerase.contracts import (
    LTX095EraseRuntimeContext,
    ObjectRuntimeState,
    _is_windowed_runtime_mode,
)
from videoerase.io.streaming import (
    ensure_window_cache_loaded as runtime_ensure_window_cache_loaded,
    evict_cache_before as runtime_evict_cache_before,
    flush_windowed_frames as runtime_flush_windowed_frames,
    materialize_object_window_mask as runtime_materialize_object_window_mask,
    release_mask_frames as runtime_release_mask_frames,
)
from videoerase.windowing.cache_ops import (
    append_passthrough_gap as runtime_append_passthrough_gap,
    create_empty_cache_like as runtime_create_empty_cache_like,
    register_runtime_task_chain_hooks as runtime_register_task_chain_hooks,
    set_object_overlap_cache as runtime_set_object_overlap_cache,
)
from videoerase.windowing.commit_ops import (
    commit_ltx095_window,
    commit_ltx095_window_to_object_output,
    record_ltx095_skipped_object_window,
    record_ltx095_skipped_window,
)
from videoerase.windowing.materializer import (
    cache_ltx095_window_text_embeddings,
)
from videoerase.windowing.planner import build_ltx095_window_specs
from videoerase.drivers.windowed import (
    finalize_ltx095_object_window_step,
    maybe_flush_ltx095_streaming_runtime,
    object_input_ready,
    run_ltx095_windowed_runtime,
)
from utils.inference_timing import record_diagnostic_stage
from utils.ltx095_text import encode_ltx095_text_pair
from utils.logging_utils import init_logger
from utils.video_io import (
    ArrayFrameCache,
    ChunkedFrameCache,
    TensorFrameCache,
    WindowedVideoStore,
    binarize_mask_tensor,
    ensure_nchw_video,
    read_mask_array,
    read_mask_tensor,
    read_video_array,
    read_video_metadata,
    read_video_tensor,
)
from utils.windowing import WindowSpec
from videoerase.windowing.handlers import (
    _build_runtime_mask,
    _build_runtime_video,
    _commit_runtime_window_to_object_output,
    _create_runtime_empty_cache_like,
    _ensure_5d_video,
    _ensure_runtime_window_cache_loaded,
    _evict_runtime_cache_before,
    _finalize_runtime_object_window_step,
    _flush_runtime_windowed_frames,
    _materialize_runtime_object_window_mask,
    _module_device,
    _record_runtime_event,
    _record_runtime_skipped_object_window,
    _record_task_state_snapshot,
    _register_runtime_task_chain_hooks,
    _release_runtime_mask_frames,
    _select_sequence_item,
    _synchronize_timing_device,
    _update_runtime_window_state,
    _window_cache_impl_name,
)

logger = init_logger(__name__)


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_progress_bar(completed: int, total: int, width: int = 24) -> str:
    total = max(int(total), 1)
    completed = max(0, min(int(completed), total))
    filled = int(round(width * completed / total))
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _as_ltx095_params(batch: Req) -> LTX095EraseSamplingParams:
    params = batch.sampling_params
    if not isinstance(params, LTX095EraseSamplingParams):
        raise TypeError(
            "LTX095ErasePipeline requires LTX095EraseSamplingParams, "
            f"got {type(params).__name__}"
        )
    return params

class LTX095ErasePipeline(ComposedPipelineBase):
    pipeline_name = "LTX095ErasePipeline"
    pipeline_config_cls = LTX095PipelineConfig
    sampling_params_cls = LTX095EraseSamplingParams

    _required_config_modules = [
        "tokenizer",
        "text_encoder",
        "vae",
        "transformer",
        "scheduler",
    ]

    @property
    def required_config_modules(self) -> list[str]:
        server_args = self.server_args
        official_parallel_context = getattr(
            server_args, "official_parallel_context", None
        )
        distributed_context = getattr(server_args, "distributed_context", None)
        if (
            official_parallel_context is not None
            and distributed_context is not None
            and bool(official_parallel_context.enabled)
            and str(official_parallel_context.distributed_compute_mode)
            == "official_vae_parallel"
            and bool(getattr(official_parallel_context, "vae_parallel_supported", True))
            and bool(getattr(official_parallel_context, "vae_parallel_enabled", False))
            and not bool(distributed_context.is_writer_rank)
        ):
            return list(official_parallel_context.non_writer_required_modules)
        return list(self._required_config_modules)

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stages(
            [
                LTX095EraseWindowValidationStage(),
                LTX095EraseTextEncodingStage(
                    text_encoder=self.get_module("text_encoder"),
                    tokenizer=self.get_module("tokenizer"),
                ),
                LTX095ErasePreprocessStage(),
                LTX095EraseConditionEncodingStage(),
                LTX095EraseSequenceParallelPrepareSyncStage(),
                LTX095EraseLatentPreparationStage(
                    scheduler=self.get_module("scheduler"),
                ),
                LTX095EraseTimestepPreparationStage(),
                LTX095EraseDenoisingStage(
                    transformer=self.get_module("transformer"),
                    scheduler=self.get_module("scheduler"),
                    pipeline=self,
                    server_args=server_args,
                ),
                LTX095EraseDecodingStage(),
                LTX095EraseWindowPostprocessStage(),
                LTX095EraseWindowCommitSyncStage(),
            ]
        )

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        self._closed = False
        self._memory_adapter_shutdown_snapshot: dict[str, object] | None = None
        self._transformer_profiler = None
        requested_quantization = str(
            getattr(server_args, "transformer_quantization", "none")
        )
        if requested_quantization == "none":
            quantization_report = {
                "requested": "none",
                "effective": "none",
                "quantized_count": 0,
                "skipped_count": 0,
            }
        elif requested_quantization in {
            "fp8_w8a8", "fp8_w8a8_triton_selective",
        }:
            from layers.quantization import FP8LinearConfig
            from models.dits.ltx095_quantization import (
                quantize_ltx095_transformer,
            )
            from parallel.stage_policy import synchronize_stage_error

            quantization_error = None
            quantization_result = None
            try:
                fused_selective = (
                    requested_quantization == "fp8_w8a8_triton_selective"
                )
                config = FP8LinearConfig.from_values(
                    backend=str(server_args.fp8_linear_backend),
                    granularity=str(server_args.fp8_linear_granularity),
                    activation_quantization=(
                        "triton_per_row" if fused_selective else "eager"
                    ),
                    use_fast_accum=bool(server_args.fp8_fast_accum),
                    strict=True,
                )
                quantization_result = quantize_ltx095_transformer(
                    self.get_module("transformer"),
                    config,
                    execution_device=torch.device(server_args.device),
                    policy=(
                        "fused_selective" if fused_selective else "all"
                    ),
                )
            except Exception as error:
                quantization_error = error
            synchronize_stage_error(
                quantization_error,
                getattr(server_args, "parallel_context", None),
            )
            if quantization_result is None:
                raise RuntimeError("LTX095 FP8 quantization produced no report")
            quantization_report = quantization_result.as_dict()
        elif requested_quantization == "int8_w8a8_viditq":
            from models.dits.ltx095_viditq_quantization import (
                quantize_ltx095_transformer_viditq,
            )
            from parallel.stage_policy import synchronize_stage_error

            quantization_error = None
            quantization_result = None
            try:
                quantization_result = quantize_ltx095_transformer_viditq(
                    self.get_module("transformer"),
                    viditq_root=os.environ.get("VIDITQ_ROOT", "/root/viditq"),
                    execution_device=torch.device(server_args.device),
                    collect_runtime_timing=(
                        os.environ.get("MGERASE_VIDITQ_LINEAR_PROFILE", "0") == "1"
                    ),
                )
            except Exception as error:
                quantization_error = error
            synchronize_stage_error(
                quantization_error,
                getattr(server_args, "parallel_context", None),
            )
            if quantization_result is None:
                raise RuntimeError("LTX095 ViDiT-Q quantization produced no report")
            quantization_report = quantization_result.as_dict()
        else:
            raise ValueError(
                f"unsupported Transformer quantization mode: {requested_quantization}"
            )
        self.transformer_quantization_report = dict(quantization_report)
        server_args.transformer_quantization_report = dict(quantization_report)
        server_args.effective_transformer_quantization = str(
            quantization_report["effective"]
        )
        logger.info("Transformer quantization: %s", quantization_report)
        requested_text_quantization = str(
            getattr(server_args, "text_encoder_quantization", "none")
        )
        text_quantization_start = time.perf_counter()
        if requested_text_quantization == "none":
            text_quantization_report = {
                "requested": "none",
                "effective": "none",
                "policy_id": None,
                "quantized_count": 0,
                "skipped_count": 0,
            }
        elif requested_text_quantization == "int8_w8a8_viditq":
            from models.text_encoders.ltx095_t5_viditq_production import (
                quantize_ltx095_t5_viditq_production,
            )
            from parallel.stage_policy import synchronize_stage_error

            text_quantization_error = None
            text_quantization_result = None
            try:
                text_quantization_result = quantize_ltx095_t5_viditq_production(
                    self.get_module("text_encoder"),
                    execution_device=torch.device(server_args.device),
                    viditq_root=os.environ.get("VIDITQ_ROOT", "/root/viditq"),
                    collect_runtime_timing=(
                        os.environ.get("MGERASE_VIDITQ_LINEAR_PROFILE", "0") == "1"
                    ),
                )
            except Exception as error:
                text_quantization_error = error
            synchronize_stage_error(
                text_quantization_error,
                getattr(server_args, "parallel_context", None),
            )
            if text_quantization_result is None:
                raise RuntimeError("LTX095 T5 ViDiT-Q quantization produced no report")
            text_quantization_report = text_quantization_result.as_dict()
        else:
            raise ValueError(
                f"unsupported text encoder quantization mode: {requested_text_quantization}"
            )
        text_quantization_report["text_encoder_quantization_seconds"] = (
            time.perf_counter() - text_quantization_start
        )
        self.text_encoder_quantization_report = dict(text_quantization_report)
        server_args.text_encoder_quantization_report = dict(text_quantization_report)
        server_args.effective_text_encoder_quantization = str(
            text_quantization_report["effective"]
        )
        logger.info("Text encoder quantization: %s", text_quantization_report)
        requested_attention_backend = str(getattr(server_args, "attention_backend", "sdpa"))
        if requested_attention_backend == "sage_fp8":
            transformer = self.get_module("transformer")
            processor = getattr(transformer, "_shared_attention_processor", None)
            if str(getattr(processor, "attention_backend", "")) != "sage_fp8":
                raise RuntimeError(
                    "LTX095 strict sage_fp8 request did not reach the attention processor"
                )
            if processor is None or not callable(
                getattr(processor, "preflight_self_attention_backend", None)
            ):
                raise RuntimeError("LTX095 Transformer has no strict attention preflight")
            attention_backend_report = processor.preflight_self_attention_backend(
                device=torch.device(server_args.device),
                dtype=(
                    server_args.resolve_component_dtype("transformer")
                    or torch.bfloat16
                ),
                head_size=64,
                num_heads=32,
            )
        else:
            attention_backend_report = {
                "requested": requested_attention_backend,
                "effective": None,
                "cross_attention_backend": "torch_sdpa",
                "fallback_reasons": [],
                "fallback_count": 0,
            }
        self.attention_backend_report = dict(attention_backend_report)
        server_args.attention_backend_report = dict(attention_backend_report)
        logger.info("Attention backend preflight: %s", attention_backend_report)
        vae = self.get_module("vae")
        official_parallel_context = getattr(
            server_args, "official_parallel_context", None
        )
        if vae is not None and official_parallel_context is not None:
            if hasattr(vae, "disable_parallel"):
                vae.disable_parallel()
            vae_parallel_supported = bool(
                getattr(vae, "supports_official_parallel", True)
            )
            official_parallel_context.vae_parallel_supported = (
                vae_parallel_supported
            )
            if not vae_parallel_supported:
                official_parallel_context.vae_parallel_enabled = False
                official_parallel_context.vae_parallel_degree = 1
                if (
                    str(official_parallel_context.distributed_compute_mode)
                    == "official_vae_parallel"
                ):
                    official_parallel_context.vae_parallel_mode = (
                        "unsupported_official_model"
                    )
            elif bool(official_parallel_context.vae_parallel_enabled) and hasattr(
                vae, "enable_parallel"
            ):
                vae.enable_parallel(
                    max_worker=int(official_parallel_context.vae_parallel_degree),
                    latent_overlap=4,
                )

        policy = server_args.resolve_resource_policy()
        distributed_context = getattr(server_args, "distributed_context", None)
        memory_modules = self._memory_modules_for_rank(
            server_args,
            dynamic_offload=policy.dynamic_offload,
        )
        self._memory_adapter = self._build_memory_adapter()
        summary = self._memory_adapter.register(
            modules=memory_modules,
            device=torch.device(server_args.device),
            dynamic_offload=policy.dynamic_offload,
            pin_memory=policy.pin_memory,
            max_weight_usage=policy.max_weight_usage,
            rank=int(getattr(distributed_context, "rank", 0)),
        )
        self.memory_registration_summary = summary.as_dict()
        logger.info("Model memory registration: %s", self.memory_registration_summary)
        profile_detail = os.environ.get("MGERASE_LTX095_TRANSFORMER_PROFILE", "").strip()
        if profile_detail:
            from profiling.cuda_module_profiler import (
                build_ltx095_cuda_profiler,
            )

            self._transformer_profiler = build_ltx095_cuda_profiler(
                self.get_module("transformer"), detail=profile_detail
            )

    def _memory_modules_for_rank(
        self,
        server_args: ServerArgs,
        *,
        dynamic_offload: bool,
    ) -> dict[str, object]:
        if not dynamic_offload:
            return self.modules

        selected = {
            name: self.modules[name]
            for name in ("transformer",)
            if name in self.modules
        }
        parallel_context = getattr(server_args, "parallel_context", None)
        plan = getattr(parallel_context, "plan", None)
        vae_parallel_active = int(getattr(plan, "vae_degree", 1) or 1) > 1
        official_context = getattr(
            server_args,
            "official_parallel_context",
            None,
        )
        vae_parallel_active = vae_parallel_active or bool(
            official_context is not None
            and getattr(official_context, "enabled", False)
            and getattr(official_context, "vae_parallel_supported", True)
            and getattr(official_context, "vae_parallel_enabled", False)
        )
        distributed_context = getattr(
            server_args,
            "distributed_context",
            None,
        )
        is_writer_rank = bool(
            getattr(distributed_context, "is_writer_rank", True)
        )
        if (
            is_writer_rank
            and str(getattr(server_args, "effective_text_encoder_quantization", "none"))
            == "int8_w8a8_viditq"
            and "text_encoder" in self.modules
        ):
            selected["text_encoder"] = self.modules["text_encoder"]
        if (
            (vae_parallel_active or is_writer_rank)
            and "vae" in self.modules
        ):
            selected["vae"] = self.modules["vae"]
        return selected

    def _build_memory_adapter(self) -> ModelMemoryAdapter:
        return ModelMemoryAdapter()

    def _snapshot_memory_adapter(self, batch) -> None:
        memory_adapter = getattr(self, "_memory_adapter", None)
        if memory_adapter is not None:
            batch.extra["memory_runtime_summary"] = (
                memory_adapter.snapshot()
            )

    def close(self, *, terminal: bool = False) -> dict[str, object]:
        if getattr(self, "_closed", False):
            return dict(
                getattr(self, "_memory_adapter_shutdown_snapshot", None)
                or {}
            )
        memory_adapter = getattr(self, "_memory_adapter", None)
        if memory_adapter is None:
            snapshot: dict[str, object] = {}
        else:
            snapshot = memory_adapter.shutdown(terminal=terminal)
        self._memory_adapter_shutdown_snapshot = snapshot
        self._closed = True
        return dict(snapshot)

    def _object_input_ready(
        self,
        object_state: ObjectRuntimeState,
        spec: WindowSpec,
    ) -> bool:
        return object_input_ready(object_state, spec)

    def _register_runtime_task_chain_hooks(
        self,
        context: LTX095EraseRuntimeContext,
        object_states: list[ObjectRuntimeState],
    ) -> None:
        _register_runtime_task_chain_hooks(context, object_states)

    def _materialize_object_window_mask(
        self,
        context: LTX095EraseRuntimeContext,
        object_state: ObjectRuntimeState,
        spec: WindowSpec,
        crop_bbox: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        return _materialize_runtime_object_window_mask(
            context=context,
            object_state=object_state,
            spec=spec,
            crop_bbox=crop_bbox,
        )

    def _prepare_global_context(
        self, batch: Req, server_args: ServerArgs
    ) -> LTX095EraseRuntimeContext:
        params = _as_ltx095_params(batch)
        resource_policy = server_args.resolve_resource_policy()
        return prepare_ltx095_runtime_context(
            batch=batch,
            params=params,
            server_args=server_args,
            resource_policy=resource_policy,
            build_runtime_video=_build_runtime_video,
            build_runtime_mask=_build_runtime_mask,
            read_video_metadata=read_video_metadata,
            read_video_array=read_video_array,
            read_mask_array=read_mask_array,
            window_store_builder=WindowedVideoStore,
            memory_adapter=self._memory_adapter,
        )

    def _prime_text_embedding_cache(
        self,
        batch: Req,
        context: LTX095EraseRuntimeContext,
        params: LTX095EraseSamplingParams,
        server_args: ServerArgs,
    ) -> None:
        if _should_skip_writer_only_stage(
            server_args,
            "LTX095EraseTextEmbeddingPrefill",
        ):
            return
        tokenizer = batch.modules.get("tokenizer")
        if tokenizer is None or batch.modules.get("text_encoder") is None:
            return

        pending_entries: list[tuple[int, int, str, str]] = []
        seen_keys = set(context.text_embedding_cache.keys())
        for object_index in range(context.object_count):
            object_scenes = _resolve_object_scenes(
                params.scenes,
                object_index=object_index,
                object_count=context.object_count,
            )
            window_specs = build_ltx095_window_specs(params, scenes=object_scenes)
            object_prompt_source = _resolve_object_value(
                params.prompt,
                object_index=object_index,
                object_count=context.object_count,
            )
            object_negative_prompt_source = _resolve_object_value(
                params.negative_prompt,
                object_index=object_index,
                object_count=context.object_count,
            )
            scene_indices = sorted(
                {int(spec.scene_index) for spec in window_specs}
            ) or [0]
            for scene_index in scene_indices:
                prompt = _select_sequence_item(object_prompt_source, scene_index) or ""
                negative_prompt = (
                    _select_sequence_item(object_negative_prompt_source, scene_index)
                    or ""
                )
                cache_key = (object_index, scene_index, prompt, negative_prompt)
                if cache_key in seen_keys:
                    continue
                seen_keys.add(cache_key)
                pending_entries.append(cache_key)

        if not pending_entries:
            return

        prefill_start = time.perf_counter()
        timing = {
            "text_encoder_quantization_seconds": float(
                getattr(self, "text_encoder_quantization_report", {}).get(
                    "text_encoder_quantization_seconds", 0.0
                )
            ),
            "text_encoder_onload_seconds": 0.0,
            "text_prompt_forward_seconds": 0.0,
            "text_negative_forward_seconds": 0.0,
            "text_embedding_cache_commit_seconds": 0.0,
            "text_encoder_offload_seconds": 0.0,
            "text_embedding_prefill_total_seconds": 0.0,
        }
        batch.extra["runtime_text_embedding_prefill_count"] = len(pending_entries)
        _record_runtime_event(
            context,
            "text_embedding_cache_prefill_begin",
            entry_count=len(pending_entries),
        )
        _synchronize_timing_device(getattr(server_args, "device", "cpu"))
        onload_start = time.perf_counter()
        text_encoder, policy = _onload_module(
            batch,
            server_args,
            "text_encoder",
        )
        _synchronize_timing_device(getattr(server_args, "device", "cpu"))
        timing["text_encoder_onload_seconds"] = time.perf_counter() - onload_start
        try:
            for object_index, scene_index, prompt, negative_prompt in pending_entries:
                embeddings = encode_ltx095_text_pair(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    pin_memory=policy.pin_memory,
                )
                embedding_timing = embeddings.get("timing_seconds", {})
                timing["text_prompt_forward_seconds"] += float(
                    embedding_timing.get("text_prompt_forward_seconds", 0.0)
                )
                timing["text_negative_forward_seconds"] += float(
                    embedding_timing.get("text_negative_forward_seconds", 0.0)
                )
                _synchronize_timing_device(getattr(server_args, "device", "cpu"))
                cache_start = time.perf_counter()
                cache_ltx095_window_text_embeddings(
                    context=context,
                    object_index=object_index,
                    scene_index=scene_index,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    prompt_embeds=embeddings["prompt_embeds"],
                    prompt_attention_mask=embeddings["prompt_attention_mask"],
                    negative_prompt_embeds=embeddings["negative_prompt_embeds"],
                    negative_attention_mask=embeddings["negative_attention_mask"],
                    record_runtime_event_fn=_record_runtime_event,
                )
                _synchronize_timing_device(getattr(server_args, "device", "cpu"))
                timing["text_embedding_cache_commit_seconds"] += (
                    time.perf_counter() - cache_start
                )
        finally:
            _synchronize_timing_device(getattr(server_args, "device", "cpu"))
            offload_start = time.perf_counter()
            _offload_module(
                batch,
                server_args,
                "text_encoder",
                policy,
                reason="text_embedding_cache_prefill",
            )
            _synchronize_timing_device(getattr(server_args, "device", "cpu"))
            timing["text_encoder_offload_seconds"] = (
                time.perf_counter() - offload_start
            )
        timing["text_embedding_prefill_total_seconds"] = (
            time.perf_counter() - prefill_start
        )
        batch.extra["text_embedding_prefill_timing"] = timing
        _record_runtime_event(
            context,
            "text_embedding_cache_prefill_complete",
            entry_count=len(pending_entries),
        )
        if not batch.suppress_logs:
            logger.info(
                "prefilled %d text embedding cache entries before runtime",
                len(pending_entries),
            )

    def _commit_window_to_object_output(
        self,
        context: LTX095EraseRuntimeContext,
        object_state: ObjectRuntimeState,
        spec: WindowSpec,
        window_batch: Req,
    ) -> None:
        _commit_runtime_window_to_object_output(
            context=context,
            object_state=object_state,
            spec=spec,
            window_batch=window_batch,
        )

    def _record_skipped_object_window(
        self,
        context: LTX095EraseRuntimeContext,
        object_state: ObjectRuntimeState,
        spec: WindowSpec,
        reason: str,
        prompt: Any,
        negative_prompt: Any,
    ) -> None:
        _record_runtime_skipped_object_window(
            context=context,
            object_state=object_state,
            spec=spec,
            reason=reason,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )

    def _finalize_object_window_step(
        self,
        context: LTX095EraseRuntimeContext,
        object_states: list[ObjectRuntimeState],
        object_state: ObjectRuntimeState,
        spec: WindowSpec,
    ) -> None:
        _finalize_runtime_object_window_step(
            context=context,
            object_states=object_states,
            object_state=object_state,
            spec=spec,
        )

    def _maybe_flush_streaming_runtime(
        self,
        context: LTX095EraseRuntimeContext,
    ) -> None:
        maybe_flush_ltx095_streaming_runtime(
            context=context,
            evict_cache_before_fn=_evict_runtime_cache_before,
            release_mask_frames_fn=_release_runtime_mask_frames,
            flush_windowed_frames_fn=_flush_runtime_windowed_frames,
        )

    def _run_windowed_object_chain(
        self,
        batch: Req,
        context: LTX095EraseRuntimeContext,
        params: LTX095EraseSamplingParams,
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
            materialize_object_window_mask=_materialize_runtime_object_window_mask,
            record_skipped_object_window=_record_runtime_skipped_object_window,
            commit_window_to_object_output=_commit_runtime_window_to_object_output,
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

    def _run_full_runtime(
        self,
        batch: Req,
        context: LTX095EraseRuntimeContext,
        params: LTX095EraseSamplingParams,
        server_args: ServerArgs,
    ) -> Req:
        from dataclasses import dataclass, field
        from typing import Callable

        @dataclass
        class _FullHandlers:
            window_cache_impl_name: Callable = field(default=lambda _: "none")
            select_sequence_item: Callable = field(default=lambda v, i: v)
            ensure_window_cache_loaded: Callable = field(default=lambda *a, **kw: None)
            record_runtime_event: Callable = field(default=lambda *a, **kw: None)
            record_skipped_window: Callable = field(default=lambda *a, **kw: None)
            commit_window: Callable = field(default=lambda *a, **kw: None)
            evict_cache_before: Callable = field(default=lambda *a, **kw: None)
            log_runtime_progress: Callable = field(default=lambda *a, **kw: None)

        handlers = _FullHandlers(
            window_cache_impl_name=_window_cache_impl_name,
            select_sequence_item=_select_sequence_item,
            ensure_window_cache_loaded=_ensure_runtime_window_cache_loaded,
            record_runtime_event=_record_runtime_event,
            record_skipped_window=record_ltx095_skipped_window,
            commit_window=self._commit_window,
            evict_cache_before=_evict_runtime_cache_before,
            log_runtime_progress=self._log_runtime_progress,
        )
        return run_ltx095_full_runtime(
            executor=self.executor,
            stages=self.stages,
            batch=batch,
            context=context,
            params=params,
            server_args=server_args,
            logger=logger,
            handlers=handlers,
        )

    def _commit_window(
        self,
        context: LTX095EraseRuntimeContext,
        spec: WindowSpec,
        window_batch: Req,
        object_index: int,
    ) -> None:
        commit_started = time.perf_counter()
        try:
            commit_ltx095_window(
                context=context,
                spec=spec,
                window_batch=window_batch,
                object_index=object_index,
            )
        finally:
            context.record_runtime_timing(
                "cache_commit", time.perf_counter() - commit_started
            )

    def _record_skipped_window(
        self,
        context: LTX095EraseRuntimeContext,
        spec: WindowSpec,
        object_index: int,
        reason: str,
        prompt: Any,
        negative_prompt: Any,
    ) -> None:
        record_ltx095_skipped_window(
            context=context,
            spec=spec,
            object_index=object_index,
            reason=reason,
            prompt=prompt,
            negative_prompt=negative_prompt,
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
        percent = 100.0 * completed_windows / max(total_windows, 1)
        bar = _format_progress_bar(completed_windows, total_windows)
        logger.info(
            "Progress %s %d/%d %.1f%% object=%d window=%d elapsed=%s eta=%s",
            bar,
            completed_windows,
            total_windows,
            percent,
            current_object_index + 1,
            current_window_index + 1,
            _format_duration(elapsed),
            _format_duration(remaining),
        )
        if context.progress_state is not None:
            object_window_count = len(context.window_specs)
            context.progress_state.update_pipeline(
                completed=completed_windows,
                total=total_windows,
                object_index=current_object_index,
                object_count=context.object_count,
                window_index=current_window_index,
                window_count=object_window_count,
            )

    def _maybe_save_output(
        self, batch: Req, context: LTX095EraseRuntimeContext
    ) -> None:
        params = _as_ltx095_params(batch)
        finalize_ltx095_output(
            batch=batch,
            context=context,
            params=params,
            flush_windowed_frames_fn=_flush_runtime_windowed_frames,
            release_mask_frames_fn=_release_runtime_mask_frames,
        )

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if getattr(self, "_closed", False):
            raise RuntimeError("LTX095ErasePipeline is closed")
        params = _as_ltx095_params(batch)
        batch.modules = self.modules if not batch.modules else batch.modules
        batch.extra["memory_registration_summary"] = dict(
            getattr(self, "memory_registration_summary", {})
        )
        batch.extra["transformer_quantization"] = dict(
            getattr(self, "transformer_quantization_report", {})
        )
        batch.extra["text_encoder_quantization"] = dict(
            getattr(self, "text_encoder_quantization_report", {})
        )
        batch.extra["attention_backend"] = dict(
            getattr(self, "attention_backend_report", {})
        )
        context_prepare_start = time.perf_counter()
        context = self._prepare_global_context(batch, server_args)
        record_diagnostic_stage(
            batch.metrics,
            "diagnostic.pipeline.context_prepare",
            time.perf_counter() - context_prepare_start,
        )
        try:
            runtime_driver_start = time.perf_counter()
            if _is_windowed_runtime_mode(context.runtime_mode):
                self._prime_text_embedding_cache(
                    batch,
                    context,
                    params,
                    server_args,
                )
                result = self._run_windowed_object_chain(
                    batch,
                    context,
                    params,
                    server_args,
                )
            else:
                result = self._run_full_runtime(
                    batch,
                    context,
                    params,
                    server_args,
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
            resource_close_start = time.perf_counter()
            try:
                close_ltx095_runtime_resources(context)
            finally:
                profiler = getattr(self, "_transformer_profiler", None)
                if profiler is not None:
                    profiler.close()
                    batch.extra["transformer_profile"] = profiler.as_dict()
                    self._transformer_profiler = None
                if (
                    str(getattr(server_args, "effective_transformer_quantization", "none"))
                    == "int8_w8a8_viditq"
                ):
                    from models.dits.ltx095_viditq_quantization import (
                        summarize_ltx095_viditq_runtime,
                    )

                    runtime = summarize_ltx095_viditq_runtime(
                        self.get_module("transformer")
                    )
                    report = dict(self.transformer_quantization_report)
                    report.update(runtime)
                    self.transformer_quantization_report = report
                    server_args.transformer_quantization_report = dict(report)
                    batch.extra["transformer_quantization"] = dict(report)
                if (
                    str(getattr(server_args, "effective_text_encoder_quantization", "none"))
                    == "int8_w8a8_viditq"
                ):
                    from models.text_encoders.ltx095_t5_viditq_quantization import (
                        summarize_ltx095_t5_viditq_runtime,
                    )

                    text_runtime = summarize_ltx095_t5_viditq_runtime(
                        self.get_module("text_encoder")
                    )
                    text_report = dict(self.text_encoder_quantization_report)
                    text_report.update(text_runtime)
                    self.text_encoder_quantization_report = text_report
                    server_args.text_encoder_quantization_report = dict(text_report)
                    batch.extra["text_encoder_quantization"] = dict(text_report)
                attention_processor = getattr(
                    self.get_module("transformer"),
                    "_shared_attention_processor",
                    None,
                )
                if attention_processor is not None and callable(
                    getattr(attention_processor, "attention_backend_report", None)
                ):
                    attention_report = attention_processor.attention_backend_report()
                    self.attention_backend_report = dict(attention_report)
                    server_args.attention_backend_report = dict(attention_report)
                    batch.extra["attention_backend"] = dict(attention_report)
                self._snapshot_memory_adapter(batch)
                record_diagnostic_stage(
                    batch.metrics,
                    "diagnostic.pipeline.resource_close",
                    time.perf_counter() - resource_close_start,
                )


EntryClass = LTX095ErasePipeline
