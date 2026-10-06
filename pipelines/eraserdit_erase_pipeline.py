"""EraserDiT erase pipeline with full-video window commit.

The whole model layer is the ported
EraserDiT algorithm, the windowing / commit / IO layer is the shared
``pipelines.runtime`` runtime.
"""

from __future__ import annotations

import time
from functools import partial

import torch

from config.eraserdit import EraserDiTEraseSamplingParams, EraserDiTPipelineConfig
from config.server_args import ServerArgs
from pipelines.base import ComposedPipelineBase
from nodes.schedule_batch import Req
from pipelines.stages.eraserdit_erase import (
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
from pipelines.stages.eraserdit_erase._common import (
    TASK_STATE_KEY,
    EraserDiTTaskState,
)
from utils.inference_timing import record_diagnostic_stage
from utils.logging_utils import init_logger
from utils.video_io import (
    read_mask_rgb_array as _read_mask_rgb_array,
)
from pipelines.runtime.context import prepare_runtime_context
from pipelines.runtime.contracts import (
    EraseRuntimeContext,
    _is_windowed_runtime_mode,
)
from pipelines.runtime.drivers.windowed import run_windowed_runtime
from pipelines.runtime.io.output import (
    close_runtime_resources,
    finalize_output,
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
        from models.adapters.eraserdit.cfg import validate_cfg_parallel
        validate_cfg_parallel(server_args)
        from models.adapters.eraserdit.mesh import resolve_mesh
        resolve_mesh(server_args)
        policy = server_args.resolve_resource_policy()
        from memory.validation import validate_memory_config
        validate_memory_config(server_args)
        from memory.telemetry import memory_observation
        self._initialization_memory = {
            "before_loading": memory_observation(server_args.device),
        }
        modules = super().load_modules(server_args, loaded_modules)
        # Preloaded components bypass the component loaders' target-device logic.
        for name, enabled in (
            ("text_encoder", policy.text_encoder_cpu_offload),
            ("transformer", policy.dit_cpu_offload or policy.dit_layerwise_offload),
            ("vae", policy.vae_cpu_offload),
        ):
            if enabled:
                modules[name].to(device="cpu")
        if (server_args.transformer_quantization != "none"
                and server_args.pipeline_config.dit_parallel_backend != "nccl"):
            from models.dits.eraserdit_quantization import quantize_transformer
            report = quantize_transformer(modules["transformer"], server_args.pipeline_config.quantization_scope,
                                          execution_device=server_args.device, mode=server_args.transformer_quantization)
            server_args.effective_transformer_quantization = report["mode"]
            server_args.transformer_quantization_report = report
        self._initialization_memory["after_loading"] = memory_observation(server_args.device)
        return modules

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self._component_compile = None
        if server_args.compile_components:
            from layers.component_compile import ComponentCompileManager
            from nodes.stages.denoising import resolve_torch_compile_mode
            self._component_compile = ComponentCompileManager(
                self.modules, server_args.compile_components, mode=resolve_torch_compile_mode())
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

        from memory.telemetry import memory_observation
        from models.adapters.eraserdit.mesh import resolve_mesh
        plan = resolve_mesh(server_args)
        devices = plan['devices'] if plan else [torch.device(server_args.device)]
        self._initialization_memory['after_replica_initialization'] = {
            str(device): memory_observation(device) for device in devices}

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        self._closed = False
        from memory.adapters.sglang_memory_adapter import SGLangMemoryAdapter
        self._memory_adapter = SGLangMemoryAdapter(self.modules, server_args)
        self.memory_registration_summary = self._memory_adapter.snapshot()
        self.memory_registration_summary["resource_policy"] = (
            server_args.resolve_resource_policy().as_dict()
        )
        self._report_attention_backend(server_args)
        from memory.telemetry import memory_observation
        self._initialization_memory["after_registration"] = memory_observation(server_args.device)
        self.memory_registration_summary["initialization_memory"] = self._initialization_memory

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

    def close(self, *, terminal: bool = False) -> dict[str, object]:
        if getattr(self, "_closed", False):
            return dict(getattr(self, "_memory_adapter_shutdown_snapshot", None) or {})
        snapshot = {}
        try:
            for stage in self.stages:
                close = getattr(stage, "close", None)
                if callable(close):
                    close()
        finally:
            try:
                component_compile = getattr(self, "_component_compile", None)
                if component_compile is not None:
                    component_compile.close()
                memory_adapter = getattr(self, "_memory_adapter", None)
                snapshot = memory_adapter.shutdown(terminal=terminal) if memory_adapter else {}
            finally:
                self._memory_adapter_shutdown_snapshot = snapshot
                self._closed = True
                # Sessions own these references; do not retain closed FSDP models,
                # CUDA modules or stage-held encoders in a reusable service object.
                self._stages.clear()
                self._stage_name_mapping.clear()
                self.modules.clear()
        return dict(snapshot)

    def _prepare_global_context(
        self, batch: Req, server_args: ServerArgs
    ) -> EraseRuntimeContext:
        params = _as_eraserdit_params(batch)
        return prepare_runtime_context(
            batch=batch,
            params=params,
            server_args=server_args,
            resource_policy=server_args.resolve_resource_policy(),
            # The baseline thresholds each RGB channel of the raw mask stream at
            # ``255/2 * mask_threshold`` (utils/pre.py:257).  The shared readers
            # default to 0.3*max and to a luma decode, either of which flips a
            # small number of pixels.
            read_mask_array=partial(
                _read_mask_rgb_array,
                threshold_ratio=float(params.mask_threshold) / 2.0,
            ),
            memory_adapter=self._memory_adapter,
        )

    def _maybe_save_output(
        self, batch: Req, context: EraseRuntimeContext
    ) -> None:
        params = _as_eraserdit_params(batch)
        finalize_output(
            batch=batch,
            context=context,
            params=params,
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
        from models.adapters.eraserdit.cfg import validate_cfg_parallel
        validate_cfg_parallel(server_args, batch)
        from models.adapters.eraserdit.mesh import resolve_mesh
        resolve_mesh(server_args, batch)
        if getattr(self, "_closed", False):
            raise RuntimeError("EraserDiTErasePipeline is closed")
        params = _as_eraserdit_params(batch)
        batch.modules = self.modules if not batch.modules else batch.modules
        from memory.validation import validate_memory_config
        validate_memory_config(server_args, batch)
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
            result = run_windowed_runtime(
                executor=self.executor, stages=self.stages, batch=batch,
                context=context, params=params, server_args=server_args, logger=logger,
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
            manager = getattr(self, "_component_compile", None)
            result.extra["component_compile"] = manager.snapshot() if manager else {}
            return result
        finally:
            try:
                close_runtime_resources(context)
            finally:
                batch.extra["memory_runtime"] = self._memory_adapter.snapshot()
                batch.extra[TASK_STATE_KEY] = None


EntryClass = EraserDiTErasePipeline
