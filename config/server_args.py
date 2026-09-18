"""Minimal server/runtime arguments for the MGErase runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch

from utils.resource_policy import (
    RuntimeResourcePolicy,
    resolve_runtime_resource_policy,
)


def _default_pipeline_config() -> SimpleNamespace:
    return SimpleNamespace(flow_shift=None)


@dataclass
class ServerArgs:
    """Small runtime configuration shared by pipeline construction and stages."""

    model_path: str = ""
    component_paths: dict[str, str] = field(default_factory=dict)
    pipeline_class_name: str | None = None
    pipeline_config: Any = field(default_factory=_default_pipeline_config)
    revision: str | None = None
    trust_remote_code: bool = False
    device: str = "cpu"
    disable_autocast: bool = False
    weight_dtype: str | None = None
    component_dtypes: dict[str, str] = field(default_factory=dict)
    # Adapters may declare their component class names explicitly instead of
    # relying on the checkpoint's ``_class_name`` (plan §4.1).
    component_architectures: dict[str, str] = field(default_factory=dict)
    resource_policy: str = "fullgpu"
    dynamic_offload: bool = False
    pin_memory: bool = False
    max_weight_usage: int = 5 * 1024**3
    vae_cpu_offload: bool = False
    dit_cpu_offload: bool = False
    text_encoder_cpu_offload: bool = False
    comfyui_mode: bool = False
    transformer_weights_path: str | None = None
    use_fsdp_inference: bool = False
    hsdp_replicate_dim: int = 1
    hsdp_shard_dim: int = 1
    backend: str | None = None
    model_id: str | None = None
    model_paths: dict[str, str] = field(default_factory=dict)
    distributed_backend: str | None = None
    distributed_init_timeout_seconds: int = 1800
    writer_rank: int = 0
    progress_rank: int = 0
    distributed_context: Any | None = None
    parallel_mode: str = "disabled"
    sp_degree: int = 0
    cfg_parallel_degree: int = 0
    vae_parallel_degree: int = 0
    parallel_context: Any | None = None
    distributed_compute_mode: str = "auto"
    vae_max_parallelism: int = 0
    vae_max_inflight_tiles: int = 1
    official_parallel_context: Any | None = None
    attention_backend: str = "sdpa"
    attention_backend_report: dict[str, Any] | None = field(default=None, init=False, repr=False)
    transformer_quantization: str = "none"
    text_encoder_quantization: str = "none"
    fp8_linear_backend: str = "auto"
    fp8_linear_granularity: str = "per_row"
    fp8_fast_accum: bool = False
    effective_transformer_quantization: str = field(default="none", init=False)
    transformer_quantization_report: dict[str, Any] | None = field(default=None, init=False, repr=False)
    effective_text_encoder_quantization: str = field(default="none", init=False)
    text_encoder_quantization_report: dict[str, Any] | None = field(default=None, init=False, repr=False)
    enable_torch_compile: bool = False
    warmup: bool = False
    warmup_steps: int = 1
    operator_fusion_backend: str = "disabled"
    operator_fusion_ops: str | tuple[str, ...] | None = None
    operator_fusion_decision: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.transformer_quantization = str(
            self.transformer_quantization
        ).strip().lower()
        if self.transformer_quantization not in {
            "none", "fp8_w8a8", "fp8_w8a8_triton_selective",
            "int8_w8a8_viditq",
        }:
            raise ValueError(
                "transformer_quantization must be one of: none, fp8_w8a8, "
                "fp8_w8a8_triton_selective, int8_w8a8_viditq"
            )
        self.text_encoder_quantization = str(
            self.text_encoder_quantization
        ).strip().lower()
        if self.text_encoder_quantization not in {"none", "int8_w8a8_viditq"}:
            raise ValueError(
                "text_encoder_quantization must be one of: none, int8_w8a8_viditq"
            )
        self.fp8_linear_backend = str(self.fp8_linear_backend).strip().lower()
        if self.fp8_linear_backend not in {"auto", "native_scaled_mm"}:
            raise ValueError(
                "fp8_linear_backend must be one of: auto, native_scaled_mm"
            )
        self.fp8_linear_granularity = str(
            self.fp8_linear_granularity
        ).strip().lower()
        if self.fp8_linear_granularity != "per_row":
            raise ValueError("fp8_linear_granularity must be per_row")
        if not isinstance(self.fp8_fast_accum, bool):
            raise ValueError("fp8_fast_accum must be boolean")
        if (
            self.transformer_quantization in {
                "fp8_w8a8", "fp8_w8a8_triton_selective",
                "int8_w8a8_viditq",
            }
            and self.weight_dtype is not None
            and _precision_to_torch_dtype(self.weight_dtype) is not torch.bfloat16
        ):
            raise ValueError(
                f"{self.transformer_quantization} requires the Transformer load "
                "dtype to be bf16"
            )
        if self.transformer_quantization == "fp8_w8a8_triton_selective":
            if self.fp8_linear_backend != "native_scaled_mm":
                raise ValueError(
                    "fp8_w8a8_triton_selective requires native_scaled_mm"
                )
            if not self.fp8_fast_accum:
                raise ValueError(
                    "fp8_w8a8_triton_selective requires fast accumulation"
                )
            if self.enable_torch_compile:
                raise ValueError(
                    "fp8_w8a8_triton_selective does not support torch.compile"
                )
        if self.transformer_quantization == "int8_w8a8_viditq":
            if self.enable_torch_compile:
                raise ValueError("int8_w8a8_viditq does not support torch.compile")
            if str(self.operator_fusion_backend).strip().lower() != "disabled":
                raise ValueError(
                    "int8_w8a8_viditq requires operator_fusion_backend=disabled"
                )
        if self.text_encoder_quantization == "int8_w8a8_viditq":
            text_dtype = self.resolve_component_dtype("text_encoder")
            if text_dtype is not None and text_dtype is not torch.bfloat16:
                raise ValueError("int8_w8a8_viditq T5 requires text encoder load dtype bf16")
            if self.enable_torch_compile:
                raise ValueError("int8_w8a8_viditq T5 does not support torch.compile")
        if type(self.warmup_steps) is not int or self.warmup_steps < 1:
            raise ValueError(
                "warmup_steps must be a positive non-bool int, got "
                f"{self.warmup_steps!r}"
            )
        if self.vae_max_inflight_tiles not in (1, 2):
            raise ValueError(
                "vae_max_inflight_tiles must be 1 or 2, got "
                f"{self.vae_max_inflight_tiles}"
            )
        self.effective_transformer_quantization = "none"
        self.effective_text_encoder_quantization = "none"

    def resolve_component_dtype(self, component_name: str) -> torch.dtype | None:
        """Resolve the preferred dtype for a component."""
        if component_name in self.component_dtypes:
            return _precision_to_torch_dtype(self.component_dtypes[component_name])

        if self.weight_dtype is not None:
            return _precision_to_torch_dtype(self.weight_dtype)

        pipeline_config = self.pipeline_config
        precision_map = {
            "transformer": "dit_precision",
            "vae": "vae_precision",
            "text_encoder": "text_encoder_precision",
        }
        precision_name = precision_map.get(component_name)
        if precision_name and hasattr(pipeline_config, precision_name):
            precision = getattr(pipeline_config, precision_name)
            if precision is not None:
                return _precision_to_torch_dtype(str(precision))
        return None

    def resolve_resource_policy(self) -> RuntimeResourcePolicy:
        return resolve_runtime_resource_policy(self)

    @classmethod
    def from_kwargs(cls, **kwargs) -> "ServerArgs":
        return cls(**kwargs)


def _precision_to_torch_dtype(
    precision: str | torch.dtype | None,
) -> torch.dtype | None:
    if precision is None:
        return None
    if isinstance(precision, torch.dtype):
        return precision

    normalized = str(precision).strip().lower()
    normalized = normalized.removeprefix("torch.")
    normalized = normalized.replace("-", "_")
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "single": torch.float32,
    }
    return mapping.get(normalized)


_GLOBAL_SERVER_ARGS = ServerArgs()


def set_global_server_args(server_args: ServerArgs) -> None:
    global _GLOBAL_SERVER_ARGS
    _GLOBAL_SERVER_ARGS = server_args


def get_global_server_args() -> ServerArgs:
    return _GLOBAL_SERVER_ARGS
