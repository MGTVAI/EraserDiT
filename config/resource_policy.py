"""Resource policy configuration and capability resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class RuntimeResourcePolicy:
    requested_policy: str
    selected_policy: str
    requested_dynamic_offload: bool
    requested_pin_memory: bool
    pin_memory: bool
    dynamic_offload: bool
    max_weight_usage: int
    text_encoder_cpu_offload: bool
    vae_cpu_offload: bool
    dit_cpu_offload: bool
    fallback_reasons: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_resource_policy_name(value: str | None) -> str:
    normalized = str(value or "fullgpu").strip().lower()
    if normalized not in {
        "fullgpu",
        "fullgpu_pin_memory",
        "component_offload",
        "dynamic_offload",
    }:
        raise ValueError(f"Unsupported resource_policy: {value}")
    return normalized


def resolve_runtime_resource_policy(server_args) -> RuntimeResourcePolicy:
    requested_policy = normalize_resource_policy_name(
        getattr(server_args, "resource_policy", "fullgpu")
    )
    pipeline_name = getattr(server_args, "pipeline_class_name", None)
    if requested_policy == "component_offload" and pipeline_name not in (
        None, "EraserDiTErasePipeline"
    ):
        raise ValueError("component_offload is currently implemented only for EraserDiT")
    fallback_reasons: list[str] = []

    requested_pin_memory = bool(getattr(server_args, "pin_memory", False))
    requested_dynamic_offload = bool(
        getattr(server_args, "dynamic_offload", False)
    )

    if requested_policy == "fullgpu":
        pass
    elif requested_policy == "fullgpu_pin_memory":
        requested_pin_memory = True
    elif requested_policy == "dynamic_offload":
        requested_dynamic_offload = True

    pin_memory = requested_pin_memory
    dynamic_offload = requested_dynamic_offload
    max_weight_usage = int(
        getattr(server_args, "max_weight_usage", 2 * 1024**3)
    )
    if max_weight_usage <= 0:
        raise ValueError("max_weight_usage must be positive")

    component_offload = requested_policy == "component_offload"
    text_offload = bool(
        component_offload or dynamic_offload or getattr(server_args, "text_encoder_cpu_offload", False)
    )
    vae_offload = bool(component_offload or dynamic_offload or getattr(server_args, "vae_cpu_offload", False))
    dit_offload = bool(component_offload or dynamic_offload or getattr(server_args, "dit_cpu_offload", False))

    if not torch.cuda.is_available():
        if pin_memory:
            fallback_reasons.append("pin_memory_disabled_without_cuda")
            pin_memory = False
        if dynamic_offload:
            fallback_reasons.append("dynamic_offload_disabled_without_cuda")
            dynamic_offload = False
            text_offload = False
            vae_offload = False
            dit_offload = False

    return RuntimeResourcePolicy(
        requested_policy=requested_policy,
        selected_policy=requested_policy,
        requested_dynamic_offload=requested_dynamic_offload,
        requested_pin_memory=requested_pin_memory,
        pin_memory=pin_memory,
        dynamic_offload=dynamic_offload,
        max_weight_usage=max_weight_usage,
        text_encoder_cpu_offload=text_offload,
        vae_cpu_offload=vae_offload,
        dit_cpu_offload=dit_offload,
        fallback_reasons=tuple(fallback_reasons),
    )
