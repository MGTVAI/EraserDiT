"""Runtime resource policy helpers for the minimal MGErase runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn

from utils.platform import get_local_torch_device


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
        "dynamic_offload",
    }:
        raise ValueError(f"Unsupported resource_policy: {value}")
    return normalized


def resolve_runtime_resource_policy(server_args) -> RuntimeResourcePolicy:
    requested_policy = normalize_resource_policy_name(
        getattr(server_args, "resource_policy", "fullgpu")
    )
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
        getattr(server_args, "max_weight_usage", 5 * 1024**3)
    )
    if max_weight_usage <= 0:
        raise ValueError("max_weight_usage must be positive")

    text_offload = bool(
        dynamic_offload or getattr(server_args, "text_encoder_cpu_offload", False)
    )
    vae_offload = bool(dynamic_offload or getattr(server_args, "vae_cpu_offload", False))
    dit_offload = bool(dynamic_offload or getattr(server_args, "dit_cpu_offload", False))

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


def module_device(module: nn.Module) -> torch.device:
    execution_device = getattr(module, "_mgerase_execution_device", None)
    if execution_device is not None:
        return torch.device(execution_device)
    first_param = next(module.parameters(), None)
    if first_param is None:
        return get_local_torch_device("cpu")
    return first_param.device


def module_dtype(module: nn.Module) -> torch.dtype | None:
    first_param = next(module.parameters(), None)
    if first_param is None:
        return None
    return first_param.dtype


def pin_module_cpu_memory(module: nn.Module) -> bool:
    changed = False
    for parameter in module.parameters():
        if parameter.device.type == "cpu" and not parameter.is_pinned():
            parameter.data = parameter.data.pin_memory()
            changed = True
    for buffer in module.buffers():
        if buffer.device.type == "cpu" and not buffer.is_pinned():
            buffer.data = buffer.data.pin_memory()
            changed = True
    return changed


def maybe_pin_tensor(tensor: torch.Tensor, enable: bool) -> torch.Tensor:
    if not enable or tensor.device.type != "cpu" or tensor.is_pinned():
        return tensor
    return tensor.pin_memory()


def move_module_to_device(
    module: nn.Module,
    target_device: torch.device,
    *,
    dtype: torch.dtype | None = None,
) -> bool:
    current_device = module_device(module)
    if current_device == target_device and (dtype is None or module_dtype(module) == dtype):
        return False
    move_kwargs: dict[str, Any] = {"device": target_device}
    if dtype is not None:
        move_kwargs["dtype"] = dtype
    if current_device.type == "cpu" or target_device.type == "cpu":
        move_kwargs["non_blocking"] = True
    module.to(**move_kwargs)
    return True
