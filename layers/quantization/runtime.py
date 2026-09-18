"""Runtime counters and ordinary-storage diagnostics for FP8 Linear modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from .fp8_linear import FP8W8A8Linear


@dataclass
class FP8LinearRuntimeStats:
    call_count: int = 0
    padded_call_count: int = 0
    input_row_count: int = 0
    failure_count: int = 0

    def record_call(self, *, rows: int, padded: bool) -> None:
        self.call_count += 1
        self.input_row_count += int(rows)
        self.padded_call_count += int(padded)

    def record_failure(self) -> None:
        self.failure_count += 1

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class FP8LinearStorageSummary:
    module_count: int
    padded_module_count: int
    weight_bytes: int
    scale_bytes: int
    bias_bytes: int
    bf16_duplicate_weight_count: int
    call_count: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def validate_ordinary_tensor_storage(module: "FP8W8A8Linear") -> None:
    if type(module.weight_fp8) is not torch.Tensor:
        raise TypeError("weight_fp8 must use an ordinary torch.Tensor")
    if type(module.weight_scale) is not torch.Tensor:
        raise TypeError("weight_scale must use an ordinary torch.Tensor")
    if module.weight_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError("weight_fp8 must use E4M3FN")
    if module.weight_scale.dtype != torch.float32:
        raise TypeError("weight_scale must use FP32")
    if hasattr(module, "weight"):
        raise ValueError("FP8 module must not retain a BF16 weight attribute")
    if module.bias is not None and not isinstance(module.bias, nn.Parameter):
        raise TypeError("bias must be an ordinary nn.Parameter")


def summarize_fp8_linear_storage(root: nn.Module) -> FP8LinearStorageSummary:
    from .fp8_linear import FP8W8A8Linear

    modules = [
        module for module in root.modules() if isinstance(module, FP8W8A8Linear)
    ]
    for module in modules:
        validate_ordinary_tensor_storage(module)
    return FP8LinearStorageSummary(
        module_count=len(modules),
        padded_module_count=sum(module.padding_required for module in modules),
        weight_bytes=sum(
            module.weight_fp8.numel() * module.weight_fp8.element_size()
            for module in modules
        ),
        scale_bytes=sum(
            module.weight_scale.numel() * module.weight_scale.element_size()
            for module in modules
        ),
        bias_bytes=sum(
            0
            if module.bias is None
            else module.bias.numel() * module.bias.element_size()
            for module in modules
        ),
        bf16_duplicate_weight_count=sum(
            int(hasattr(module, "weight")) for module in modules
        ),
        call_count=sum(module.runtime_stats.call_count for module in modules),
    )
