"""Ordinary-storage FP8 E4M3FN W8A8 Linear implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import (
    FP8ActivationQuantization,
    FP8LinearBackend,
    FP8LinearConfig,
)
from .registry import FP8BackendSelection
from .runtime import FP8LinearRuntimeStats

FP8_ALIGNMENT = 16
FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_WEIGHT_INPUT_DTYPES = {
    torch.bfloat16,
    torch.float16,
    torch.float32,
}


@dataclass(frozen=True)
class FP8WeightQuantization:
    weight_fp8: torch.Tensor
    weight_scale: torch.Tensor
    logical_in_features: int
    padded_in_features: int

    @property
    def padding_required(self) -> bool:
        return self.logical_in_features != self.padded_in_features


def align_fp8_dimension(value: int, alignment: int = FP8_ALIGNMENT) -> int:
    if value <= 0:
        raise ValueError("dimension must be positive")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def _validate_quantization_input(
    tensor: torch.Tensor,
    *,
    name: str,
    validate_finite: bool,
) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if tensor.dtype not in _WEIGHT_INPUT_DTYPES:
        allowed = ", ".join(str(dtype) for dtype in _WEIGHT_INPUT_DTYPES)
        raise TypeError(f"{name} dtype must be one of: {allowed}")
    if validate_finite and not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} contains NaN or Inf")


def quantize_fp8_per_row(
    tensor: torch.Tensor,
    *,
    validate_finite: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_quantization_input(
        tensor,
        name="tensor",
        validate_finite=validate_finite,
    )
    source = tensor.float()
    amax = source.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(
        amax > 0,
        amax / FP8_MAX,
        torch.ones_like(amax),
    ).to(torch.float32)
    quantized = torch.clamp(source / scale, -FP8_MAX, FP8_MAX).to(
        torch.float8_e4m3fn
    )
    return quantized, scale


def quantize_fp8_activation(
    tensor: torch.Tensor,
    backend: FP8ActivationQuantization,
) -> tuple[torch.Tensor, torch.Tensor]:
    if backend is FP8ActivationQuantization.EAGER:
        return quantize_fp8_per_row(tensor, validate_finite=False)
    if backend is FP8ActivationQuantization.TRITON_PER_ROW:
        from .fp8_activation import quantize_fp8_per_row_triton

        return quantize_fp8_per_row_triton(tensor)
    raise ValueError(f"unsupported FP8 activation quantization backend: {backend}")


def quantize_fp8_weight_per_row(
    weight: torch.Tensor,
    *,
    padded_in_features: int | None = None,
) -> FP8WeightQuantization:
    _validate_quantization_input(
        weight,
        name="weight",
        validate_finite=True,
    )
    out_features, in_features = weight.shape
    if out_features % FP8_ALIGNMENT != 0:
        raise ValueError(
            "native_scaled_mm requires out_features to be divisible by 16"
        )
    effective_in_features = (
        align_fp8_dimension(in_features)
        if padded_in_features is None
        else int(padded_in_features)
    )
    if effective_in_features < in_features:
        raise ValueError("padded_in_features cannot shrink the weight")
    if effective_in_features % FP8_ALIGNMENT != 0:
        raise ValueError("padded_in_features must be divisible by 16")
    padded_weight = (
        F.pad(weight, (0, effective_in_features - in_features))
        if effective_in_features != in_features
        else weight
    )
    weight_fp8, weight_scale = quantize_fp8_per_row(
        padded_weight,
        validate_finite=False,
    )
    return FP8WeightQuantization(
        weight_fp8=weight_fp8,
        weight_scale=weight_scale,
        logical_in_features=in_features,
        padded_in_features=effective_in_features,
    )


def native_scaled_mm(
    activation_fp8: torch.Tensor,
    weight_fp8: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    use_fast_accum: bool,
) -> torch.Tensor:
    if activation_fp8.device.type != "cuda":
        raise RuntimeError("native_scaled_mm execution requires CUDA tensors")
    return torch._scaled_mm(
        activation_fp8,
        weight_fp8.t(),
        scale_a=activation_scale,
        scale_b=weight_scale.t(),
        out_dtype=output_dtype,
        use_fast_accum=use_fast_accum,
    )


def _is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    return bool(compiler is not None and compiler.is_compiling())


class FP8W8A8Linear(nn.Module):
    """Inference-only Linear with true FP8 activation and weight operands."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        weight_fp8: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        config: FP8LinearConfig,
        selection: FP8BackendSelection,
        padded_in_features: int,
    ) -> None:
        super().__init__()
        if selection.selected is not FP8LinearBackend.NATIVE_SCALED_MM:
            raise ValueError(
                "FP8W8A8Linear requires a frozen native_scaled_mm selection"
            )
        if config.backend is FP8LinearBackend.DISABLED:
            raise ValueError("disabled config cannot construct FP8W8A8Linear")
        if weight_fp8.dtype != torch.float8_e4m3fn:
            raise TypeError("weight_fp8 must use torch.float8_e4m3fn")
        if weight_scale.dtype != torch.float32:
            raise TypeError("weight_scale must use torch.float32")
        if tuple(weight_fp8.shape) != (out_features, padded_in_features):
            raise ValueError("weight_fp8 shape does not match module metadata")
        if tuple(weight_scale.shape) != (out_features, 1):
            raise ValueError("weight_scale must have shape [out_features, 1]")
        if out_features % FP8_ALIGNMENT != 0:
            raise ValueError("out_features must be divisible by 16")
        if padded_in_features % FP8_ALIGNMENT != 0:
            raise ValueError("padded_in_features must be divisible by 16")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.padded_in_features = int(padded_in_features)
        self.config = config
        self.selection = selection
        self.runtime_stats = FP8LinearRuntimeStats()
        self.register_buffer(
            "weight_fp8",
            weight_fp8.detach().contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "weight_scale",
            weight_scale.detach().contiguous(),
            persistent=True,
        )
        if bias is None:
            self.register_parameter("bias", None)
        else:
            bias_value = bias.detach().to(dtype=config.output_dtype)
            self.bias = nn.Parameter(
                bias_value.contiguous(),
                requires_grad=False,
            )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        config: FP8LinearConfig,
        selection: FP8BackendSelection,
    ) -> "FP8W8A8Linear":
        if not isinstance(linear, nn.Linear):
            raise TypeError("from_linear expects torch.nn.Linear")
        quantized = quantize_fp8_weight_per_row(linear.weight.detach())
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            weight_fp8=quantized.weight_fp8,
            weight_scale=quantized.weight_scale,
            bias=linear.bias,
            config=config,
            selection=selection,
            padded_in_features=quantized.padded_in_features,
        )

    @property
    def padding_required(self) -> bool:
        return self.padded_in_features != self.in_features

    @property
    def weight_nbytes(self) -> int:
        return self.weight_fp8.numel() * self.weight_fp8.element_size()

    @property
    def scale_nbytes(self) -> int:
        return self.weight_scale.numel() * self.weight_scale.element_size()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.ndim < 2:
            raise ValueError("FP8 Linear input must have at least two dimensions")
        if input_tensor.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input last dimension {self.in_features}, "
                f"got {input_tensor.shape[-1]}"
            )
        if not input_tensor.dtype.is_floating_point:
            raise TypeError("FP8 Linear input must be floating point")
        if input_tensor.device != self.weight_fp8.device:
            raise RuntimeError(
                "input and FP8 weight must be on the same device"
            )
        leading_shape = tuple(input_tensor.shape[:-1])
        flattened = input_tensor.reshape(-1, self.in_features)
        if self.padding_required:
            flattened = F.pad(
                flattened,
                (0, self.padded_in_features - self.in_features),
            )
        try:
            activation_fp8, activation_scale = quantize_fp8_activation(
                flattened,
                self.config.activation_quantization,
            )
            output = native_scaled_mm(
                activation_fp8,
                self.weight_fp8,
                activation_scale,
                self.weight_scale,
                output_dtype=self.config.output_dtype,
                use_fast_accum=self.config.use_fast_accum,
            )
            if self.bias is not None:
                output = output + self.bias
        except Exception:
            if not _is_compiling():
                self.runtime_stats.record_failure()
            raise
        if not _is_compiling():
            self.runtime_stats.record_call(
                rows=flattened.shape[0],
                padded=self.padding_required,
            )
        return output.reshape(*leading_shape, self.out_features)

    def metadata(self) -> dict[str, Any]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "padded_in_features": self.padded_in_features,
            "padding_required": self.padding_required,
            "weight_dtype": str(self.weight_fp8.dtype).removeprefix("torch."),
            "weight_scale_dtype": str(
                self.weight_scale.dtype
            ).removeprefix("torch."),
            "bias_dtype": (
                None
                if self.bias is None
                else str(self.bias.dtype).removeprefix("torch.")
            ),
            "weight_nbytes": self.weight_nbytes,
            "scale_nbytes": self.scale_nbytes,
            "config": self.config.as_dict(),
            "selection": self.selection.as_dict(),
            "runtime": self.runtime_stats.as_dict(),
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"padded_in_features={self.padded_in_features}, "
            f"bias={self.bias is not None}, "
            f"fast_accum={self.config.use_fast_accum}"
        )
