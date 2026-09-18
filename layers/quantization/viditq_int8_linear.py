"""BF16-in / ViDiT-Q INT8 W8A8 / BF16-out inference Linear."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .viditq_registry import (
    VIDITQ_DEFAULT_ROOT,
    VIDITQ_M_ALIGNMENT,
    ViDiTQKernelSet,
    require_viditq_shape,
    resolve_viditq_kernels,
)


@dataclass(frozen=True)
class ViDiTQWeightQuantization:
    weight_int8: torch.Tensor
    weight_scale: torch.Tensor
    weight_zp: torch.Tensor
    constant_row_count: int


@dataclass
class ViDiTQLinearRuntimeStats:
    call_count: int = 0
    logical_row_count: int = 0
    effective_row_count: int = 0
    padded_row_count: int = 0
    padded_call_count: int = 0
    failure_count: int = 0
    fused_quant_call_count: int = 0
    asymmetric_qgemm_call_count: int = 0
    fallback_count: int = 0
    activation_quantization_ms: float = 0.0
    row_padding_ms: float = 0.0
    qgemm_ms: float = 0.0
    total_ms: float = 0.0

    def record_call(
        self,
        *,
        logical_rows: int,
        effective_rows: int,
        timing_ms: dict[str, float] | None = None,
    ) -> None:
        padding = int(effective_rows) - int(logical_rows)
        self.call_count += 1
        self.logical_row_count += int(logical_rows)
        self.effective_row_count += int(effective_rows)
        self.padded_row_count += padding
        self.padded_call_count += int(padding > 0)
        self.fused_quant_call_count += 1
        self.asymmetric_qgemm_call_count += 1
        if timing_ms is not None:
            self.activation_quantization_ms += timing_ms["activation_quantization"]
            self.row_padding_ms += timing_ms["row_padding"]
            self.qgemm_ms += timing_ms["qgemm"]
            self.total_ms += timing_ms["total"]

    def record_failure(self) -> None:
        self.failure_count += 1

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass
class ViDiTQWorkspace:
    """Per-call-owner buffers; never shared across streams or concurrent calls."""

    logical_rows: int
    effective_rows: int
    in_features: int
    scale_input: torch.Tensor
    sum_input: torch.Tensor
    activation_padded: torch.Tensor | None
    scale_padded: torch.Tensor | None
    sum_padded: torch.Tensor | None

    @classmethod
    def allocate(
        cls,
        *,
        logical_rows: int,
        in_features: int,
        device: torch.device | str,
    ) -> "ViDiTQWorkspace":
        effective_rows = align_viditq_rows(logical_rows)
        resolved = torch.device(device)
        scale_input = torch.empty(
            logical_rows,
            dtype=torch.bfloat16,
            device=resolved,
        )
        sum_input = torch.empty_like(scale_input)
        if effective_rows == logical_rows:
            activation_padded = None
            scale_padded = None
            sum_padded = None
        else:
            activation_padded = torch.empty(
                (effective_rows, in_features),
                dtype=torch.int8,
                device=resolved,
            )
            scale_padded = torch.empty(
                effective_rows,
                dtype=torch.bfloat16,
                device=resolved,
            )
            sum_padded = torch.empty_like(scale_padded)
        return cls(
            logical_rows=logical_rows,
            effective_rows=effective_rows,
            in_features=in_features,
            scale_input=scale_input,
            sum_input=sum_input,
            activation_padded=activation_padded,
            scale_padded=scale_padded,
            sum_padded=sum_padded,
        )

    def validate(
        self,
        *,
        logical_rows: int,
        in_features: int,
        device: torch.device,
    ) -> None:
        if self.logical_rows != logical_rows or self.in_features != in_features:
            raise ValueError("ViDiT-Q workspace shape does not match the input")
        tensors = [self.scale_input, self.sum_input]
        tensors.extend(
            tensor
            for tensor in (
                self.activation_padded,
                self.scale_padded,
                self.sum_padded,
            )
            if tensor is not None
        )
        if any(tensor.device != device for tensor in tensors):
            raise RuntimeError("ViDiT-Q workspace must be on the input device")


def align_viditq_rows(
    rows: int,
    alignment: int = VIDITQ_M_ALIGNMENT,
) -> int:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((rows + alignment - 1) // alignment) * alignment


def pad_viditq_rows(
    tensor: torch.Tensor,
    *,
    alignment: int = VIDITQ_M_ALIGNMENT,
) -> tuple[torch.Tensor, int]:
    if tensor.ndim < 1:
        raise ValueError("row tensor must have at least one dimension")
    logical_rows = int(tensor.shape[0])
    effective_rows = align_viditq_rows(logical_rows, alignment)
    if effective_rows == logical_rows:
        return tensor, logical_rows
    padding = torch.zeros(
        (effective_rows - logical_rows, *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat((tensor, padding), dim=0).contiguous(), logical_rows


def quantize_viditq_weight_asymmetric(
    weight: torch.Tensor,
) -> ViDiTQWeightQuantization:
    if weight.ndim != 2:
        raise ValueError("weight must be two-dimensional")
    if weight.dtype != torch.bfloat16:
        raise TypeError("weight must be bfloat16")
    if not bool(torch.isfinite(weight).all().item()):
        raise ValueError("weight contains non-finite values")
    source = weight.detach()
    weight_max = source.max(dim=-1).values
    weight_min = source.min(dim=-1).values
    weight_scale = ((weight_max - weight_min) / 255.0).to(torch.bfloat16)
    constant_rows = weight_scale == 0
    constant_row_count = int(constant_rows.sum().item())
    if constant_row_count:
        raise ValueError(
            f"weight contains {constant_row_count} constant output rows"
        )
    weight_zp = (
        torch.round(weight_min / weight_scale) + 128
    ).to(torch.int16)
    weight_int8 = torch.clamp(
        torch.round(source / weight_scale.unsqueeze(1))
        - weight_zp.to(source.dtype).unsqueeze(1),
        -128,
        127,
    ).to(torch.int8)
    return ViDiTQWeightQuantization(
        weight_int8=weight_int8.contiguous(),
        weight_scale=weight_scale.contiguous(),
        weight_zp=weight_zp.contiguous(),
        constant_row_count=constant_row_count,
    )


def dequantize_viditq_weight(
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
) -> torch.Tensor:
    if weight_int8.ndim != 2:
        raise ValueError("weight_int8 must be two-dimensional")
    if weight_int8.dtype != torch.int8:
        raise TypeError("weight_int8 must be int8")
    if weight_scale.dtype != torch.bfloat16:
        raise TypeError("weight_scale must be bfloat16")
    if weight_zp.dtype != torch.int16:
        raise TypeError("weight_zp must be int16")
    if weight_scale.shape != (weight_int8.shape[0],):
        raise ValueError("weight_scale shape does not match weight_int8")
    if weight_zp.shape != (weight_int8.shape[0],):
        raise ValueError("weight_zp shape does not match weight_int8")
    return (
        (
            weight_int8.float()
            + weight_zp.float().unsqueeze(1)
        )
        * weight_scale.float().unsqueeze(1)
    ).to(torch.bfloat16)


def _is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    return bool(compiler is not None and compiler.is_compiling())


def _execute(
    module: "ViDiTQW8A8BF16Linear",
    flattened: torch.Tensor,
    *,
    workspace: ViDiTQWorkspace | None,
    collect_timing: bool,
) -> tuple[torch.Tensor, dict[str, float] | None]:
    logical_rows = int(flattened.shape[0])
    effective_rows = align_viditq_rows(logical_rows)
    if workspace is None:
        scale_input = torch.empty(
            logical_rows,
            dtype=torch.bfloat16,
            device=flattened.device,
        )
        sum_input = torch.empty_like(scale_input)
    else:
        workspace.validate(
            logical_rows=logical_rows,
            in_features=module.in_features,
            device=flattened.device,
        )
        scale_input = workspace.scale_input
        sum_input = workspace.sum_input

    events = (
        tuple(torch.cuda.Event(enable_timing=True) for _ in range(5))
        if collect_timing
        else None
    )
    if events is not None:
        events[0].record()
    activation_int8 = module._kernels.quant_sum_bf16(
        flattened.contiguous(),
        sum_input,
        scale_input,
    )
    if events is not None:
        events[1].record()

    if effective_rows != logical_rows:
        if workspace is None:
            activation_int8, _ = pad_viditq_rows(activation_int8)
            scale_input, _ = pad_viditq_rows(scale_input)
            sum_input, _ = pad_viditq_rows(sum_input)
        else:
            if (
                workspace.activation_padded is None
                or workspace.scale_padded is None
                or workspace.sum_padded is None
            ):
                raise ValueError("padded input requires padded workspace buffers")
            workspace.activation_padded.zero_()
            workspace.scale_padded.zero_()
            workspace.sum_padded.zero_()
            workspace.activation_padded[:logical_rows].copy_(activation_int8)
            workspace.scale_padded[:logical_rows].copy_(scale_input)
            workspace.sum_padded[:logical_rows].copy_(sum_input)
            activation_int8 = workspace.activation_padded
            scale_input = workspace.scale_padded
            sum_input = workspace.sum_padded
    activation_int8 = activation_int8.contiguous()
    scale_input = scale_input.contiguous()
    sum_input = sum_input.contiguous()
    if events is not None:
        events[2].record()
    output = module._kernels.w8a8_bf16_bias_weight_asym(
        activation_int8,
        module.weight_int8,
        module.bias,
        scale_input,
        module.weight_scale,
        sum_input,
        module.weight_zp,
    )
    if events is not None:
        events[3].record()
    output = output[:logical_rows]
    if events is not None:
        events[4].record()
        torch.cuda.synchronize(flattened.device)
        timing = {
            "activation_quantization": float(events[0].elapsed_time(events[1])),
            "row_padding": float(events[1].elapsed_time(events[2])),
            "qgemm": float(events[2].elapsed_time(events[3])),
            "total": float(events[0].elapsed_time(events[4])),
        }
    else:
        timing = None
    return output, timing


class ViDiTQW8A8BF16Linear(nn.Module):
    """Inference-only main-aligned asymmetric ViDiT-Q Linear."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        weight_int8: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zp: torch.Tensor,
        bias: torch.Tensor | None,
        has_bias: bool,
        kernels: ViDiTQKernelSet,
        collect_runtime_timing: bool = False,
    ) -> None:
        super().__init__()
        require_viditq_shape(
            in_features=in_features,
            out_features=out_features,
        )
        if weight_int8.dtype != torch.int8:
            raise TypeError("weight_int8 must be int8")
        if weight_scale.dtype != torch.bfloat16:
            raise TypeError("weight_scale must be bfloat16")
        if weight_zp.dtype != torch.int16:
            raise TypeError("weight_zp must be int16")
        if tuple(weight_int8.shape) != (out_features, in_features):
            raise ValueError("weight_int8 shape does not match module metadata")
        if tuple(weight_scale.shape) != (out_features,):
            raise ValueError("weight_scale must have shape [out_features]")
        if tuple(weight_zp.shape) != (out_features,):
            raise ValueError("weight_zp must have shape [out_features]")
        devices = {weight_int8.device, weight_scale.device, weight_zp.device}
        if len(devices) != 1:
            raise RuntimeError("ViDiT-Q weight buffers must share one device")
        if has_bias and bias is None:
            raise ValueError("has_bias=True requires a bias tensor")
        physical_bias = (
            torch.zeros(
                out_features,
                dtype=torch.bfloat16,
                device=weight_int8.device,
            )
            if bias is None
            else bias.detach().to(
                device=weight_int8.device,
                dtype=torch.bfloat16,
            )
        )
        if tuple(physical_bias.shape) != (out_features,):
            raise ValueError("bias must have shape [out_features]")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.has_bias = bool(has_bias)
        self.collect_runtime_timing = bool(collect_runtime_timing)
        self.runtime_stats = ViDiTQLinearRuntimeStats()
        self._kernels = kernels
        self.register_buffer("weight_int8", weight_int8.detach().contiguous())
        self.register_buffer("weight_scale", weight_scale.detach().contiguous())
        self.register_buffer("weight_zp", weight_zp.detach().contiguous())
        self.register_buffer("bias", physical_bias.contiguous())

    @classmethod
    def empty(
        cls,
        *,
        in_features: int,
        out_features: int,
        has_bias: bool,
        device: torch.device | str,
        kernels: ViDiTQKernelSet,
        collect_runtime_timing: bool = False,
    ) -> "ViDiTQW8A8BF16Linear":
        resolved = torch.device(device)
        return cls(
            in_features=in_features,
            out_features=out_features,
            weight_int8=torch.empty(
                (out_features, in_features),
                dtype=torch.int8,
                device=resolved,
            ),
            weight_scale=torch.empty(
                out_features,
                dtype=torch.bfloat16,
                device=resolved,
            ),
            weight_zp=torch.empty(
                out_features,
                dtype=torch.int16,
                device=resolved,
            ),
            bias=(
                torch.empty(
                    out_features,
                    dtype=torch.bfloat16,
                    device=resolved,
                )
                if has_bias
                else None
            ),
            has_bias=has_bias,
            kernels=kernels,
            collect_runtime_timing=collect_runtime_timing,
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        kernels: ViDiTQKernelSet | None = None,
        viditq_root: Path | str = VIDITQ_DEFAULT_ROOT,
        collect_runtime_timing: bool = False,
    ) -> "ViDiTQW8A8BF16Linear":
        if not isinstance(linear, nn.Linear):
            raise TypeError("from_linear expects torch.nn.Linear")
        require_viditq_shape(
            in_features=linear.in_features,
            out_features=linear.out_features,
        )
        if linear.weight.dtype != torch.bfloat16:
            raise TypeError("source Linear weight must be bfloat16")
        if linear.weight.device.type != "cuda":
            raise RuntimeError("source Linear weight must be on CUDA")
        if linear.bias is not None and linear.bias.dtype != torch.bfloat16:
            raise TypeError("source Linear bias must be bfloat16")
        resolved_kernels = kernels or resolve_viditq_kernels(
            device=linear.weight.device,
            viditq_root=viditq_root,
        )
        quantized = quantize_viditq_weight_asymmetric(linear.weight.detach())
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            weight_int8=quantized.weight_int8,
            weight_scale=quantized.weight_scale,
            weight_zp=quantized.weight_zp,
            bias=linear.bias,
            has_bias=linear.bias is not None,
            kernels=resolved_kernels,
            collect_runtime_timing=collect_runtime_timing,
        )

    def forward_with_workspace(
        self,
        input_tensor: torch.Tensor,
        *,
        workspace: ViDiTQWorkspace | None,
    ) -> torch.Tensor:
        if input_tensor.ndim < 2:
            raise ValueError("ViDiT-Q Linear input must have at least two dimensions")
        if input_tensor.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input last dimension {self.in_features}, "
                f"got {input_tensor.shape[-1]}"
            )
        if input_tensor.dtype != torch.bfloat16:
            raise TypeError("ViDiT-Q Linear input must be bfloat16")
        if input_tensor.device.type != "cuda":
            raise RuntimeError("ViDiT-Q Linear execution requires CUDA")
        if input_tensor.device != self.weight_int8.device:
            raise RuntimeError("input and ViDiT-Q buffers must share one device")
        leading_shape = tuple(input_tensor.shape[:-1])
        flattened = input_tensor.reshape(-1, self.in_features)
        try:
            output, timing = _execute(
                self,
                flattened,
                workspace=workspace,
                collect_timing=self.collect_runtime_timing,
            )
        except Exception:
            if not _is_compiling():
                self.runtime_stats.record_failure()
            raise
        if not _is_compiling():
            self.runtime_stats.record_call(
                logical_rows=int(flattened.shape[0]),
                effective_rows=align_viditq_rows(int(flattened.shape[0])),
                timing_ms=timing,
            )
        return output.reshape(*leading_shape, self.out_features)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.forward_with_workspace(input_tensor, workspace=None)

    def storage_summary(self) -> dict[str, Any]:
        return {
            "weight_bytes": self.weight_int8.numel() * self.weight_int8.element_size(),
            "scale_bytes": self.weight_scale.numel() * self.weight_scale.element_size(),
            "zero_point_bytes": self.weight_zp.numel() * self.weight_zp.element_size(),
            "bias_bytes": self.bias.numel() * self.bias.element_size(),
            "has_bias": self.has_bias,
            "bf16_duplicate_weight_count": int(hasattr(self, "weight")),
            "ordinary_buffer_count": 4,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "has_bias": self.has_bias,
            "weight_dtype": "int8",
            "weight_scale_dtype": "bfloat16",
            "weight_zp_dtype": "int16",
            "bias_dtype": "bfloat16",
            "storage": self.storage_summary(),
            "kernel": self._kernels.as_dict(),
            "runtime": self.runtime_stats.as_dict(),
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"has_bias={self.has_bias}, weight_scheme=asymmetric"
        )


def validate_viditq_ordinary_storage(
    module: ViDiTQW8A8BF16Linear,
) -> None:
    expected = {
        "weight_int8": torch.int8,
        "weight_scale": torch.bfloat16,
        "weight_zp": torch.int16,
        "bias": torch.bfloat16,
    }
    for name, dtype in expected.items():
        tensor = getattr(module, name)
        if type(tensor) is not torch.Tensor:
            raise TypeError(f"{name} must use an ordinary torch.Tensor")
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must use {dtype}")
    if hasattr(module, "weight"):
        raise ValueError("ViDiT-Q module must not retain a BF16 weight")
    if dict(module.named_parameters()):
        raise ValueError("ViDiT-Q storage must use buffers, not Parameters")
