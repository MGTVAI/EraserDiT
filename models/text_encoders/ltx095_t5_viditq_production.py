"""Signed Phase 2b.3 production policy for LTX0.9.5 T5 ViDiT-Q."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from layers.quantization import (
    VIDITQ_DEFAULT_ROOT,
    ViDiTQKernelSet,
    ViDiTQW8A8BF16Linear,
    evaluate_viditq_shape,
    resolve_viditq_kernels,
    validate_viditq_ordinary_storage,
)
from models.text_encoders.ltx095_t5_viditq_quantization import (
    LTX095_T5_LINEAR_COUNT,
    LTX095_T5_LINEAR_FQNS,
    LTX095T5ViDiTQCoverageError,
    _fingerprint,
    _has_active_lora_adapter,
    _replace_submodule,
    _validate_model_identity,
    discover_ltx095_t5_linear_modules,
    summarize_ltx095_t5_viditq_runtime,
)

POLICY_ID = "b048_01_balanced"
POLICY_FQNS = frozenset(
    {
        "encoder.block.10.layer.0.SelfAttention.o",
        "encoder.block.10.layer.1.DenseReluDense.wi_0",
        "encoder.block.10.layer.1.DenseReluDense.wi_1",
        "encoder.block.11.layer.0.SelfAttention.o",
        "encoder.block.11.layer.1.DenseReluDense.wi_0",
        "encoder.block.11.layer.1.DenseReluDense.wi_1",
        "encoder.block.12.layer.0.SelfAttention.o",
        "encoder.block.12.layer.1.DenseReluDense.wi_0",
        "encoder.block.12.layer.1.DenseReluDense.wi_1",
        "encoder.block.13.layer.0.SelfAttention.o",
        "encoder.block.13.layer.1.DenseReluDense.wi_1",
        "encoder.block.14.layer.0.SelfAttention.o",
        "encoder.block.14.layer.1.DenseReluDense.wi_0",
        "encoder.block.14.layer.1.DenseReluDense.wi_1",
        "encoder.block.15.layer.0.SelfAttention.o",
        "encoder.block.16.layer.0.SelfAttention.o",
        "encoder.block.17.layer.0.SelfAttention.o",
        "encoder.block.18.layer.0.SelfAttention.o",
        "encoder.block.18.layer.1.DenseReluDense.wi_1",
        "encoder.block.19.layer.0.SelfAttention.o",
        "encoder.block.19.layer.1.DenseReluDense.wi_0",
        "encoder.block.19.layer.1.DenseReluDense.wi_1",
        "encoder.block.20.layer.0.SelfAttention.o",
        "encoder.block.20.layer.1.DenseReluDense.wi_0",
        "encoder.block.20.layer.1.DenseReluDense.wi_1",
        "encoder.block.21.layer.0.SelfAttention.o",
        "encoder.block.21.layer.1.DenseReluDense.wi_0",
        "encoder.block.21.layer.1.DenseReluDense.wi_1",
        "encoder.block.22.layer.0.SelfAttention.o",
        "encoder.block.22.layer.1.DenseReluDense.wi_0",
        "encoder.block.22.layer.1.DenseReluDense.wi_1",
        "encoder.block.23.layer.0.SelfAttention.o",
        "encoder.block.23.layer.1.DenseReluDense.wi_0",
        "encoder.block.23.layer.1.DenseReluDense.wi_1",
        "encoder.block.3.layer.0.SelfAttention.o",
        "encoder.block.4.layer.0.SelfAttention.o",
        "encoder.block.4.layer.1.DenseReluDense.wi_1",
        "encoder.block.5.layer.0.SelfAttention.o",
        "encoder.block.5.layer.1.DenseReluDense.wi_1",
        "encoder.block.6.layer.0.SelfAttention.o",
        "encoder.block.6.layer.1.DenseReluDense.wi_1",
        "encoder.block.7.layer.0.SelfAttention.o",
        "encoder.block.7.layer.1.DenseReluDense.wi_0",
        "encoder.block.7.layer.1.DenseReluDense.wi_1",
        "encoder.block.8.layer.0.SelfAttention.o",
        "encoder.block.8.layer.1.DenseReluDense.wi_1",
        "encoder.block.9.layer.0.SelfAttention.o",
        "encoder.block.9.layer.1.DenseReluDense.wi_1",
    }
)
POLICY_FQN_SHA256 = hashlib.sha256(
    "\n".join(sorted(POLICY_FQNS)).encode("utf-8")
).hexdigest()
SIGNED_POLICY_FQN_SHA256 = (
    "8148e4ddbc82d1c18bbb9fc9e3d894b8077122ed8b394f9cdaea22341d70a6b6"
)
if len(POLICY_FQNS) != 48 or POLICY_FQN_SHA256 != SIGNED_POLICY_FQN_SHA256:
    raise RuntimeError("signed LTX095 T5 production policy diverged")


@dataclass(frozen=True)
class LTX095T5ProductionQuantizationReport:
    requested: str
    effective: str
    backend: str
    policy_id: str
    policy_fqn_sha256: str
    expected_count: int
    discovered_count: int
    selected_count: int
    quantized_count: int
    skipped_count: int
    remaining_bf16_linear_count: int
    int8_weight_bytes: int
    bf16_scale_bytes: int
    int16_zero_point_bytes: int
    bf16_bias_bytes: int
    remaining_bf16_weight_bytes: int
    bf16_weight_bytes_removed: int
    bf16_duplicate_weight_count: int
    dependency_fingerprints: tuple[dict[str, Any], ...]
    compute_capability: tuple[int, int]
    runtime_call_count: int
    fallback_count: int

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["compute_capability"] = list(self.compute_capability)
        return result


def validate_ltx095_t5_viditq_production(
    text_encoder: nn.Module,
    report: LTX095T5ProductionQuantizationReport,
) -> None:
    quantized = {
        name: module
        for name, module in text_encoder.named_modules()
        if name and isinstance(module, ViDiTQW8A8BF16Linear)
    }
    if set(quantized) != POLICY_FQNS:
        raise LTX095T5ViDiTQCoverageError("production T5 ViDiT-Q coverage mismatch")
    remaining = dict(discover_ltx095_t5_linear_modules(text_encoder))
    if set(remaining) != LTX095_T5_LINEAR_FQNS - POLICY_FQNS:
        raise LTX095T5ViDiTQCoverageError("production remaining BF16 coverage mismatch")
    if any(module.weight.dtype is not torch.bfloat16 for module in remaining.values()):
        raise LTX095T5ViDiTQCoverageError("production remaining Linear is not BF16")
    for module in quantized.values():
        validate_viditq_ordinary_storage(module)
    if report.bf16_duplicate_weight_count:
        raise LTX095T5ViDiTQCoverageError("T5 BF16 duplicate weights detected")


def quantize_ltx095_t5_viditq_production(
    text_encoder: nn.Module,
    *,
    execution_device: torch.device | str,
    viditq_root: Path | str = VIDITQ_DEFAULT_ROOT,
    kernels: ViDiTQKernelSet | None = None,
    collect_runtime_timing: bool = False,
) -> LTX095T5ProductionQuantizationReport:
    _validate_model_identity(text_encoder)
    existing = getattr(text_encoder, "_mgerase_ltx095_t5_production_report", None)
    if existing is not None:
        if not isinstance(existing, LTX095T5ProductionQuantizationReport):
            raise TypeError("invalid existing production T5 ViDiT-Q report")
        validate_ltx095_t5_viditq_production(text_encoder, existing)
        return existing
    if getattr(text_encoder, "_mgerase_ltx095_t5_viditq_report", None) is not None:
        raise ValueError("LTX095 T5 is already quantized with another policy")
    if _has_active_lora_adapter(text_encoder):
        raise ValueError("LTX095 T5 ViDiT-Q does not support active LoRA adapters")

    discovered = discover_ltx095_t5_linear_modules(text_encoder)
    discovered_map = dict(discovered)
    if set(discovered_map) != LTX095_T5_LINEAR_FQNS or len(discovered) != LTX095_T5_LINEAR_COUNT:
        raise LTX095T5ViDiTQCoverageError("T5 Linear census diverged before production conversion")
    for name in POLICY_FQNS:
        module = discovered_map[name]
        support = evaluate_viditq_shape(
            in_features=module.in_features, out_features=module.out_features
        )
        if not support.supported:
            raise LTX095T5ViDiTQCoverageError(
                f"production T5 module unsupported: {name}: {support.reason}"
            )

    source_device_types = {module.weight.device.type for module in discovered_map.values()}
    target_device = torch.device(execution_device)
    if "meta" in source_device_types:
        if source_device_types != {"meta"}:
            raise RuntimeError("mixed meta/materialized T5 weights are unsupported")
    elif any(device_type != "cuda" for device_type in source_device_types):
        if target_device.type != "cuda":
            raise RuntimeError("CPU-loaded T5 ViDiT-Q conversion requires CUDA")
        text_encoder.to(target_device)

    resolved = kernels or resolve_viditq_kernels(
        device=target_device, viditq_root=viditq_root
    )
    replacements = {
        name: ViDiTQW8A8BF16Linear.from_linear(
            discovered_map[name],
            kernels=resolved,
            viditq_root=viditq_root,
            collect_runtime_timing=collect_runtime_timing,
        )
        for name in sorted(POLICY_FQNS)
    }
    for name, replacement in replacements.items():
        _replace_submodule(text_encoder, name, replacement)

    storage = [module.storage_summary() for module in replacements.values()]
    remaining = [
        discovered_map[name]
        for name in sorted(LTX095_T5_LINEAR_FQNS - POLICY_FQNS)
    ]
    removed = sum(discovered_map[name].weight.numel() * 2 for name in POLICY_FQNS)
    report = LTX095T5ProductionQuantizationReport(
        requested="int8_w8a8_viditq",
        effective="int8_w8a8_viditq",
        backend="viditq_extension",
        policy_id=POLICY_ID,
        policy_fqn_sha256=POLICY_FQN_SHA256,
        expected_count=LTX095_T5_LINEAR_COUNT,
        discovered_count=len(discovered),
        selected_count=len(replacements),
        quantized_count=len(replacements),
        skipped_count=len(remaining),
        remaining_bf16_linear_count=len(remaining),
        int8_weight_bytes=sum(item["weight_bytes"] for item in storage),
        bf16_scale_bytes=sum(item["scale_bytes"] for item in storage),
        int16_zero_point_bytes=sum(item["zero_point_bytes"] for item in storage),
        bf16_bias_bytes=sum(item["bias_bytes"] for item in storage),
        remaining_bf16_weight_bytes=sum(
            module.weight.numel() * module.weight.element_size() for module in remaining
        ),
        bf16_weight_bytes_removed=removed,
        bf16_duplicate_weight_count=sum(
            item["bf16_duplicate_weight_count"] for item in storage
        ),
        dependency_fingerprints=(
            _fingerprint(resolved.fused_module_path),
            _fingerprint(resolved.qgemm_module_path),
        ),
        compute_capability=resolved.compute_capability,
        runtime_call_count=0,
        fallback_count=resolved.fallback_count,
    )
    validate_ltx095_t5_viditq_production(text_encoder, report)
    setattr(text_encoder, "_mgerase_ltx095_t5_production_report", report)
    return report


__all__ = [
    "LTX095T5ProductionQuantizationReport",
    "POLICY_FQNS",
    "POLICY_FQN_SHA256",
    "POLICY_ID",
    "quantize_ltx095_t5_viditq_production",
    "validate_ltx095_t5_viditq_production",
]
