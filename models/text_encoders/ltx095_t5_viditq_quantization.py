"""Strict main-aligned ViDiT-Q INT8 W8A8 adapter for LTX0.9.5 T5."""

from __future__ import annotations

import hashlib
import json
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

LTX095_T5_BLOCK_COUNT = 24
LTX095_T5_LINEAR_COUNT = 168
LTX095_T5_SELECTED_COUNT = 144
LTX095_T5_SKIPPED_COUNT = 24
LTX095_T5_SELECTED_PARAMETER_RATIO = 0.782608695652174


def _block_fqns(block: int) -> tuple[str, ...]:
    prefix = f"encoder.block.{block}"
    return (
        *(f"{prefix}.layer.0.SelfAttention.{name}" for name in ("q", "k", "v", "o")),
        f"{prefix}.layer.1.DenseReluDense.wi_0",
        f"{prefix}.layer.1.DenseReluDense.wi_1",
        f"{prefix}.layer.1.DenseReluDense.wo",
    )


LTX095_T5_LINEAR_FQNS = frozenset(
    name for block in range(LTX095_T5_BLOCK_COUNT) for name in _block_fqns(block)
)
LTX095_T5_SKIPPED_FQNS = frozenset(
    f"encoder.block.{block}.layer.1.DenseReluDense.wo"
    for block in range(LTX095_T5_BLOCK_COUNT)
)
LTX095_T5_SELECTED_FQNS = LTX095_T5_LINEAR_FQNS - LTX095_T5_SKIPPED_FQNS


def _fqn_hash(names: frozenset[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


LTX095_T5_LINEAR_FQN_SHA256 = _fqn_hash(LTX095_T5_LINEAR_FQNS)
LTX095_T5_SELECTED_FQN_SHA256 = _fqn_hash(LTX095_T5_SELECTED_FQNS)
LTX095_T5_SKIPPED_FQN_SHA256 = _fqn_hash(LTX095_T5_SKIPPED_FQNS)


class LTX095T5ViDiTQCoverageError(RuntimeError):
    """Raised when the frozen T5 168/144/24 contract diverges."""


@dataclass(frozen=True)
class LTX095T5ViDiTQQuantizationReport:
    requested: str
    effective: str
    backend: str
    expected_count: int
    discovered_count: int
    selected_count: int
    quantized_count: int
    skipped_count: int
    missing_count: int
    unexpected_count: int
    remaining_bf16_linear_count: int
    selected_parameter_ratio: float
    expected_fqn_sha256: str
    selected_fqn_sha256: str
    skipped_fqn_sha256: str
    attention_projection_count: int
    ffn_up_projection_count: int
    ffn_down_bf16_count: int
    int8_weight_bytes: int
    bf16_scale_bytes: int
    int16_zero_point_bytes: int
    bf16_bias_bytes: int
    remaining_bf16_weight_bytes: int
    bf16_duplicate_weight_count: int
    dependency_fingerprints: tuple[dict[str, Any], ...]
    compute_capability: tuple[int, int]
    runtime_call_count: int
    fallback_count: int
    skipped_records: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["compute_capability"] = list(self.compute_capability)
        payload["dependency_fingerprints"] = [
            dict(item) for item in self.dependency_fingerprints
        ]
        payload["skipped_records"] = [dict(item) for item in self.skipped_records]
        return payload

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))


def discover_ltx095_t5_linear_modules(text_encoder: nn.Module) -> list[tuple[str, nn.Linear]]:
    return sorted(
        (
            (name, module)
            for name, module in text_encoder.named_modules()
            if name and isinstance(module, nn.Linear)
        ),
        key=lambda item: item[0],
    )


def _replace_submodule(root: nn.Module, fqn: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = fqn.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, replacement)


def _has_active_lora_adapter(module: nn.Module) -> bool:
    if not getattr(module, "peft_config", None):
        return False
    active = getattr(module, "active_adapters", None)
    if callable(active):
        active = active()
    if active is None:
        value = getattr(module, "active_adapter", None)
        active = [value] if value else []
    return bool(active)


def _validate_model_identity(text_encoder: nn.Module) -> None:
    config = getattr(text_encoder, "config", None)
    if config is None or getattr(config, "model_type", None) != "t5":
        raise TypeError("LTX095 T5 ViDiT-Q requires a T5EncoderModel contract")
    if getattr(config, "d_model", None) != 4096:
        raise TypeError("LTX095 T5 contract requires d_model=4096")
    if getattr(config, "d_ff", None) != 10240:
        raise TypeError("LTX095 T5 contract requires d_ff=10240")
    if getattr(config, "num_layers", None) != LTX095_T5_BLOCK_COUNT:
        raise TypeError("LTX095 T5 contract requires num_layers=24")


def _fingerprint(path_value: str) -> dict[str, Any]:
    path = Path(path_value)
    record: dict[str, Any] = {"path": str(path)}
    try:
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        record.update(
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            sha256=digest.hexdigest(),
        )
    except OSError as error:
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def summarize_ltx095_t5_viditq_runtime(text_encoder: nn.Module) -> dict[str, Any]:
    modules = [
        module
        for module in text_encoder.modules()
        if isinstance(module, ViDiTQW8A8BF16Linear)
    ]
    fields = (
        "logical_row_count",
        "effective_row_count",
        "padded_row_count",
        "padded_call_count",
        "failure_count",
        "fused_quant_call_count",
        "asymmetric_qgemm_call_count",
        "fallback_count",
        "activation_quantization_ms",
        "row_padding_ms",
        "qgemm_ms",
        "total_ms",
    )
    payload: dict[str, Any] = {
        "module_count": len(modules),
        "runtime_call_count": sum(module.runtime_stats.call_count for module in modules),
    }
    for field in fields:
        payload[field] = sum(getattr(module.runtime_stats, field) for module in modules)
    return payload


def validate_ltx095_t5_viditq_coverage(
    text_encoder: nn.Module,
    report: LTX095T5ViDiTQQuantizationReport,
) -> None:
    quantized = {
        name: module
        for name, module in text_encoder.named_modules()
        if name and isinstance(module, ViDiTQW8A8BF16Linear)
    }
    missing = sorted(LTX095_T5_SELECTED_FQNS - quantized.keys())
    unexpected = sorted(quantized.keys() - LTX095_T5_SELECTED_FQNS)
    if missing or unexpected:
        raise LTX095T5ViDiTQCoverageError(
            f"T5 ViDiT-Q coverage mismatch: missing={len(missing)} {missing[:4]}, "
            f"unexpected={len(unexpected)} {unexpected[:4]}"
        )
    remaining = dict(discover_ltx095_t5_linear_modules(text_encoder))
    if set(remaining) != LTX095_T5_SKIPPED_FQNS:
        raise LTX095T5ViDiTQCoverageError(
            f"remaining BF16 T5 Linear mismatch: {sorted(remaining)[:8]}"
        )
    for name, module in remaining.items():
        shape = (module.out_features, module.in_features)
        if shape != (4096, 10240) or module.weight.dtype is not torch.bfloat16:
            raise LTX095T5ViDiTQCoverageError(
                f"invalid skipped T5 module {name}: shape={shape}, dtype={module.weight.dtype}"
            )
    for module in quantized.values():
        validate_viditq_ordinary_storage(module)
    runtime = summarize_ltx095_t5_viditq_runtime(text_encoder)
    if runtime["module_count"] != report.quantized_count:
        raise LTX095T5ViDiTQCoverageError("T5 report/module count mismatch")
    if report.bf16_duplicate_weight_count:
        raise LTX095T5ViDiTQCoverageError("T5 BF16 duplicate weights detected")


def quantize_ltx095_t5_viditq(
    text_encoder: nn.Module,
    *,
    execution_device: torch.device | str,
    viditq_root: Path | str = VIDITQ_DEFAULT_ROOT,
    kernels: ViDiTQKernelSet | None = None,
    collect_runtime_timing: bool = False,
) -> LTX095T5ViDiTQQuantizationReport:
    _validate_model_identity(text_encoder)
    existing = getattr(text_encoder, "_mgerase_ltx095_t5_viditq_report", None)
    if existing is not None:
        if not isinstance(existing, LTX095T5ViDiTQQuantizationReport):
            raise TypeError("invalid existing LTX095 T5 ViDiT-Q report")
        validate_ltx095_t5_viditq_coverage(text_encoder, existing)
        return existing
    if getattr(text_encoder, "_mgerase_ltx095_t5_quantization_report", None) is not None:
        raise ValueError("LTX095 T5 is already quantized with another mode")
    if _has_active_lora_adapter(text_encoder):
        raise ValueError("LTX095 T5 ViDiT-Q does not support active LoRA adapters")

    discovered = discover_ltx095_t5_linear_modules(text_encoder)
    discovered_map = dict(discovered)
    names = set(discovered_map)
    missing = sorted(LTX095_T5_LINEAR_FQNS - names)
    unexpected = sorted(names - LTX095_T5_LINEAR_FQNS)
    if missing or unexpected or len(discovered) != LTX095_T5_LINEAR_COUNT:
        raise LTX095T5ViDiTQCoverageError(
            f"T5 Linear coverage mismatch: missing={len(missing)} {missing[:4]}, "
            f"unexpected={len(unexpected)} {unexpected[:4]}, discovered={len(discovered)}"
        )
    for name in LTX095_T5_SELECTED_FQNS:
        module = discovered_map[name]
        support = evaluate_viditq_shape(
            in_features=module.in_features,
            out_features=module.out_features,
        )
        if not support.supported:
            raise LTX095T5ViDiTQCoverageError(
                f"selected T5 module unsupported: {name}: {support.reason}"
            )
    for name in LTX095_T5_SKIPPED_FQNS:
        module = discovered_map[name]
        support = evaluate_viditq_shape(
            in_features=module.in_features,
            out_features=module.out_features,
        )
        if (module.out_features, module.in_features) != (4096, 10240) or support.supported:
            raise LTX095T5ViDiTQCoverageError(
                f"skipped T5 module contract diverged: {name}: {support}"
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

    resolved_kernels = kernels or resolve_viditq_kernels(
        device=execution_device,
        viditq_root=viditq_root,
    )
    replacements: dict[str, ViDiTQW8A8BF16Linear] = {}
    for name in sorted(LTX095_T5_SELECTED_FQNS):
        replacements[name] = ViDiTQW8A8BF16Linear.from_linear(
            discovered_map[name],
            kernels=resolved_kernels,
            viditq_root=viditq_root,
            collect_runtime_timing=collect_runtime_timing,
        )
    for name, replacement in replacements.items():
        _replace_submodule(text_encoder, name, replacement)

    storage = [module.storage_summary() for module in replacements.values()]
    remaining = [discovered_map[name] for name in sorted(LTX095_T5_SKIPPED_FQNS)]
    report = LTX095T5ViDiTQQuantizationReport(
        requested="int8_w8a8_viditq",
        effective="int8_w8a8_viditq",
        backend="viditq_extension",
        expected_count=LTX095_T5_LINEAR_COUNT,
        discovered_count=len(discovered),
        selected_count=len(replacements),
        quantized_count=len(replacements),
        skipped_count=len(remaining),
        missing_count=0,
        unexpected_count=0,
        remaining_bf16_linear_count=len(remaining),
        selected_parameter_ratio=LTX095_T5_SELECTED_PARAMETER_RATIO,
        expected_fqn_sha256=LTX095_T5_LINEAR_FQN_SHA256,
        selected_fqn_sha256=LTX095_T5_SELECTED_FQN_SHA256,
        skipped_fqn_sha256=LTX095_T5_SKIPPED_FQN_SHA256,
        attention_projection_count=96,
        ffn_up_projection_count=48,
        ffn_down_bf16_count=24,
        int8_weight_bytes=sum(item["weight_bytes"] for item in storage),
        bf16_scale_bytes=sum(item["scale_bytes"] for item in storage),
        int16_zero_point_bytes=sum(item["zero_point_bytes"] for item in storage),
        bf16_bias_bytes=sum(item["bias_bytes"] for item in storage),
        remaining_bf16_weight_bytes=sum(
            module.weight.numel() * module.weight.element_size() for module in remaining
        ),
        bf16_duplicate_weight_count=sum(
            item["bf16_duplicate_weight_count"] for item in storage
        ),
        dependency_fingerprints=(
            _fingerprint(resolved_kernels.fused_module_path),
            _fingerprint(resolved_kernels.qgemm_module_path),
        ),
        compute_capability=resolved_kernels.compute_capability,
        runtime_call_count=0,
        fallback_count=resolved_kernels.fallback_count,
        skipped_records=tuple(
            {
                "fqn": name,
                "shape": [4096, 10240],
                "out_features": 4096,
                "in_features": 10240,
                "reason": "main policy requires in_features <= 8192",
            }
            for name in sorted(LTX095_T5_SKIPPED_FQNS)
        ),
    )
    validate_ltx095_t5_viditq_coverage(text_encoder, report)
    setattr(text_encoder, "_mgerase_ltx095_t5_viditq_report", report)
    return report


__all__ = [
    "LTX095T5ViDiTQCoverageError",
    "LTX095T5ViDiTQQuantizationReport",
    "LTX095_T5_LINEAR_COUNT",
    "LTX095_T5_LINEAR_FQNS",
    "LTX095_T5_LINEAR_FQN_SHA256",
    "LTX095_T5_SELECTED_COUNT",
    "LTX095_T5_SELECTED_FQNS",
    "LTX095_T5_SELECTED_FQN_SHA256",
    "LTX095_T5_SELECTED_PARAMETER_RATIO",
    "LTX095_T5_SKIPPED_COUNT",
    "LTX095_T5_SKIPPED_FQNS",
    "LTX095_T5_SKIPPED_FQN_SHA256",
    "discover_ltx095_t5_linear_modules",
    "quantize_ltx095_t5_viditq",
    "summarize_ltx095_t5_viditq_runtime",
    "validate_ltx095_t5_viditq_coverage",
]
