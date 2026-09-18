"""Strict selective ViDiT-Q INT8 W8A8 adapter for LTX-Video 0.9.5."""

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
from models.dits.ltx095_quantization import (
    EXPECTED_LTX095_LINEAR_FQNS,
    EXPECTED_LTX095_LINEAR_FQN_SHA256,
    LTX095_TRANSFORMER_LINEAR_COUNT,
    discover_ltx095_linear_modules,
)

LTX095_VIDITQ_SELECTED_COUNT = 285
LTX095_VIDITQ_SELECTED_PARAMETER_RATIO = 0.9994534558529764
LTX095_VIDITQ_SKIPPED = {
    "proj_in": (
        (2048, 257),
        "K is not divisible by 64 and fused activation K is unsupported",
    ),
    "time_embed.emb.timestep_embedder.linear_1": (
        (2048, 256),
        "main policy requires in_features >= 2048",
    ),
}
LTX095_VIDITQ_SELECTED_FQNS = frozenset(
    EXPECTED_LTX095_LINEAR_FQNS - LTX095_VIDITQ_SKIPPED.keys()
)
LTX095_VIDITQ_SELECTED_FQN_SHA256 = hashlib.sha256(
    "\n".join(sorted(LTX095_VIDITQ_SELECTED_FQNS)).encode("utf-8")
).hexdigest()


class LTX095ViDiTQCoverageError(RuntimeError):
    """Raised when the frozen LTX095 287/285/2 contract diverges."""


@dataclass(frozen=True)
class LTX095ViDiTQQuantizationReport:
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
    skipped_records: tuple[dict[str, Any], ...]
    self_attention_projection_count: int
    cross_attention_projection_count: int
    ffn_linear_count: int
    other_linear_count: int
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

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["compute_capability"] = list(self.compute_capability)
        payload["skipped_records"] = [dict(item) for item in self.skipped_records]
        payload["dependency_fingerprints"] = [
            dict(item) for item in self.dependency_fingerprints
        ]
        return payload

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))


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


def _category_counts(names: set[str]) -> dict[str, int]:
    self_attention = sum(".attn1." in name for name in names)
    cross_attention = sum(".attn2." in name for name in names)
    ffn = sum(".ff." in name for name in names)
    return {
        "self_attention": self_attention,
        "cross_attention": cross_attention,
        "ffn": ffn,
        "other": len(names) - self_attention - cross_attention - ffn,
    }


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


def summarize_ltx095_viditq_runtime(transformer: nn.Module) -> dict[str, Any]:
    modules = [
        module
        for module in transformer.modules()
        if isinstance(module, ViDiTQW8A8BF16Linear)
    ]
    return {
        "module_count": len(modules),
        "runtime_call_count": sum(module.runtime_stats.call_count for module in modules),
        "failure_count": sum(module.runtime_stats.failure_count for module in modules),
        "fallback_count": sum(module.runtime_stats.fallback_count for module in modules),
        "logical_row_count": sum(module.runtime_stats.logical_row_count for module in modules),
        "effective_row_count": sum(module.runtime_stats.effective_row_count for module in modules),
        "padded_row_count": sum(module.runtime_stats.padded_row_count for module in modules),
        "padded_call_count": sum(module.runtime_stats.padded_call_count for module in modules),
        "activation_quantization_ms": sum(module.runtime_stats.activation_quantization_ms for module in modules),
        "row_padding_ms": sum(module.runtime_stats.row_padding_ms for module in modules),
        "qgemm_ms": sum(module.runtime_stats.qgemm_ms for module in modules),
        "linear_total_ms": sum(module.runtime_stats.total_ms for module in modules),
    }


def validate_ltx095_viditq_coverage(
    transformer: nn.Module,
    report: LTX095ViDiTQQuantizationReport,
) -> None:
    quantized = {
        name: module
        for name, module in transformer.named_modules()
        if name and isinstance(module, ViDiTQW8A8BF16Linear)
    }
    missing = sorted(LTX095_VIDITQ_SELECTED_FQNS - quantized.keys())
    unexpected = sorted(quantized.keys() - LTX095_VIDITQ_SELECTED_FQNS)
    if missing or unexpected:
        raise LTX095ViDiTQCoverageError(
            f"LTX095 ViDiT-Q coverage mismatch: missing={len(missing)} {missing[:8]}, "
            f"unexpected={len(unexpected)} {unexpected[:8]}"
        )
    remaining = dict(discover_ltx095_linear_modules(transformer))
    if set(remaining) != set(LTX095_VIDITQ_SKIPPED):
        raise LTX095ViDiTQCoverageError(
            f"remaining BF16 Linear mismatch: {sorted(remaining)}"
        )
    for name, (shape, _reason) in LTX095_VIDITQ_SKIPPED.items():
        module = remaining[name]
        actual = (module.out_features, module.in_features)
        if actual != shape or module.weight.dtype is not torch.bfloat16:
            raise LTX095ViDiTQCoverageError(
                f"invalid skipped module {name}: shape={actual}, dtype={module.weight.dtype}"
            )
    for module in quantized.values():
        validate_viditq_ordinary_storage(module)
    runtime = summarize_ltx095_viditq_runtime(transformer)
    if runtime["module_count"] != report.quantized_count:
        raise LTX095ViDiTQCoverageError("report/module quantized count mismatch")
    if report.bf16_duplicate_weight_count:
        raise LTX095ViDiTQCoverageError("BF16 duplicate weights detected")


def quantize_ltx095_transformer_viditq(
    transformer: nn.Module,
    *,
    execution_device: torch.device | str,
    viditq_root: Path | str = VIDITQ_DEFAULT_ROOT,
    kernels: ViDiTQKernelSet | None = None,
    collect_runtime_timing: bool = False,
) -> LTX095ViDiTQQuantizationReport:
    existing = getattr(transformer, "_mgerase_ltx095_viditq_report", None)
    if existing is not None:
        if not isinstance(existing, LTX095ViDiTQQuantizationReport):
            raise TypeError("invalid existing LTX095 ViDiT-Q report")
        validate_ltx095_viditq_coverage(transformer, existing)
        return existing
    if getattr(transformer, "_mgerase_ltx095_quantization_report", None) is not None:
        raise ValueError("LTX095 Transformer is already FP8-quantized")
    if _has_active_lora_adapter(transformer):
        raise ValueError("LTX095 ViDiT-Q INT8 W8A8 does not support active LoRA adapters")

    discovered = discover_ltx095_linear_modules(transformer)
    discovered_map = dict(discovered)
    discovered_names = set(discovered_map)
    missing = sorted(EXPECTED_LTX095_LINEAR_FQNS - discovered_names)
    unexpected = sorted(discovered_names - EXPECTED_LTX095_LINEAR_FQNS)
    if missing or unexpected or len(discovered) != LTX095_TRANSFORMER_LINEAR_COUNT:
        raise LTX095ViDiTQCoverageError(
            f"LTX095 Linear coverage mismatch: missing={len(missing)} {missing[:8]}, "
            f"unexpected={len(unexpected)} {unexpected[:8]}, discovered={len(discovered)}"
        )

    for name, (expected_shape, expected_reason) in LTX095_VIDITQ_SKIPPED.items():
        module = discovered_map[name]
        shape = (module.out_features, module.in_features)
        support = evaluate_viditq_shape(
            in_features=module.in_features,
            out_features=module.out_features,
        )
        if shape != expected_shape or support.supported:
            raise LTX095ViDiTQCoverageError(
                f"skipped contract diverged for {name}: shape={shape}, support={support}"
            )
        if expected_reason not in (support.reason or "") and name != "proj_in":
            raise LTX095ViDiTQCoverageError(
                f"skipped reason diverged for {name}: {support.reason}"
            )
    for name in LTX095_VIDITQ_SELECTED_FQNS:
        module = discovered_map[name]
        support = evaluate_viditq_shape(
            in_features=module.in_features,
            out_features=module.out_features,
        )
        if not support.supported:
            raise LTX095ViDiTQCoverageError(
                f"selected module is unsupported: {name}: {support.reason}"
            )

    source_device_types = {
        module.weight.device.type for module in discovered_map.values()
    }
    if "cpu" in source_device_types:
        target_device = torch.device(execution_device)
        if target_device.type != "cuda":
            raise RuntimeError(
                "CPU-loaded LTX095 ViDiT-Q conversion requires a CUDA execution_device"
            )
        transformer.to(target_device)

    resolved_kernels = kernels or resolve_viditq_kernels(
        device=execution_device,
        viditq_root=viditq_root,
    )
    replacements = {
        name: ViDiTQW8A8BF16Linear.from_linear(
            discovered_map[name],
            kernels=resolved_kernels,
            viditq_root=viditq_root,
            collect_runtime_timing=collect_runtime_timing,
        )
        for name in sorted(LTX095_VIDITQ_SELECTED_FQNS)
    }
    for name, replacement in replacements.items():
        _replace_submodule(transformer, name, replacement)

    storage = [module.storage_summary() for module in replacements.values()]
    remaining = [discovered_map[name] for name in LTX095_VIDITQ_SKIPPED]
    categories = _category_counts(set(replacements))
    skipped_records = tuple(
        {
            "fqn": name,
            "shape": list(shape),
            "out_features": shape[0],
            "in_features": shape[1],
            "reason": reason,
        }
        for name, (shape, reason) in sorted(LTX095_VIDITQ_SKIPPED.items())
    )
    report = LTX095ViDiTQQuantizationReport(
        requested="int8_w8a8_viditq",
        effective="int8_w8a8_viditq",
        backend="viditq_extension",
        expected_count=LTX095_TRANSFORMER_LINEAR_COUNT,
        discovered_count=len(discovered),
        selected_count=len(replacements),
        quantized_count=len(replacements),
        skipped_count=len(remaining),
        missing_count=0,
        unexpected_count=0,
        remaining_bf16_linear_count=len(remaining),
        selected_parameter_ratio=LTX095_VIDITQ_SELECTED_PARAMETER_RATIO,
        expected_fqn_sha256=EXPECTED_LTX095_LINEAR_FQN_SHA256,
        selected_fqn_sha256=LTX095_VIDITQ_SELECTED_FQN_SHA256,
        skipped_records=skipped_records,
        self_attention_projection_count=categories["self_attention"],
        cross_attention_projection_count=categories["cross_attention"],
        ffn_linear_count=categories["ffn"],
        other_linear_count=categories["other"],
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
    )
    validate_ltx095_viditq_coverage(transformer, report)
    setattr(transformer, "_mgerase_ltx095_viditq_report", report)
    return report


__all__ = [
    "LTX095ViDiTQCoverageError",
    "LTX095ViDiTQQuantizationReport",
    "LTX095_VIDITQ_SELECTED_COUNT",
    "LTX095_VIDITQ_SELECTED_FQNS",
    "LTX095_VIDITQ_SELECTED_FQN_SHA256",
    "LTX095_VIDITQ_SKIPPED",
    "quantize_ltx095_transformer_viditq",
    "summarize_ltx095_viditq_runtime",
    "validate_ltx095_viditq_coverage",
]
