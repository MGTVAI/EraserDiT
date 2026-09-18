"""Strict LTX0.9.5 adapter for the model-agnostic FP8 W8A8 Linear backend."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from layers.quantization import (
    FP8ActivationQuantization,
    FP8BackendSelection,
    FP8LinearBackend,
    FP8LinearConfig,
    FP8W8A8Linear,
    resolve_fp8_linear_backend,
    summarize_fp8_linear_storage,
)

LTX095_TRANSFORMER_LINEAR_COUNT = 287
LTX095_TRANSFORMER_BLOCK_COUNT = 28


def _build_expected_linear_fqns() -> frozenset[str]:
    names = {
        "caption_projection.linear_1",
        "caption_projection.linear_2",
        "time_embed.emb.timestep_embedder.linear_1",
        "time_embed.emb.timestep_embedder.linear_2",
        "time_embed.linear",
        "proj_in",
        "proj_out",
    }
    for index in range(LTX095_TRANSFORMER_BLOCK_COUNT):
        prefix = f"transformer_blocks.{index}"
        for attention in ("attn1", "attn2"):
            for projection in ("to_q", "to_k", "to_v", "to_out.0"):
                names.add(f"{prefix}.{attention}.{projection}")
        names.add(f"{prefix}.ff.net.0.proj")
        names.add(f"{prefix}.ff.net.2")
    if len(names) != LTX095_TRANSFORMER_LINEAR_COUNT:
        raise AssertionError("invalid frozen LTX095 Linear contract")
    return frozenset(names)


EXPECTED_LTX095_LINEAR_FQNS = _build_expected_linear_fqns()
EXPECTED_LTX095_LINEAR_FQN_SHA256 = hashlib.sha256(
    "\n".join(sorted(EXPECTED_LTX095_LINEAR_FQNS)).encode("utf-8")
).hexdigest()


def _build_fused_selective_linear_fqns() -> frozenset[str]:
    names: set[str] = set()
    for index in range(LTX095_TRANSFORMER_BLOCK_COUNT):
        prefix = f"transformer_blocks.{index}"
        for projection in ("to_q", "to_k", "to_v", "to_out.0"):
            names.add(f"{prefix}.attn1.{projection}")
        for projection in ("to_q", "to_out.0"):
            names.add(f"{prefix}.attn2.{projection}")
        names.add(f"{prefix}.ff.net.0.proj")
        names.add(f"{prefix}.ff.net.2")
    if len(names) != 224:
        raise AssertionError("invalid fused-selective LTX095 Linear contract")
    if not names < EXPECTED_LTX095_LINEAR_FQNS:
        raise AssertionError("fused-selective contract must be a strict subset")
    return frozenset(names)


LTX095_FUSED_SELECTIVE_LINEAR_FQNS = _build_fused_selective_linear_fqns()
LTX095_FUSED_SELECTIVE_LINEAR_COUNT = len(
    LTX095_FUSED_SELECTIVE_LINEAR_FQNS
)


class LTX095QuantizationCoverageError(RuntimeError):
    """Raised before execution when the frozen 287-Linear contract diverges."""


@dataclass(frozen=True)
class LTX095QuantizationReport:
    requested: str
    effective: str
    backend: str | None
    granularity: str | None
    policy: str
    activation_quantization: str | None
    use_fast_accum: bool
    expected_count: int
    discovered_count: int
    quantized_count: int
    padded_count: int
    skipped_count: int
    remaining_bf16_linear_count: int
    bf16_duplicate_weight_count: int
    self_attention_projection_count: int
    cross_attention_projection_count: int
    ffn_linear_count: int
    other_linear_count: int
    weight_bytes: int
    scale_bytes: int
    bias_bytes: int
    expected_fqn_sha256: str
    compute_capability: tuple[int, int] | None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.compute_capability is not None:
            payload["compute_capability"] = list(self.compute_capability)
        return payload

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))


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


def discover_ltx095_linear_modules(
    transformer: nn.Module,
) -> tuple[tuple[str, nn.Linear], ...]:
    if not isinstance(transformer, nn.Module):
        raise TypeError("transformer must be torch.nn.Module")
    return tuple(
        (name, module)
        for name, module in transformer.named_modules()
        if name and isinstance(module, nn.Linear)
    )


def _coverage_difference(
    discovered_names: set[str],
) -> tuple[list[str], list[str]]:
    missing = sorted(EXPECTED_LTX095_LINEAR_FQNS - discovered_names)
    unexpected = sorted(discovered_names - EXPECTED_LTX095_LINEAR_FQNS)
    return missing, unexpected


def _format_coverage_error(missing: list[str], unexpected: list[str]) -> str:
    return (
        "LTX095 Linear coverage mismatch: "
        f"missing={len(missing)} {missing[:8]}, "
        f"unexpected={len(unexpected)} {unexpected[:8]}"
    )


def _replace_submodule(root: nn.Module, fqn: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = fqn.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, replacement)


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


def make_disabled_ltx095_quantization_report() -> LTX095QuantizationReport:
    return LTX095QuantizationReport(
        requested="none",
        effective="none",
        backend=None,
        granularity=None,
        policy="disabled",
        activation_quantization=None,
        use_fast_accum=False,
        expected_count=LTX095_TRANSFORMER_LINEAR_COUNT,
        discovered_count=0,
        quantized_count=0,
        padded_count=0,
        skipped_count=len(discovered) - storage.module_count,
        remaining_bf16_linear_count=len(discovered) - storage.module_count,
        bf16_duplicate_weight_count=0,
        self_attention_projection_count=0,
        cross_attention_projection_count=0,
        ffn_linear_count=0,
        other_linear_count=0,
        weight_bytes=0,
        scale_bytes=0,
        bias_bytes=0,
        expected_fqn_sha256=EXPECTED_LTX095_LINEAR_FQN_SHA256,
        compute_capability=None,
    )


def validate_ltx095_quantization_coverage(
    transformer: nn.Module,
    report: LTX095QuantizationReport,
) -> None:
    if report.policy == "all":
        expected_quantized = EXPECTED_LTX095_LINEAR_FQNS
    elif report.policy == "fused_selective":
        expected_quantized = LTX095_FUSED_SELECTIVE_LINEAR_FQNS
    else:
        raise LTX095QuantizationCoverageError(
            f"unsupported LTX095 FP8 policy in report: {report.policy}"
        )
    quantized_names = {
        name
        for name, module in transformer.named_modules()
        if name and isinstance(module, FP8W8A8Linear)
    }
    missing = sorted(expected_quantized - quantized_names)
    unexpected = sorted(quantized_names - expected_quantized)
    if missing or unexpected:
        raise LTX095QuantizationCoverageError(
            _format_coverage_error(missing, unexpected)
        )
    remaining_names = {
        name for name, _ in discover_ltx095_linear_modules(transformer)
    }
    expected_remaining = EXPECTED_LTX095_LINEAR_FQNS - expected_quantized
    if remaining_names != expected_remaining:
        raise LTX095QuantizationCoverageError(
            "remaining BF16 Linear contract mismatch: "
            f"expected={len(expected_remaining)}, actual={len(remaining_names)}"
        )
    storage = summarize_fp8_linear_storage(transformer)
    if storage.module_count != len(expected_quantized):
        raise LTX095QuantizationCoverageError(
            f"quantized module count is {storage.module_count}, "
            f"expected {len(expected_quantized)}"
        )
    if storage.bf16_duplicate_weight_count:
        raise LTX095QuantizationCoverageError("BF16 duplicate weights detected")
    if report.quantized_count != storage.module_count:
        raise LTX095QuantizationCoverageError("report/module quantized count mismatch")


def quantize_ltx095_transformer(
    transformer: nn.Module,
    config: FP8LinearConfig,
    *,
    execution_device: torch.device | str,
    selection: FP8BackendSelection | None = None,
    policy: str = "all",
) -> LTX095QuantizationReport:
    if config.backend is FP8LinearBackend.DISABLED:
        raise ValueError("disabled FP8 config must use the pipeline hard bypass")
    normalized_policy = str(policy).strip().lower()
    if normalized_policy == "all":
        selected_names = EXPECTED_LTX095_LINEAR_FQNS
        requested_mode = "fp8_w8a8"
    elif normalized_policy == "fused_selective":
        selected_names = LTX095_FUSED_SELECTIVE_LINEAR_FQNS
        requested_mode = "fp8_w8a8_triton_selective"
        if config.backend is not FP8LinearBackend.NATIVE_SCALED_MM:
            raise ValueError("fused_selective requires native_scaled_mm")
        if config.activation_quantization is not FP8ActivationQuantization.TRITON_PER_ROW:
            raise ValueError(
                "fused_selective requires triton_per_row activation quantization"
            )
        if not config.use_fast_accum:
            raise ValueError("fused_selective requires fast accumulation")
    else:
        raise ValueError(f"unsupported LTX095 FP8 policy: {policy}")
    existing = getattr(transformer, "_mgerase_ltx095_quantization_report", None)
    if existing is not None:
        if not isinstance(existing, LTX095QuantizationReport):
            raise TypeError("invalid existing LTX095 quantization report")
        if existing.policy != normalized_policy:
            raise ValueError("existing LTX095 quantization policy does not match request")
        validate_ltx095_quantization_coverage(transformer, existing)
        return existing
    if _has_active_lora_adapter(transformer):
        raise ValueError("LTX095 FP8 W8A8 does not support active LoRA adapters")

    discovered = discover_ltx095_linear_modules(transformer)
    discovered_names = {name for name, _ in discovered}
    missing, unexpected = _coverage_difference(discovered_names)
    if missing or unexpected:
        raise LTX095QuantizationCoverageError(
            _format_coverage_error(missing, unexpected)
        )
    if len(discovered) != LTX095_TRANSFORMER_LINEAR_COUNT:
        raise LTX095QuantizationCoverageError(
            f"discovered {len(discovered)} Linear modules, expected 287"
        )

    frozen_selection = selection or resolve_fp8_linear_backend(
        config,
        torch.device(execution_device),
    )
    replacements = {
        name: FP8W8A8Linear.from_linear(
            module,
            config=config,
            selection=frozen_selection,
        )
        for name, module in discovered
        if name in selected_names
    }
    for name, replacement in replacements.items():
        _replace_submodule(transformer, name, replacement)

    storage = summarize_fp8_linear_storage(transformer)
    categories = _category_counts(set(selected_names))
    report = LTX095QuantizationReport(
        requested=requested_mode,
        effective=requested_mode,
        backend=frozen_selection.selected.value,
        granularity=config.granularity.value,
        policy=normalized_policy,
        activation_quantization=config.activation_quantization.value,
        use_fast_accum=config.use_fast_accum,
        expected_count=LTX095_TRANSFORMER_LINEAR_COUNT,
        discovered_count=len(discovered),
        quantized_count=storage.module_count,
        padded_count=storage.padded_module_count,
        skipped_count=len(discovered) - storage.module_count,
        remaining_bf16_linear_count=len(discovered) - storage.module_count,
        bf16_duplicate_weight_count=storage.bf16_duplicate_weight_count,
        self_attention_projection_count=categories["self_attention"],
        cross_attention_projection_count=categories["cross_attention"],
        ffn_linear_count=categories["ffn"],
        other_linear_count=categories["other"],
        weight_bytes=storage.weight_bytes,
        scale_bytes=storage.scale_bytes,
        bias_bytes=storage.bias_bytes,
        expected_fqn_sha256=EXPECTED_LTX095_LINEAR_FQN_SHA256,
        compute_capability=frozen_selection.compute_capability,
    )
    validate_ltx095_quantization_coverage(transformer, report)
    setattr(transformer, "_mgerase_ltx095_quantization_report", report)
    return report


__all__ = [
    "EXPECTED_LTX095_LINEAR_FQNS",
    "EXPECTED_LTX095_LINEAR_FQN_SHA256",
    "LTX095_FUSED_SELECTIVE_LINEAR_COUNT",
    "LTX095_FUSED_SELECTIVE_LINEAR_FQNS",
    "LTX095QuantizationCoverageError",
    "LTX095QuantizationReport",
    "LTX095_TRANSFORMER_LINEAR_COUNT",
    "discover_ltx095_linear_modules",
    "make_disabled_ltx095_quantization_report",
    "quantize_ltx095_transformer",
    "validate_ltx095_quantization_coverage",
]
