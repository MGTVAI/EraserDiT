"""Low-overhead CUDA module profiling for one LTX0.9.5 videoerase case.

The profiler is intentionally opt-in.  Forward hooks only enqueue CUDA events;
elapsed times are resolved after the request with one device synchronization.
Container boundaries and leaf operators are reported separately because their
times overlap and therefore must not be added together.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


SCHEMA_VERSION = 1
PROFILE_ENV_VAR = "MGERASE_LTX095_CASE_PROFILE"
PROFILE_MODES = frozenset({"quantizable", "detailed"})


@dataclass(frozen=True)
class ProfileTarget:
    name: str
    component: str
    scope_kind: str
    module: nn.Module


@dataclass(frozen=True)
class _PendingSample:
    start: Any
    input_tensors: tuple[tuple[tuple[int, ...], str, str], ...]


@dataclass(frozen=True)
class _CompletedSample:
    start: Any
    end: Any
    input_tensors: tuple[tuple[tuple[int, ...], str, str], ...]
    output_tensors: tuple[tuple[tuple[int, ...], str, str], ...]


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * float(percentile)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _iter_tensors(value: Any, *, _depth: int = 0):
    if _depth > 4:
        return
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item, _depth=_depth + 1)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item, _depth=_depth + 1)
        return
    for attribute in ("sample", "last_hidden_state"):
        if hasattr(value, attribute):
            yield from _iter_tensors(getattr(value, attribute), _depth=_depth + 1)
            return


def _tensor_signature(value: Any) -> tuple[tuple[tuple[int, ...], str, str], ...]:
    return tuple(
        (
            tuple(int(size) for size in tensor.shape),
            str(tensor.dtype).removeprefix("torch."),
            str(tensor.device.type),
        )
        for tensor in _iter_tensors(value)
    )


def _signature_as_list(
    signature: tuple[tuple[tuple[int, ...], str, str], ...],
) -> list[dict[str, Any]]:
    return [
        {"shape": list(shape), "dtype": dtype, "device_type": device_type}
        for shape, dtype, device_type in signature
    ]


def _module_weight_metadata(module: nn.Module) -> dict[str, Any]:
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor):
        weight_shape = [int(size) for size in weight.shape]
        weight_dtype = str(weight.dtype).removeprefix("torch.")
        weight_bytes = int(weight.numel() * weight.element_size())
    else:
        weight_shape = None
        weight_dtype = None
        weight_bytes = 0
    direct_parameter_count = 0
    direct_parameter_bytes = 0
    for parameter in module.parameters(recurse=False):
        try:
            direct_parameter_count += int(parameter.numel())
            direct_parameter_bytes += int(parameter.numel() * parameter.element_size())
        except (RuntimeError, ValueError):
            continue
    metadata = {
        "weight_shape": weight_shape,
        "weight_dtype": weight_dtype,
        "weight_bytes": weight_bytes,
        "direct_parameter_count": direct_parameter_count,
        "direct_parameter_bytes": direct_parameter_bytes,
    }
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        metadata["convolution"] = {
            "kernel_size": list(module.kernel_size),
            "stride": list(module.stride),
            "padding": list(module.padding),
            "dilation": list(module.dilation),
            "groups": int(module.groups),
        }
    return metadata


def _quantization_candidate(module: nn.Module) -> dict[str, Any]:
    short_type = module.__class__.__name__
    weight = getattr(module, "weight", None)
    if isinstance(module, nn.Conv3d):
        return {
            "candidate_kind": "conv3d_w8a8",
            "requires_exact_shape_benchmark": True,
        }
    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        return {
            "candidate_kind": "conv_w8a8",
            "requires_exact_shape_benchmark": True,
        }
    if isinstance(module, nn.Linear) or (
        isinstance(weight, torch.Tensor)
        and weight.ndim == 2
        and "linear" in short_type.lower()
    ):
        return {
            "candidate_kind": "linear_w8a8",
            "requires_exact_shape_benchmark": True,
        }
    if isinstance(module, nn.Embedding):
        return {
            "candidate_kind": "embedding_weight_storage",
            "requires_exact_shape_benchmark": True,
        }
    return {
        "candidate_kind": None,
        "requires_exact_shape_benchmark": False,
    }


class AsyncCudaModuleProfiler:
    """Record module CUDA intervals without synchronizing inside hooks."""

    def __init__(
        self,
        targets: Sequence[ProfileTarget],
        *,
        detail: str,
        max_samples_per_module: int = 0,
        event_factory: Callable[..., Any] | None = None,
        synchronize: Callable[[], None] | None = None,
        require_cuda_tensor: bool = True,
    ) -> None:
        normalized_detail = str(detail).strip().lower()
        if normalized_detail not in PROFILE_MODES:
            raise ValueError(
                f"case profile detail must be one of {sorted(PROFILE_MODES)}"
            )
        if type(max_samples_per_module) is not int or max_samples_per_module < 0:
            raise ValueError("max_samples_per_module must be a non-negative integer")
        names = [target.name for target in targets]
        if len(names) != len(set(names)):
            raise ValueError("case profile target names must be unique")
        module_ids = [id(target.module) for target in targets]
        if len(module_ids) != len(set(module_ids)):
            raise ValueError("a module cannot be registered as multiple profile targets")
        self.detail = normalized_detail
        self.max_samples_per_module = max_samples_per_module
        self._targets = tuple(targets)
        self._target_by_id = {id(target.module): target for target in targets}
        self._event_factory = event_factory or torch.cuda.Event
        self._synchronize = synchronize or torch.cuda.synchronize
        self._require_cuda_tensor = bool(require_cuda_tensor)
        self._starts: dict[int, list[_PendingSample]] = defaultdict(list)
        self._samples: dict[int, list[_CompletedSample]] = defaultdict(list)
        self._skipped_non_cuda_calls: dict[int, int] = defaultdict(int)
        self._dropped_sample_limit_calls: dict[int, int] = defaultdict(int)
        self._handles: list[Any] = []
        self._report: dict[str, Any] | None = None
        self._final_synchronize_count = 0

    def install(self) -> "AsyncCudaModuleProfiler":
        if self._handles:
            raise RuntimeError("case module profiler is already installed")
        if self._report is not None:
            raise RuntimeError("finalized case module profiler cannot be reinstalled")
        for target in self._targets:
            self._handles.append(
                target.module.register_forward_pre_hook(
                    self._pre_hook,
                    with_kwargs=True,
                )
            )
            self._handles.append(
                target.module.register_forward_hook(
                    self._post_hook,
                    with_kwargs=True,
                )
            )
        return self

    def _pre_hook(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        module_id = id(module)
        if self._require_cuda_tensor and not any(
            tensor.is_cuda for tensor in _iter_tensors((inputs, kwargs))
        ):
            self._skipped_non_cuda_calls[module_id] += 1
            return
        if self.max_samples_per_module and (
            len(self._samples[module_id]) + len(self._starts[module_id])
            >= self.max_samples_per_module
        ):
            self._dropped_sample_limit_calls[module_id] += 1
            return
        start = self._event_factory(enable_timing=True)
        start.record()
        self._starts[module_id].append(
            _PendingSample(
                start=start,
                input_tensors=_tensor_signature((inputs, kwargs)),
            )
        )

    def _post_hook(
        self,
        module: nn.Module,
        _inputs: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        module_id = id(module)
        starts = self._starts.get(module_id)
        if not starts:
            return
        pending = starts.pop()
        end = self._event_factory(enable_timing=True)
        end.record()
        self._samples[module_id].append(
            _CompletedSample(
                start=pending.start,
                end=end,
                input_tensors=pending.input_tensors,
                output_tensors=_tensor_signature(output),
            )
        )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _module_row(
        self,
        target: ProfileTarget,
        resolved_samples: Sequence[tuple[_CompletedSample, float]],
    ) -> dict[str, Any]:
        durations = [duration for _, duration in resolved_samples]
        signature_groups: dict[
            tuple[
                tuple[tuple[tuple[int, ...], str, str], ...],
                tuple[tuple[tuple[int, ...], str, str], ...],
            ],
            list[float],
        ] = defaultdict(list)
        for sample, duration in resolved_samples:
            signature_groups[(sample.input_tensors, sample.output_tensors)].append(
                duration
            )
        shapes = []
        for (input_signature, output_signature), values in sorted(
            signature_groups.items(), key=lambda item: sum(item[1]), reverse=True
        ):
            shapes.append(
                {
                    "call_count": len(values),
                    "total_ms": float(sum(values)),
                    "mean_ms": float(statistics.fmean(values)),
                    "input_tensors": _signature_as_list(input_signature),
                    "output_tensors": _signature_as_list(output_signature),
                }
            )
        row = {
            "name": target.name,
            "component": target.component,
            "scope_kind": target.scope_kind,
            "module_type": (
                f"{target.module.__class__.__module__}."
                f"{target.module.__class__.__qualname__}"
            ),
            "module_type_short": target.module.__class__.__name__,
            "call_count": len(durations),
            "total_ms": float(sum(durations)),
            "mean_ms": float(statistics.fmean(durations)) if durations else None,
            "median_ms": float(statistics.median(durations)) if durations else None,
            "p95_ms": float(_percentile(durations, 0.95)) if durations else None,
            "min_ms": float(min(durations)) if durations else None,
            "max_ms": float(max(durations)) if durations else None,
            "shape_signatures": shapes,
            "skipped_non_cuda_calls": int(
                self._skipped_non_cuda_calls.get(id(target.module), 0)
            ),
            "dropped_sample_limit_calls": int(
                self._dropped_sample_limit_calls.get(id(target.module), 0)
            ),
            **_module_weight_metadata(target.module),
            **_quantization_candidate(target.module),
        }
        return row

    def as_dict(self) -> dict[str, Any]:
        if self._report is not None:
            return self._report
        recorded_sample_count = sum(len(values) for values in self._samples.values())
        if recorded_sample_count:
            self._synchronize()
            self._final_synchronize_count += 1
        resolved_by_module: dict[int, list[tuple[_CompletedSample, float]]] = {}
        for module_id, samples in self._samples.items():
            resolved_by_module[module_id] = [
                (sample, float(sample.start.elapsed_time(sample.end)))
                for sample in samples
            ]
        rows = [
            self._module_row(
                target,
                resolved_by_module.get(id(target.module), ()),
            )
            for target in self._targets
        ]
        roots = {
            row["component"]: float(row["total_ms"])
            for row in rows
            if row["scope_kind"] == "component_root"
        }
        leaf_totals: dict[str, float] = defaultdict(float)
        for row in rows:
            if row["scope_kind"] == "leaf_operator":
                leaf_totals[row["component"]] += float(row["total_ms"])
        for row in rows:
            root_total = roots.get(row["component"], 0.0)
            leaf_total = leaf_totals.get(row["component"], 0.0)
            row["percent_of_component_root_cuda"] = (
                100.0 * float(row["total_ms"]) / root_total
                if root_total > 0.0
                else None
            )
            row["percent_of_profiled_leaf_cuda"] = (
                100.0 * float(row["total_ms"]) / leaf_total
                if row["scope_kind"] == "leaf_operator" and leaf_total > 0.0
                else None
            )

        components = {}
        for component in sorted({target.component for target in self._targets}):
            root_total = roots.get(component, 0.0)
            leaf_total = leaf_totals.get(component, 0.0)
            components[component] = {
                "component_root_cuda_ms": root_total,
                "profiled_leaf_cuda_ms": leaf_total,
                "profiled_leaf_coverage_percent": (
                    100.0 * leaf_total / root_total if root_total > 0.0 else None
                ),
                "unattributed_cuda_ms": max(root_total - leaf_total, 0.0),
                "root_call_count": sum(
                    int(row["call_count"])
                    for row in rows
                    if row["component"] == component
                    and row["scope_kind"] == "component_root"
                ),
                "leaf_module_count": sum(
                    1
                    for row in rows
                    if row["component"] == component
                    and row["scope_kind"] == "leaf_operator"
                ),
            }

        shape_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in rows:
            if row["scope_kind"] != "leaf_operator":
                continue
            for signature in row["shape_signatures"]:
                primary_input = (
                    signature["input_tensors"][0]
                    if signature["input_tensors"]
                    else None
                )
                key = (
                    row["component"],
                    row["module_type_short"],
                    row["candidate_kind"],
                    tuple(row["weight_shape"] or ()),
                    tuple((primary_input or {}).get("shape", ())),
                    (primary_input or {}).get("dtype"),
                )
                group = shape_groups.setdefault(
                    key,
                    {
                        "component": row["component"],
                        "module_type_short": row["module_type_short"],
                        "candidate_kind": row["candidate_kind"],
                        "weight_shape": row["weight_shape"],
                        "primary_input": primary_input,
                        "call_count": 0,
                        "total_ms": 0.0,
                        "module_names": [],
                    },
                )
                group["call_count"] += int(signature["call_count"])
                group["total_ms"] += float(signature["total_ms"])
                group["module_names"].append(row["name"])
        shape_group_rows = []
        for group in shape_groups.values():
            group["module_names"] = sorted(set(group["module_names"]))
            shape_group_rows.append(group)
        shape_group_rows.sort(key=lambda item: float(item["total_ms"]), reverse=True)

        rows.sort(
            key=lambda item: (
                item["component"],
                {"component_root": 0, "block_boundary": 1, "attention_boundary": 2}.get(
                    item["scope_kind"], 3
                ),
                -float(item["total_ms"]),
                item["name"],
            )
        )
        candidates = sorted(
            (
                row
                for row in rows
                if row["scope_kind"] == "leaf_operator"
                and row["candidate_kind"] is not None
            ),
            key=lambda item: float(item["total_ms"]),
            reverse=True,
        )
        unmatched_start_count = sum(len(values) for values in self._starts.values())
        self._report = {
            "schema_version": SCHEMA_VERSION,
            "kind": "ltx095_case_cuda_module_profile",
            "diagnostic_only": True,
            "detail": self.detail,
            "timing_contract": {
                "hook_behavior": "enqueue_cuda_events_only",
                "synchronizes_each_sample": False,
                "final_synchronize_count": self._final_synchronize_count,
                "component_roots_are_inclusive": True,
                "block_and_attention_boundaries_are_inclusive": True,
                "leaf_operators_are_additive_when_executed_on_one_stream": True,
                "do_not_sum_scope_kinds": True,
            },
            "target_count": len(self._targets),
            "recorded_sample_count": recorded_sample_count,
            "unmatched_start_count": unmatched_start_count,
            "skipped_non_cuda_call_count": int(
                sum(self._skipped_non_cuda_calls.values())
            ),
            "dropped_sample_limit_call_count": int(
                sum(self._dropped_sample_limit_calls.values())
            ),
            "components": components,
            "modules": rows,
            "quantization_candidates": candidates,
            "operator_shape_groups": shape_group_rows,
        }
        return self._report


def _is_attention_boundary(name: str, module: nn.Module) -> bool:
    short_type = module.__class__.__name__.lower()
    final_name = name.rsplit(".", 1)[-1].lower()
    return (
        "attention" in short_type
        or final_name in {"attn", "attn1", "attn2", "selfattention"}
    )


def _is_block_boundary(module: nn.Module) -> bool:
    return "block" in module.__class__.__name__.lower()


def _is_quantizable_leaf(module: nn.Module) -> bool:
    return _quantization_candidate(module)["candidate_kind"] is not None


def _component_targets(
    *,
    component: str,
    root: nn.Module,
    detail: str,
) -> list[ProfileTarget]:
    targets = [
        ProfileTarget(
            name=component,
            component=component,
            scope_kind="component_root",
            module=root,
        )
    ]
    for local_name, module in root.named_modules():
        if not local_name:
            continue
        full_name = f"{component}.{local_name}"
        children = tuple(module.children())
        if children:
            if _is_attention_boundary(local_name, module):
                targets.append(
                    ProfileTarget(
                        full_name, component, "attention_boundary", module
                    )
                )
            elif _is_block_boundary(module):
                targets.append(
                    ProfileTarget(full_name, component, "block_boundary", module)
                )
            continue
        if detail == "detailed" or _is_quantizable_leaf(module):
            targets.append(
                ProfileTarget(full_name, component, "leaf_operator", module)
            )
    return targets


def build_ltx095_case_profiler(
    *,
    text_encoder: nn.Module | None,
    transformer: nn.Module | None,
    vae: nn.Module | None,
    detail: str,
    max_samples_per_module: int = 0,
) -> AsyncCudaModuleProfiler:
    """Build and install a profiler for the four quantizable model components."""

    normalized_detail = str(detail).strip().lower()
    if normalized_detail not in PROFILE_MODES:
        raise ValueError(
            f"case profile detail must be one of {sorted(PROFILE_MODES)}"
        )
    roots: list[tuple[str, nn.Module]] = []
    if text_encoder is not None:
        roots.append(("text_encoder", text_encoder))
    if transformer is not None:
        roots.append(("transformer", transformer))
    if vae is not None:
        encoder = getattr(vae, "encoder", None)
        decoder = getattr(vae, "decoder", None)
        if isinstance(encoder, nn.Module):
            roots.append(("vae.encoder", encoder))
        if isinstance(decoder, nn.Module):
            roots.append(("vae.decoder", decoder))
    targets = [
        target
        for component, root in roots
        for target in _component_targets(
            component=component,
            root=root,
            detail=normalized_detail,
        )
    ]
    if not targets:
        raise ValueError("case profiler discovered no module targets")
    return AsyncCudaModuleProfiler(
        targets,
        detail=normalized_detail,
        max_samples_per_module=max_samples_per_module,
    ).install()


__all__ = [
    "AsyncCudaModuleProfiler",
    "PROFILE_ENV_VAR",
    "PROFILE_MODES",
    "ProfileTarget",
    "build_ltx095_case_profiler",
]
