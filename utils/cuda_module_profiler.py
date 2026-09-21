"""Opt-in CUDA module profiling used by the LTX095 runtime."""

from __future__ import annotations

import statistics
from typing import Any, Mapping, Sequence

import torch
from torch import nn


class CudaModuleProfiler:
    """Explicit opt-in CUDA-event profiler; synchronizes and is diagnostic only."""

    def __init__(self, modules: Mapping[str, Sequence[nn.Module]]) -> None:
        self._modules = {name: tuple(values) for name, values in modules.items()}
        self._starts: dict[int, list[torch.cuda.Event]] = {}
        self._samples: dict[str, list[float]] = {name: [] for name in modules}
        self._handles: list[Any] = []

    def install(self) -> "CudaModuleProfiler":
        if self._handles:
            raise RuntimeError("CUDA module profiler is already installed")
        for category, modules in self._modules.items():
            for module in modules:
                self._handles.append(
                    module.register_forward_pre_hook(self._make_pre_hook())
                )
                self._handles.append(
                    module.register_forward_hook(self._make_post_hook(category))
                )
        return self

    def _make_pre_hook(self):
        def hook(module, _inputs) -> None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._starts.setdefault(id(module), []).append(event)
        return hook

    def _make_post_hook(self, category: str):
        def hook(module, _inputs, _output) -> None:
            starts = self._starts.get(id(module))
            if not starts:
                raise RuntimeError("CUDA profiler post-hook has no matching start")
            start = starts.pop()
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            end.synchronize()
            self._samples[category].append(float(start.elapsed_time(end)))
        return hook

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def as_dict(self) -> dict[str, Any]:
        if any(starts for starts in self._starts.values()):
            raise RuntimeError("CUDA profiler contains unmatched start events")
        return {
            "diagnostic_only": True,
            "synchronizes_each_sample": True,
            "categories": {
                name: {
                    "call_count": len(samples),
                    "total_ms": float(sum(samples)),
                    "median_ms": (
                        float(statistics.median(samples)) if samples else None
                    ),
                    "min_ms": float(min(samples)) if samples else None,
                    "max_ms": float(max(samples)) if samples else None,
                }
                for name, samples in self._samples.items()
            },
        }


def build_ltx095_cuda_profiler(
    transformer: nn.Module,
    *,
    detail: str,
) -> CudaModuleProfiler:
    normalized = detail.strip().lower()
    if normalized not in {"root", "detailed"}:
        raise ValueError("profile detail must be root or detailed")
    modules: dict[str, Sequence[nn.Module]] = {"transformer": (transformer,)}
    if normalized == "detailed":
        attention = tuple(
            module
            for name, module in transformer.named_modules()
            if name.endswith(".attn1") or name.endswith(".attn2")
        )
        skipped = tuple(
            transformer.get_submodule(name)
            for name in (
                "proj_in",
                "time_embed.emb.timestep_embedder.linear_1",
            )
        )
        modules.update(attention=attention, skipped_bf16=skipped)
    return CudaModuleProfiler(modules).install()


__all__ = ("CudaModuleProfiler", "build_ltx095_cuda_profiler")
