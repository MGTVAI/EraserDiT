"""Capability probe and strict backend selection for FP8 W8A8 Linear."""

from __future__ import annotations

import importlib.metadata
from dataclasses import asdict, dataclass
from typing import Callable

import torch

from .config import FP8LinearBackend, FP8LinearConfig


class FP8BackendUnavailableError(RuntimeError):
    def __init__(self, requested: str, reason: str) -> None:
        super().__init__(
            f"FP8 Linear backend {requested!r} is unavailable: {reason}"
        )
        self.requested = requested
        self.reason = reason


@dataclass(frozen=True)
class FP8BackendSupport:
    supported: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.supported and self.reason is not None:
            raise ValueError("supported backend cannot carry a failure reason")
        if not self.supported and not self.reason:
            raise ValueError("unsupported backend requires a reason")


@dataclass(frozen=True)
class FP8BackendSelection:
    requested: FP8LinearBackend
    selected: FP8LinearBackend
    fallback_reasons: tuple[str, ...] = ()
    compute_capability: tuple[int, int] | None = None

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["requested"] = self.requested.value
        payload["selected"] = self.selected.value
        payload["fallback_reasons"] = list(self.fallback_reasons)
        if self.compute_capability is not None:
            payload["compute_capability"] = list(self.compute_capability)
        return payload


def evaluate_native_scaled_mm_support(
    *,
    device_type: str,
    compute_capability: tuple[int, int] | None,
    cuda_available: bool,
    has_scaled_mm: bool,
) -> FP8BackendSupport:
    if device_type != "cuda":
        return FP8BackendSupport(False, "native_scaled_mm requires CUDA")
    if not cuda_available:
        return FP8BackendSupport(False, "torch.cuda.is_available() is false")
    if not has_scaled_mm:
        return FP8BackendSupport(False, "torch._scaled_mm is unavailable")
    if compute_capability is None:
        return FP8BackendSupport(False, "compute capability is unknown")
    if compute_capability < (8, 9):
        return FP8BackendSupport(
            False,
            "E4M3FN Tensor Core GEMM requires compute capability >= 8.9",
        )
    return FP8BackendSupport(True)


def probe_native_scaled_mm(
    device: torch.device,
) -> tuple[FP8BackendSupport, tuple[int, int] | None]:
    resolved = torch.device(device)
    capability: tuple[int, int] | None = None
    if resolved.type == "cuda" and torch.cuda.is_available():
        try:
            capability = tuple(torch.cuda.get_device_capability(resolved))
        except (AssertionError, RuntimeError, ValueError) as exc:
            return (
                FP8BackendSupport(
                    False,
                    f"compute capability probe failed: {type(exc).__name__}: {exc}",
                ),
                None,
            )
    return (
        evaluate_native_scaled_mm_support(
            device_type=resolved.type,
            compute_capability=capability,
            cuda_available=torch.cuda.is_available(),
            has_scaled_mm=hasattr(torch, "_scaled_mm"),
        ),
        capability,
    )


def probe_torchao() -> FP8BackendSupport:
    try:
        version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return FP8BackendSupport(False, "torchao is not installed")
    return FP8BackendSupport(
        False,
        "torchao "
        f"{version} is comparison-only; no production tensor-subclass backend "
        "is registered for MGErase ordinary storage",
    )


def resolve_fp8_linear_backend(
    config: FP8LinearConfig,
    device: torch.device,
    *,
    native_probe: Callable[
        [torch.device],
        tuple[FP8BackendSupport, tuple[int, int] | None],
    ] = probe_native_scaled_mm,
) -> FP8BackendSelection:
    requested = config.backend
    if requested is FP8LinearBackend.DISABLED:
        return FP8BackendSelection(
            requested=requested,
            selected=FP8LinearBackend.DISABLED,
        )
    if requested is FP8LinearBackend.TORCHAO:
        support = probe_torchao()
        raise FP8BackendUnavailableError(
            requested.value,
            support.reason or "torchao backend is unsupported",
        )

    support, capability = native_probe(torch.device(device))
    if support.supported:
        return FP8BackendSelection(
            requested=requested,
            selected=FP8LinearBackend.NATIVE_SCALED_MM,
            compute_capability=capability,
        )
    reason = support.reason or "native_scaled_mm capability probe failed"
    raise FP8BackendUnavailableError(requested.value, reason)
