"""Lazy capability registry for the external ViDiT-Q INT8 kernels."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import torch

VIDITQ_DEFAULT_ROOT = Path("/root/viditq")
VIDITQ_M_ALIGNMENT = 128
VIDITQ_MAX_K = 8192
VIDITQ_MIN_K = 2048


class ViDiTQKernelUnavailableError(RuntimeError):
    """Raised before execution when the required compiled kernel is unavailable."""


@dataclass(frozen=True)
class ViDiTQSupport:
    supported: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.supported and self.reason is not None:
            raise ValueError("supported capability cannot carry a reason")
        if not self.supported and not self.reason:
            raise ValueError("unsupported capability requires a reason")


@dataclass(frozen=True)
class ViDiTQKernelSet:
    """Frozen callable set used by one quantized Linear instance."""

    quant_sum_bf16: Callable[..., torch.Tensor]
    w8a8_bf16_bias_weight_asym: Callable[..., torch.Tensor]
    fused_module_path: str
    qgemm_module_path: str
    compute_capability: tuple[int, int]

    @property
    def fallback_count(self) -> int:
        return 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "fused_module_path": self.fused_module_path,
            "qgemm_module_path": self.qgemm_module_path,
            "compute_capability": list(self.compute_capability),
            "required_symbols_present": True,
            "fallback_count": 0,
        }


def evaluate_viditq_shape(
    *,
    in_features: int,
    out_features: int,
) -> ViDiTQSupport:
    if in_features <= 0 or out_features <= 0:
        return ViDiTQSupport(False, "Linear dimensions must be positive")
    reasons: list[str] = []
    if in_features > VIDITQ_MAX_K:
        reasons.append("main policy requires in_features <= 8192")
    if in_features < VIDITQ_MIN_K:
        reasons.append("main policy requires in_features >= 2048")
    if in_features % 64 != 0:
        reasons.append("main policy requires in_features % 64 == 0")
    if out_features % 128 != 0:
        reasons.append("main policy requires out_features % 128 == 0")
    if in_features <= 4096:
        if in_features % 128 != 0:
            reasons.append(
                "fused activation kernel requires in_features % 128 == 0 "
                "when in_features <= 4096"
            )
    elif in_features % 256 != 0:
        reasons.append(
            "fused activation kernel requires in_features % 256 == 0 "
            "when in_features > 4096"
        )
    return (
        ViDiTQSupport(False, "; ".join(reasons))
        if reasons
        else ViDiTQSupport(True)
    )


def require_viditq_shape(*, in_features: int, out_features: int) -> None:
    support = evaluate_viditq_shape(
        in_features=in_features,
        out_features=out_features,
    )
    if not support.supported:
        raise ValueError(
            "unsupported ViDiT-Q Linear shape "
            f"(out_features={out_features}, in_features={in_features}): "
            f"{support.reason}"
        )


def evaluate_viditq_device_support(
    *,
    device_type: str,
    cuda_available: bool,
    compute_capability: tuple[int, int] | None,
) -> ViDiTQSupport:
    if device_type != "cuda":
        return ViDiTQSupport(False, "ViDiT-Q execution requires CUDA")
    if not cuda_available:
        return ViDiTQSupport(False, "torch.cuda.is_available() is false")
    if compute_capability is None:
        return ViDiTQSupport(False, "CUDA compute capability is unknown")
    if compute_capability < (8, 0):
        return ViDiTQSupport(
            False,
            "ViDiT-Q INT8 Tensor Core kernels require compute capability >= 8.0",
        )
    return ViDiTQSupport(True)


def _module_path(module: ModuleType | object) -> str:
    return str(getattr(module, "__file__", "<unknown>"))


def _default_module_loader(name: str) -> ModuleType:
    return importlib.import_module(name)


def resolve_viditq_kernels(
    *,
    device: torch.device | str,
    viditq_root: Path | str = VIDITQ_DEFAULT_ROOT,
    cuda_available: bool | None = None,
    compute_capability: tuple[int, int] | None = None,
    module_loader: Callable[[str], ModuleType | object] = _default_module_loader,
) -> ViDiTQKernelSet:
    """Resolve required symbols lazily without importing ViDiT-Q at package import."""

    resolved_device = torch.device(device)
    available = torch.cuda.is_available() if cuda_available is None else cuda_available
    capability = compute_capability
    if (
        capability is None
        and resolved_device.type == "cuda"
        and available
    ):
        try:
            capability = tuple(torch.cuda.get_device_capability(resolved_device))
        except (AssertionError, RuntimeError, ValueError) as exc:
            raise ViDiTQKernelUnavailableError(
                f"failed to probe {resolved_device}: {type(exc).__name__}: {exc}"
            ) from exc
    support = evaluate_viditq_device_support(
        device_type=resolved_device.type,
        cuda_available=bool(available),
        compute_capability=capability,
    )
    if not support.supported:
        raise ViDiTQKernelUnavailableError(support.reason or "unsupported device")

    if module_loader is _default_module_loader:
        kernel_root = str(Path(viditq_root).expanduser().resolve() / "kernels")
        if kernel_root not in sys.path:
            sys.path.insert(0, kernel_root)
        importlib.invalidate_caches()
    try:
        fused = module_loader("viditq_extension.fused")
        qgemm = module_loader("viditq_extension.qgemm")
    except Exception as exc:
        raise ViDiTQKernelUnavailableError(
            f"failed to import ViDiT-Q from {Path(viditq_root)}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    quant = getattr(fused, "quant_sum_bf16", None)
    gemm = getattr(qgemm, "w8a8_bf16_bias_weight_asym", None)
    missing: list[str] = []
    if not callable(quant):
        missing.append("viditq_extension.fused.quant_sum_bf16")
    if not callable(gemm):
        missing.append(
            "viditq_extension.qgemm.w8a8_bf16_bias_weight_asym"
        )
    if missing:
        raise ViDiTQKernelUnavailableError(
            "missing required symbol(s): "
            + ", ".join(missing)
            + f"; fused={_module_path(fused)}; qgemm={_module_path(qgemm)}"
        )
    if capability is None:
        raise ViDiTQKernelUnavailableError("CUDA compute capability is unknown")
    return ViDiTQKernelSet(
        quant_sum_bf16=quant,
        w8a8_bf16_bias_weight_asym=gemm,
        fused_module_path=_module_path(fused),
        qgemm_module_path=_module_path(qgemm),
        compute_capability=capability,
    )
