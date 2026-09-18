"""Side-effect-free fused-op registry and request selection."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

from .config import (
    QK_RMSNORM_ROPE_OP,
    RMSNORM_ADALN_OP,
    normalize_operator_fusion_backend,
    normalize_operator_fusion_ops,
)


@dataclass(frozen=True)
class OperatorFusionRegistration:
    name: str
    signed_sp_degrees: frozenset[int]


@dataclass(frozen=True)
class OperatorFusionDecision:
    requested_backend: str
    requested_ops: tuple[str, ...]
    effective_ops: tuple[str, ...]
    fallback_reasons: tuple[str, ...]
    sp_degree: int
    forced: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_backend": self.requested_backend,
            "requested_ops": list(self.requested_ops),
            "effective_ops": list(self.effective_ops),
            "fallback_reasons": list(self.fallback_reasons),
            "sp_degree": self.sp_degree,
            "forced": self.forced,
        }


# Stage 1 and stage 2 passed formal 1080p acceptance on SP1/SP2/SP4.  ``auto``
# may therefore select both by default on those signed topologies while the
# global backend default remains ``disabled``.
_REGISTRY = {
    QK_RMSNORM_ROPE_OP: OperatorFusionRegistration(
        name=QK_RMSNORM_ROPE_OP,
        signed_sp_degrees=frozenset({1, 2, 4}),
    ),
    RMSNORM_ADALN_OP: OperatorFusionRegistration(
        name=RMSNORM_ADALN_OP,
        signed_sp_degrees=frozenset({1, 2, 4}),
    ),
}


def _triton_is_available() -> bool:
    try:
        return importlib.util.find_spec("triton") is not None
    except (ImportError, ValueError):
        return False


def _resolve_sp_degree(server_args: Any) -> int:
    parallel_context = getattr(server_args, "parallel_context", None)
    plan = getattr(parallel_context, "plan", None)
    if plan is not None:
        return max(1, int(getattr(plan, "sp_degree", 1) or 1))
    return max(1, int(getattr(server_args, "sp_degree", 1) or 1))


def resolve_operator_fusion_decision(server_args: Any) -> OperatorFusionDecision:
    backend = normalize_operator_fusion_backend(
        getattr(server_args, "operator_fusion_backend", "disabled")
    )
    requested_ops = normalize_operator_fusion_ops(
        getattr(server_args, "operator_fusion_ops", None)
    )
    sp_degree = _resolve_sp_degree(server_args)
    compile_enabled = bool(getattr(server_args, "enable_torch_compile", False))

    if backend == "triton" and compile_enabled:
        raise ValueError(
            "operator_fusion_backend='triton' cannot be combined with "
            "enable_torch_compile=True"
        )

    if backend == "disabled":
        return OperatorFusionDecision(
            requested_backend=backend,
            requested_ops=requested_ops,
            effective_ops=(),
            fallback_reasons=("backend_disabled",),
            sp_degree=sp_degree,
            forced=False,
        )

    if not _triton_is_available():
        if backend == "triton":
            raise RuntimeError("Triton is unavailable for forced operator fusion")
        return OperatorFusionDecision(
            requested_backend=backend,
            requested_ops=requested_ops,
            effective_ops=(),
            fallback_reasons=("triton_unavailable",),
            sp_degree=sp_degree,
            forced=False,
        )

    if backend == "triton":
        return OperatorFusionDecision(
            requested_backend=backend,
            requested_ops=requested_ops,
            effective_ops=requested_ops,
            fallback_reasons=(),
            sp_degree=sp_degree,
            forced=True,
        )

    if compile_enabled:
        return OperatorFusionDecision(
            requested_backend=backend,
            requested_ops=requested_ops,
            effective_ops=(),
            fallback_reasons=("torch_compile_active",),
            sp_degree=sp_degree,
            forced=False,
        )

    effective_ops: list[str] = []
    fallback_reasons: list[str] = []
    for op_name in requested_ops:
        registration = _REGISTRY[op_name]
        if not registration.signed_sp_degrees:
            fallback_reasons.append("op_not_signed")
        elif sp_degree not in registration.signed_sp_degrees:
            fallback_reasons.append("topology_not_signed")
        else:
            effective_ops.append(op_name)
    return OperatorFusionDecision(
        requested_backend=backend,
        requested_ops=requested_ops,
        effective_ops=tuple(effective_ops),
        fallback_reasons=tuple(dict.fromkeys(fallback_reasons)),
        sp_degree=sp_degree,
        forced=False,
    )


def get_operator_fusion_decision(server_args: Any) -> OperatorFusionDecision:
    decision = getattr(server_args, "operator_fusion_decision", None)
    if isinstance(decision, OperatorFusionDecision):
        return decision
    return resolve_operator_fusion_decision(server_args)
