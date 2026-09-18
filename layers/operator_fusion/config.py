"""Configuration contract for optional fused operators."""

from __future__ import annotations

from collections.abc import Iterable

QK_RMSNORM_ROPE_OP = "qk_rmsnorm_rope"
RMSNORM_ADALN_OP = "rmsnorm_adaln"

OPERATOR_FUSION_BACKENDS = ("disabled", "auto", "triton")
OPERATOR_FUSION_OPS = (QK_RMSNORM_ROPE_OP, RMSNORM_ADALN_OP)


def normalize_operator_fusion_backend(value: object) -> str:
    backend = str(value).strip().lower()
    if backend not in OPERATOR_FUSION_BACKENDS:
        raise ValueError(
            "operator_fusion_backend must be one of "
            f"{OPERATOR_FUSION_BACKENDS}, got {value!r}"
        )
    return backend


def normalize_operator_fusion_ops(
    value: str | Iterable[str] | None,
) -> tuple[str, ...]:
    """Normalize an op selection; only an unset selection means every known op."""

    if value is None:
        requested = OPERATOR_FUSION_OPS
    elif isinstance(value, str):
        requested = tuple(part.strip() for part in value.split(",") if part.strip())
        if not requested:
            raise ValueError("operator_fusion_ops must not be empty when specified")
    else:
        requested = tuple(str(part).strip() for part in value if str(part).strip())
        if not requested:
            raise ValueError("operator_fusion_ops must not be empty when specified")

    unknown = tuple(op for op in requested if op not in OPERATOR_FUSION_OPS)
    if unknown:
        raise ValueError(
            f"unknown operator fusion op(s): {unknown}; supported={OPERATOR_FUSION_OPS}"
        )
    return tuple(dict.fromkeys(requested))
