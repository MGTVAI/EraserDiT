"""Optional fused operators for the local MGErase runtime."""

from .config import (
    OPERATOR_FUSION_BACKENDS,
    QK_RMSNORM_ROPE_OP,
    RMSNORM_ADALN_OP,
    normalize_operator_fusion_backend,
    normalize_operator_fusion_ops,
)
from .registry import (
    OperatorFusionDecision,
    get_operator_fusion_decision,
    resolve_operator_fusion_decision,
)

__all__ = [
    "OPERATOR_FUSION_BACKENDS",
    "OperatorFusionDecision",
    "QK_RMSNORM_ROPE_OP",
    "RMSNORM_ADALN_OP",
    "get_operator_fusion_decision",
    "normalize_operator_fusion_backend",
    "normalize_operator_fusion_ops",
    "resolve_operator_fusion_decision",
]
