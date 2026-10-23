"""Opt-in cross-attention Query RMSNorm with native PyTorch reductions."""
import torch

from .config import CROSS_Q_RMSNORM_NATIVE_OP
from .rmsnorm_adaln import native_norm_failure
from .runtime import record_operator_fusion_call


def _capability_failure(query, norm):
    failure = native_norm_failure(norm)
    if failure is not None:
        return failure
    if query.ndim != 3 or query.shape[-1] != 2048 or query.numel() == 0:
        return 'query_shape'
    if not query.is_cuda or query.dtype != torch.bfloat16:
        return 'query_device_dtype'
    if not query.is_contiguous():
        return 'query_layout'
    weight = norm.weight
    if weight is None or weight.shape != (2048,) or norm.bias is not None:
        return 'norm_affine'
    if weight.dtype != torch.bfloat16 or weight.device != query.device or not weight.is_contiguous():
        return 'weight_device_dtype_layout'
    if norm.eps != 1e-5:
        return 'norm_epsilon'
    return None


def apply_fused_cross_q_rmsnorm(query, norm, *, decision):
    enabled = CROSS_Q_RMSNORM_NATIVE_OP in decision.effective_ops
    if not enabled:
        return norm(query)
    failure = _capability_failure(query, norm)
    if failure is not None and decision.forced:
        raise RuntimeError(f'forced {CROSS_Q_RMSNORM_NATIVE_OP} capability check failed: {failure}')
    if failure is not None:
        record_operator_fusion_call(CROSS_Q_RMSNORM_NATIVE_OP, shape=tuple(query.shape),
            eligible=False, fused=False, fallback_reason=failure)
        return norm(query)
    from .triton.cross_q_rmsnorm import triton_cross_q_rmsnorm_native
    output = triton_cross_q_rmsnorm_native(query, norm.weight, norm.eps)
    record_operator_fusion_call(CROSS_Q_RMSNORM_NATIVE_OP, shape=tuple(query.shape),
        eligible=True, fused=True)
    return output
