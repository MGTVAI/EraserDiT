"""Fuse RMSNorm pointwise work and retain native mean/epsilon/rsqrt order.

The first kernel casts and squares into the same contiguous FP32 shape used
by Diffusers. The last kernel preserves both native BF16 rounding boundaries.
This is a partial fusion; it does not replace the reduction with tl.sum.
"""
import torch
from layers.audited_allocations import empty_like as audited_empty_like
import triton
import triton.language as tl

from .qk_rmsnorm_rope import _mul_rn_f32
from .rmsnorm_native import _inverse_rms


@triton.jit
def _norm_affine(Q, W, R, O, N: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    value = tl.load(Q + index, valid, 0).to(tl.float32)
    inverse = tl.load(R + index // WIDTH, valid, 0)
    normalized = _mul_rn_f32(value, inverse).to(tl.bfloat16).to(tl.float32)
    weight = tl.load(W + index % WIDTH, valid, 0).to(tl.float32)
    tl.store(O + index, _mul_rn_f32(normalized, weight).to(tl.bfloat16), valid)


def triton_cross_q_rmsnorm_native(query, weight, eps):
    inverse = _inverse_rms(query, eps)
    output = audited_empty_like(query, label='cross_q_output')
    _norm_affine[(triton.cdiv(query.numel(),1024),)](
        query, weight, inverse, output, query.numel(), query.shape[-1], 1024,
        num_warps=4)
    return output
