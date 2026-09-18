"""Joint Q/K RMSNorm + adjacent-pair RoPE Triton kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mul_rn_f32(lhs, rhs):
    """Keep the eager RoPE multiplication's standalone FP32 rounding."""

    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [lhs, rhs],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rn_f32(lhs, rhs):
    """Prevent contraction across eager RoPE's multiply/add boundary."""

    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [lhs, rhs],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _joint_qk_rmsnorm_rope_kernel(
    query_ptr,
    key_ptr,
    query_weight_ptr,
    key_weight_ptr,
    cos_ptr,
    sin_ptr,
    query_output_ptr,
    key_output_ptr,
    WIDTH: tl.constexpr,
    EPSILON: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, WIDTH)
    row_offsets = row_idx * WIDTH + offsets

    query = tl.load(query_ptr + row_offsets).to(tl.float32)
    query_variance = tl.sum(query * query, axis=0) / WIDTH
    query_rstd = tl.rsqrt(query_variance + EPSILON)
    query_weight = tl.load(query_weight_ptr + offsets).to(tl.float32)
    # torch.nn.RMSNorm returns BF16 before the caller applies RoPE. Preserve
    # that observable rounding boundary while keeping the intermediate local.
    normalized_query = (query * query_rstd * query_weight).to(tl.bfloat16)
    query_pairs = tl.reshape(normalized_query, (WIDTH // 2, 2))
    query_even, query_odd = tl.split(query_pairs)
    rotated_query = tl.reshape(
        tl.join(-query_odd, query_even), (WIDTH,)
    ).to(tl.float32)
    cos = tl.load(cos_ptr + row_offsets)
    sin = tl.load(sin_ptr + row_offsets)
    query_output = _add_rn_f32(
        _mul_rn_f32(normalized_query.to(tl.float32), cos),
        _mul_rn_f32(rotated_query, sin),
    )
    tl.store(query_output_ptr + row_offsets, query_output)

    key = tl.load(key_ptr + row_offsets).to(tl.float32)
    key_variance = tl.sum(key * key, axis=0) / WIDTH
    key_rstd = tl.rsqrt(key_variance + EPSILON)
    key_weight = tl.load(key_weight_ptr + offsets).to(tl.float32)
    normalized_key = (key * key_rstd * key_weight).to(tl.bfloat16)
    key_pairs = tl.reshape(normalized_key, (WIDTH // 2, 2))
    key_even, key_odd = tl.split(key_pairs)
    rotated_key = tl.reshape(tl.join(-key_odd, key_even), (WIDTH,)).to(
        tl.float32
    )
    cos = tl.load(cos_ptr + row_offsets)
    sin = tl.load(sin_ptr + row_offsets)
    key_output = _add_rn_f32(
        _mul_rn_f32(normalized_key.to(tl.float32), cos),
        _mul_rn_f32(rotated_key, sin),
    )
    tl.store(key_output_ptr + row_offsets, key_output)


def triton_qk_rmsnorm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply paired full-width RMSNorm and RoPE in one Triton launch."""

    width = query.shape[-1]
    row_count = query.numel() // width
    query_output = torch.empty_like(query)
    key_output = torch.empty_like(key)
    _joint_qk_rmsnorm_rope_kernel[(row_count,)](
        query,
        key,
        query_weight,
        key_weight,
        cos,
        sin,
        query_output,
        key_output,
        WIDTH=width,
        EPSILON=epsilon,
        num_warps=8,
    )
    return query_output, key_output
