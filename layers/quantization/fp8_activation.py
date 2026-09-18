"""Experimental fused activation quantization for native FP8 W8A8 Linear."""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

_SUPPORTED_DTYPES = {torch.bfloat16, torch.float16, torch.float32}


@triton.jit
def _quantize_fp8_per_row_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < width
    values = tl.load(
        input_ptr + row * width + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(values), axis=0)
    scale = tl.where(amax > 0.0, amax / 448.0, 1.0)
    scaled = values / scale
    scaled = tl.maximum(tl.minimum(scaled, 448.0), -448.0)
    tl.store(output_ptr + row * width + columns, scaled, mask=mask)
    tl.store(scale_ptr + row, scale)


def fused_fp8_per_row_available() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "backend": "triton_per_row",
        "triton_available": True,
        "triton_version": triton.__version__,
        "cuda_available": cuda_available,
        "reason": "available" if cuda_available else "CUDA is unavailable",
    }


def quantize_fp8_per_row_triton(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return E4M3 activations and one FP32 dequant scale per logical row."""

    if tensor.ndim != 2:
        raise ValueError("tensor must be two-dimensional")
    if tensor.dtype not in _SUPPORTED_DTYPES:
        allowed = ", ".join(str(dtype) for dtype in _SUPPORTED_DTYPES)
        raise TypeError(f"tensor dtype must be one of: {allowed}")
    if tensor.device.type != "cuda":
        raise RuntimeError("Triton FP8 per-row quantization requires CUDA")
    rows, width = tensor.shape
    if rows <= 0 or width <= 0:
        raise ValueError("tensor dimensions must be positive")
    source = tensor.contiguous()
    output = torch.empty_like(source, dtype=torch.float8_e4m3fn)
    scale = torch.empty((rows, 1), device=source.device, dtype=torch.float32)
    block_size = triton.next_power_of_2(width)
    if block_size > 65536:
        raise ValueError("row width exceeds fused Triton kernel limit")
    num_warps = 8 if block_size >= 2048 else 4
    _quantize_fp8_per_row_kernel[(rows,)](
        source,
        output,
        scale,
        width,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output, scale
