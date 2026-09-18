"""Offload tools for HtoD / DtoH module weight transfer with pin_memory support.

Adapted from MGErase_origin utils_inference/memory/module_offload/offload_tools.py.
External dependencies (test_time decorator, GlobalValues) removed.
"""

from __future__ import annotations

import gc
import math
import os
from typing import Dict

import torch
import torch.nn as nn

# Lazy import to avoid circular dependency
_FlexibleModuleExtentBase: type | None = None


def _get_flexible_base():
    global _FlexibleModuleExtentBase
    if _FlexibleModuleExtentBase is None:
        from memory.backends.flexible_module_extent_base import (
            FlexibleModuleExtentBase as _Base,
        )
        _FlexibleModuleExtentBase = _Base
    return _FlexibleModuleExtentBase


# ---------------------------------------------------------------------------
# Slice-run helper (kept from origin, may be useful for large tensors)
# ---------------------------------------------------------------------------

def slice_run(
    func,
    input: torch.Tensor,
    output: torch.Tensor,
    in_dim=0,
    batch_num=2,
    out_dim=None,
):
    if out_dim is None:
        out_dim = in_dim

    in_shape = input.shape[in_dim]
    out_shape = output.shape[out_dim]

    batch_num = min(batch_num, in_shape, out_shape)
    in_stride = math.ceil(in_shape / batch_num)

    if in_shape >= out_shape:
        assert in_shape % out_shape == 0, "bad shape"
        radio = in_shape // out_shape
        out_stride = in_stride // radio
    else:
        assert out_shape % in_shape == 0, "bad shape"
        radio = out_shape // in_shape
        out_stride = in_stride * radio

    for i in range(batch_num):
        start_in = i * in_stride
        end_in = min((i + 1) * in_stride, in_shape)
        start_out = i * out_stride
        end_out = min((i + 1) * out_stride, out_shape)
        if end_in <= start_in or end_out <= start_out:
            break
        out_mem = output.narrow(dim=out_dim, start=start_out, length=(end_out - start_out))
        input_mem = input.narrow(dim=in_dim, start=start_in, length=(end_in - start_in))
        out_mem[...] = func(input_mem)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def flush_memory(sync=True):
    gc.collect()
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


def malloc_pin_memory(
    module: nn.Module,
    ref_device: torch.device | None = None,
    skip_flexible: bool = True,
    contigous: bool = True,
):
    """Allocate pinned CPU memory mirror for module weights."""
    return __malloc_pin_memory__(
        module, ref_device, skip_flexible=skip_flexible, contigous=contigous
    )


def release_module_pin_memory(module: nn.Module) -> None:
    """Release recursively registered pinned mirrors; safe to call repeatedly."""
    for submodule in module.modules():
        if hasattr(submodule, "pin_param"):
            delattr(submodule, "pin_param")
        if hasattr(submodule, "pin_buffer"):
            delattr(submodule, "pin_buffer")


def offload_module(
    module: nn.Module,
    contain_sub: bool = True,
    skip_flexible: bool = True,
    check_device: bool = True,
):
    """Move module weights from GPU to CPU (DtoH)."""
    return __offload_module__(
        module,
        contain_sub=contain_sub,
        skip_flexible=skip_flexible,
        check_device=check_device,
        pre_vious_name=module.__class__.__name__,
    )


def cuda_module(
    module: nn.Module,
    device: torch.device,
    contain_sub: bool = True,
    skip_flexible: bool = True,
    check_device: bool = True,
):
    """Move module weights from CPU to GPU (HtoD) using non_blocking pin_memory."""
    return __cuda_module__(
        module,
        device,
        contain_sub=contain_sub,
        skip_flexible=skip_flexible,
        check_device=check_device,
        pre_vious_name=module.__class__.__name__,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_parameter(
    tensor: torch.Tensor,
    *,
    requires_grad: bool,
) -> nn.Parameter:
    if isinstance(tensor, nn.Parameter) and tensor.requires_grad == requires_grad:
        return tensor
    return nn.Parameter(tensor.detach(), requires_grad=requires_grad)


def __malloc_pin_memory__(
    module: nn.Module,
    ref_device: torch.device | None = None,
    skip_flexible: bool = True,
    contigous: bool = True,
):
    assert module is not None, "module is None"
    FlexibleBase = _get_flexible_base()

    if isinstance(module, FlexibleBase):
        return
    if hasattr(module, "is_flexible") and module.is_flexible() and skip_flexible:
        return

    pin_param = getattr(module, "pin_param", {})
    for name, param in module.named_parameters(recurse=False):
        if name not in pin_param:
            if contigous:
                pinned = param.contiguous().cpu().pin_memory()
            else:
                pinned = param.cpu().pin_memory()
            pin_param[name] = _as_parameter(
                pinned,
                requires_grad=param.requires_grad,
            )

    pin_buffer = getattr(module, "pin_buffer", {})
    for name, buffer in module.named_buffers(recurse=False):
        if name not in pin_buffer:
            if contigous:
                pin_buffer[name] = buffer.contiguous().cpu().pin_memory()
            else:
                pin_buffer[name] = buffer.cpu().pin_memory()

    setattr(module, "pin_param", pin_param)
    setattr(module, "pin_buffer", pin_buffer)

    for name, sub_module in module.named_children():
        if len(name) < 1:
            continue
        __malloc_pin_memory__(
            sub_module,
            ref_device=ref_device,
            skip_flexible=skip_flexible,
            contigous=contigous,
        )


def __offload_module__(
    module: nn.Module,
    contain_sub: bool = True,
    skip_flexible: bool = True,
    check_device: bool = True,
    pre_vious_name: str = "",
):
    assert module is not None, "module is None"
    FlexibleBase = _get_flexible_base()

    if isinstance(module, FlexibleBase):
        return
    if hasattr(module, "is_flexible") and module.is_flexible() and skip_flexible:
        return

    DEBUG = bool(int(os.environ.get("MEMORY_OFFLOAD_DEBUG", "0")))

    pin_param = getattr(module, "pin_param", {})
    for name, param in module.named_parameters(recurse=False):
        long_name = f"{pre_vious_name}.{name}" if pre_vious_name else name
        if check_device and param.device.type == "cpu":
            if DEBUG:
                print(
                    f"warning offload-tools, param({long_name}) already on CPU before offload"
                )
        if DEBUG:
            print(f"offload-tools offload param {long_name}")
        target = param.cpu() if name not in pin_param else pin_param[name]
        module._parameters[name] = _as_parameter(
            target,
            requires_grad=param.requires_grad,
        )

    pin_buffer = getattr(module, "pin_buffer", {})
    for name, buffer in module.named_buffers(recurse=False):
        long_name = f"{pre_vious_name}.{name}" if pre_vious_name else name
        if check_device and buffer.device.type == "cpu":
            if DEBUG:
                print(
                    f"warning offload-tools, buffer({long_name}) already on CPU before offload"
                )
        if DEBUG:
            print(f"offload-tools offload buffer {long_name}")
        if name not in pin_buffer:
            module._buffers[name] = buffer.cpu()
        else:
            module._buffers[name] = pin_buffer[name]

    if contain_sub:
        for name, sub_module in module.named_children():
            if not name:
                continue
            __offload_module__(
                sub_module,
                contain_sub=contain_sub,
                skip_flexible=skip_flexible,
                check_device=check_device,
                pre_vious_name=f"{pre_vious_name}.{name}",
            )


def __cuda_module__(
    module: nn.Module,
    device: torch.device,
    contain_sub: bool = True,
    skip_flexible: bool = True,
    check_device: bool = True,
    pre_vious_name: str = "",
):
    assert module is not None, "module is None"
    FlexibleBase = _get_flexible_base()

    if isinstance(module, FlexibleBase):
        return
    if hasattr(module, "is_flexible") and module.is_flexible() and skip_flexible:
        return

    DEBUG = bool(int(os.environ.get("MEMORY_OFFLOAD_DEBUG", "0")))

    pin_param = getattr(module, "pin_param", {})
    for name, param in module.named_parameters(recurse=False):
        long_name = f"{pre_vious_name}.{name}" if pre_vious_name else name
        if check_device and param.device.type != "cpu":
            if DEBUG:
                print(
                    f"warning offload-tools, param({long_name}) not on CPU before onload"
                )
        if DEBUG:
            print(f"offload-tools onload param {long_name}")
        source = param if name not in pin_param else pin_param[name]
        module._parameters[name] = _as_parameter(
            source.to(device=device, non_blocking=True),
            requires_grad=param.requires_grad,
        )

    pin_buffer = getattr(module, "pin_buffer", {})
    for name, buffer in module.named_buffers(recurse=False):
        long_name = f"{pre_vious_name}.{name}" if pre_vious_name else name
        if check_device and buffer.device.type != "cpu":
            if DEBUG:
                print(
                    f"warning offload-tools, buffer({long_name}) not on CPU before onload"
                )
        if DEBUG:
            print(f"offload-tools onload buffer {long_name}")
        if name not in pin_buffer:
            module._buffers[name] = buffer.to(device=device, non_blocking=True)
        else:
            module._buffers[name] = pin_buffer[name].to(device=device, non_blocking=True)

    if contain_sub:
        for name, sub_module in module.named_children():
            if not name:
                continue
            __cuda_module__(
                sub_module,
                device=device,
                contain_sub=contain_sub,
                skip_flexible=skip_flexible,
                check_device=check_device,
                pre_vious_name=f"{pre_vious_name}.{name}",
            )
