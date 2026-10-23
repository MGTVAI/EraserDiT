"""Approximate FFN GEMM GELU_BIAS -> FP8 epilogue, opt-in SM89 eager path.

Unlike AdaLN->FP8, this accepts the existing BF16 input and preserves its
original static input quantization. The GEMM output/GELU rounding changes.
"""
import ctypes as ct
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import torch
from torch import nn
from layers.quantization.eraserdit_fp8 import NativeStaticFp8Linear, _quantize_activation
from layers.quantization.gelu import ProjectionBeforeFusedGelu
from layers.audited_allocations import empty, scaled_mm


@lru_cache(maxsize=1)
def library():
    path = os.environ.get('MGERASE_Q6_FFN_LIBRARY', '')
    if not path or not Path(path).is_file():
        raise RuntimeError('build scripts/build_ffn_epilogue.py and set MGERASE_Q6_FFN_LIBRARY')
    expected = os.environ.get('MGERASE_Q6_FFN_LIBRARY_SHA256')
    if expected and hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise RuntimeError('FFN epilogue library changed after benchmark preparation')
    lib = ct.CDLL(str(Path(path).resolve()))
    P, I = ct.c_void_p, ct.c_int
    lib.lt_create.argtypes = [I,I,I,P,P,P,P,I]; lib.lt_create.restype = P
    lib.lt_count.argtypes = [P]; lib.lt_count.restype = I
    lib.lt_run.argtypes = [P,I,P,P,P,P,P,I]; lib.lt_run.restype = I
    lib.lt_info.argtypes = [P,I,ct.POINTER(I),ct.POINTER(ct.c_float),ct.POINTER(ct.c_size_t)]
    lib.lt_destroy.argtypes = [P]; lib.lt_error.restype = ct.c_char_p
    return lib


class Workspace:
    def __init__(self):
        self.tensor = None

    def get(self, device):
        if self.tensor is None or self.tensor.device != device:
            self.tensor = torch.empty(32*1024**2, device=device, dtype=torch.uint8)
        return self.tensor


class EpilogueFeedForward(nn.Module):
    def __init__(self, original, workspace):
        super().__init__()
        self.net = original.net
        self.workspace = workspace
        self.handle = None
        self.signature = None
        self.calls = 0
        self.plans_created = 0
        self.register_buffer('output_inverse_scale', torch.tensor(8., dtype=torch.float32,
                             device=self.net[0].proj.weight_scale.device))
        library()

    def _plan(self, x):
        up = self.net[0].proj
        signature = (x.shape[0], str(x.device), up.weight_fp8.data_ptr(), up.weight_scale.data_ptr(),
                     up.activation_scale.data_ptr(), up.bias.data_ptr(), self.output_inverse_scale.data_ptr())
        lib = library()
        if self.signature == signature:
            return self.handle
        if self.handle:
            lib.lt_destroy(self.handle)
            self.handle = None
            self.signature = None
        handle = lib.lt_create(x.shape[0], 8192, 2048, up.weight_scale.data_ptr(),
            up.activation_scale.data_ptr(), up.bias.data_ptr(), self.output_inverse_scale.data_ptr(), 2)
        if not handle:
            raise RuntimeError(lib.lt_error().decode())
        info, waves, size = (ct.c_int*7)(), ct.c_float(), ct.c_size_t()
        status = lib.lt_info(handle, 0, info, ct.byref(waves), ct.byref(size))
        if status or list(info) != [60,20,15,1,0,1,0] or size.value > 32*1024**2:
            lib.lt_destroy(handle)
            raise RuntimeError('FFN epilogue heuristic differs from validated CUDA/cuBLASLt configuration')
        self.handle, self.signature = handle, signature
        self.plans_created += 1
        return handle

    def forward(self, value, *args, **kwargs):
        if (args or kwargs or self.training or torch.is_grad_enabled() or torch.compiler.is_compiling()
                or value.dtype != torch.bfloat16 or not value.is_cuda or not value.is_contiguous()
                or value.shape[-1] != 2048 or value.numel()//2048 != 32640):
            raise ValueError('FFN epilogue supports contiguous BF16 [*,32640,2048], eager inference only')
        up, down = self.net[0].proj, self.net[2]
        x = _quantize_activation(value.reshape(-1,2048), None, True, .125)[0]
        handle = self._plan(x)
        output = empty((32640,8192), device=x.device, dtype=torch.float8_e4m3fn, label='ffn_epilogue_output')
        workspace = self.workspace.get(x.device)
        with torch.cuda.device(x.device):
            status = library().lt_run(handle, 0, up.weight_fp8.data_ptr(), x.data_ptr(), output.data_ptr(),
                workspace.data_ptr(), torch.cuda.current_stream(x.device).cuda_stream, 1)
        if status:
            raise RuntimeError(f'FFN epilogue launch failed: {status}')
        result = scaled_mm(output, down.weight_fp8.t(), scale_a=down.activation_scale,
            scale_b=down.weight_scale, bias=down.bias, out_dtype=torch.bfloat16,
            use_fast_accum=True, label='ffn_epilogue_down_gemm')
        up.calls += 1
        up.gelu_calls += 1
        down.calls += 1
        self.calls += 1
        return result.reshape(*value.shape[:-1],2048)

    def __del__(self):
        if getattr(self, 'handle', None):
            try:
                library().lt_destroy(self.handle)
            except Exception:
                pass


def convert_ffn(model):
    # Validate every block before changing the first one.
    for block in model.transformer_blocks:
        net = block.ff.net
        if (len(net) != 3 or type(net[0]) is not ProjectionBeforeFusedGelu
                or not (type(net[1]) is nn.Identity or type(net[1]) is nn.Dropout and net[1].p == 0)
                or type(net[0].proj) is not NativeStaticFp8Linear or type(net[2]) is not NativeStaticFp8Linear
                or (net[0].proj.in_features, net[0].proj.out_features) != (2048,8192)
                or (net[2].in_features, net[2].out_features) != (8192,2048)
                or net[0].proj.bias is None or net[0].proj.gelu_lut is not None or net[2].gelu_lut is None):
            raise ValueError('FFN epilogue requires the original static FP8 FFN GELU/dropout layout')
    workspace = Workspace()
    for block in model.transformer_blocks:
        original = block.ff
        block.ff = EpilogueFeedForward(original, workspace).train(original.training)
    return dict(layers=len(model.transformer_blocks), workspace_bytes=32*1024**2,
                input='original BF16 -> static FP8 .125', approximate=True)
