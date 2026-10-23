"""Opt-in frozen, per-block QKV W8A8. All buffers follow normal offload.

One shared RN activation quantization, three independent weight scales/GEMMs.
No BF16 weight duplicate is retained. Calibration is input/checkpoint specific.
"""
import hashlib
import json
import math
from pathlib import Path
import torch
import triton
from torch import nn
from layers.quantization.eraserdit_fp8 import _quant_fp8_tensor
from layers.audited_allocations import empty_like, scaled_mm


def read_calibration(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if not data.get('completed') or not data.get('all_holdout_within_range') or len(data.get('layers', [])) != 28:
        raise ValueError('QKV calibration requires all 28 layers and completed holdout validation')
    for i, layer in enumerate(data['layers']):
        scale = layer.get('scale')
        if layer.get('block') != i or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f'invalid QKV scale at block {i}')
    return data, hashlib.sha256(raw).hexdigest()


class SharedQKV(nn.Module):
    def __init__(self, attn, scale, execution_device):
        super().__init__()
        storage = attn.to_q.weight.device
        for name in ('q', 'k', 'v'):
            source = getattr(attn, 'to_'+name)
            if type(source) is not nn.Linear or source.weight.shape != (2048, 2048) or source.weight.dtype != torch.bfloat16:
                raise ValueError('QKV FP8 requires unmodified BF16 2048x2048 Linear')
            weight = source.weight.detach().to(device=execution_device, dtype=torch.float32)
            maximum = weight.abs().amax()
            weight_scale = torch.where(maximum > 0, maximum/448., torch.ones_like(maximum))
            packed = (weight/weight_scale).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
            self.register_buffer(name+'_weight', packed.to(storage))
            self.register_buffer(name+'_scale', weight_scale.to(storage))
            self.register_buffer(name+'_bias', source.bias.detach().clone() if source.bias is not None else None)
        self.register_buffer('activation_scale', torch.tensor(scale, device=storage, dtype=torch.float32))
        self.calls = 0

    def forward(self, x):
        if (self.training or torch.is_grad_enabled() or torch.compiler.is_compiling()
                or not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous()
                or x.shape[-1] != 2048 or x.numel()//2048 % 16
                or x.device != self.activation_scale.device):
            raise ValueError('QKV FP8 requires same-device contiguous BF16, SM89 eager inference, M multiple of 16')
        quant = empty_like(x, dtype=torch.float8_e4m3fn, label='qkv_quant')
        _quant_fp8_tensor[(triton.cdiv(x.numel(), 8192),)](
            x, quant, self.activation_scale, x.numel(), 8192, None, False, num_warps=8)
        outputs = []
        for name in ('q', 'k', 'v'):
            result = scaled_mm(quant.reshape(-1, 2048), getattr(self, name+'_weight').t(),
                scale_a=self.activation_scale, scale_b=getattr(self, name+'_scale'),
                bias=getattr(self, name+'_bias'), out_dtype=torch.bfloat16,
                use_fast_accum=True, label='qkv_gemm')
            outputs.append(result.reshape(*x.shape[:-1], 2048))
        self.calls += 1
        return tuple(outputs)


def convert_qkv(model, path, execution_device):
    data, digest = read_calibration(path)
    if len(model.transformer_blocks) != 28 or getattr(model, 'peft_config', None):
        raise ValueError('QKV FP8 requires the 28-layer checkpoint without LoRA')
    for i, block in enumerate(model.transformer_blocks):
        attn = block.attn1
        group = SharedQKV(attn, data['layers'][i]['scale'], execution_device)
        group.train(attn.training)
        attn.add_module('qkv_fp8', group)
        for name in ('q', 'k', 'v'):
            setattr(attn, 'to_'+name, nn.Identity())
    return dict(layers=28, linears=84, calibration_sha256=digest,
                original_bf16_qkv_retained=False, activation_rounding='RN E4M3',
                scope='self-attention QKV only; output/cross projections remain BF16')
