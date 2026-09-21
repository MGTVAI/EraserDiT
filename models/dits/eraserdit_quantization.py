"""Selective EraserDiT-only INT8 conversion and observable runtime reporting."""
import time
import torch
from torch import nn


def validate_quantization(args, batch=None):
    mode = getattr(args, 'transformer_quantization', 'none')
    if mode == 'none':
        return
    if mode != 'int8_w8a8_native':
        raise ValueError('EraserDiT supports only none or int8_w8a8_native')
    c = args.pipeline_config
    if args.resource_policy != 'fullgpu' or args.enable_torch_compile:
        raise ValueError('EraserDiT INT8 currently requires fullgpu without compile')
    if args.operator_fusion_backend != 'disabled':
        raise ValueError('EraserDiT INT8 currently requires operator fusion disabled')
    if any(getattr(c,k,1) != 1 for k in ('sp_degree','cfg_degree','vae_degree')) or c.cfg_parallel_device:
        raise ValueError('EraserDiT INT8 first release is single-GPU only')
    if batch is not None and (batch.transformer_cache_mode != 'off' or batch.cache_text_projections):
        raise ValueError('EraserDiT INT8 first release requires caches off')


def selected_names(model, scope):
    suffixes = ['ff.net.0.proj', 'ff.net.2']
    if scope == 'blocks':
        suffixes += ['attn1.to_q','attn1.to_k','attn1.to_v','attn1.to_out.0',
                     'attn2.to_q','attn2.to_out.0']
    elif scope != 'ffn':
        raise ValueError('quantization scope must be blocks or ffn')
    return [f'transformer_blocks.{i}.{s}' for i in range(len(model.transformer_blocks)) for s in suffixes]


def quantize_transformer(model, scope='blocks'):
    from layers.quantization.eraserdit_int8 import NativeInt8Linear
    existing = getattr(model, '_eraserdit_int8_report', None)
    if existing is not None:
        if existing['scope'] != scope:
            raise ValueError('model already quantized with a different scope')
        return existing
    if getattr(model,'peft_config',None):
        raise ValueError('INT8 conversion with LoRA adapters is not supported')
    names = selected_names(model, scope)
    for name in names:
        module = model.get_submodule(name)
        if not isinstance(module, nn.Linear) or module.in_features%32 or module.out_features%32:
            raise ValueError(f'unsupported INT8 target: {name}')
        if module.weight.dtype != torch.bfloat16 or module.weight.device.type != 'cuda':
            raise ValueError(f'INT8 target must be CUDA BF16: {name}')
    device = model.get_submodule(names[0]).weight.device
    if torch.cuda.get_device_capability(device) < (8,0):
        raise ValueError('this INT8 implementation requires CUDA capability >= 8.0')
    # Execute the real backend before mutating any model layer.
    probe = torch.zeros((32,32), device=device,dtype=torch.int8)
    torch._int_mm(probe, probe.t())
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    original_bytes = quantized_bytes = 0
    for name in names:
        original = model.get_submodule(name)
        original_bytes += sum(t.numel()*t.element_size() for t in original.parameters())
        replacement = NativeInt8Linear.from_linear(original)
        quantized_bytes += sum(t.numel()*t.element_size() for t in replacement.buffers())
        parent, _, child = name.rpartition('.')
        setattr(model.get_submodule(parent), child, replacement)
    torch.cuda.synchronize(device)
    report = dict(mode='int8_w8a8_native',scope=scope,quantized_count=len(names),
                  selected_names=names,source_linear_bytes=original_bytes,
                  quantized_linear_bytes=quantized_bytes,bf16_duplicate_weight_count=0,
                  backend='torch._int_mm + triton quantization/epilogue',
                  conversion_seconds=time.perf_counter()-started)
    model._eraserdit_int8_report = report
    return report


def runtime_report(model):
    from layers.quantization.eraserdit_int8 import NativeInt8Linear
    base = getattr(model,'_eraserdit_int8_report',None)
    if base is None:
        return dict(mode='none',runtime_call_count=0)
    modules = [m for m in model.modules() if isinstance(m,NativeInt8Linear)]
    return dict(base,runtime_call_count=sum(m.calls for m in modules),
                executed_module_count=sum(m.calls>0 for m in modules), fallback_count=0)
