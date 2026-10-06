"""Selective EraserDiT-only W8A8 conversion and observable runtime reporting."""
import time
import torch
from torch import nn


def validate_quantization(args, batch=None):
    mode = getattr(args, 'transformer_quantization', 'none')
    if mode == 'none':
        return
    if mode not in ('int8_w8a8_native', 'fp8_w8a8_native', 'fp8_w8a8_tensorwise', 'fp8_w8a8_static'):
        raise ValueError('EraserDiT supports none, int8_w8a8_native, fp8_w8a8_native or fp8_w8a8_tensorwise / fp8_w8a8_static')
    c = args.pipeline_config
    if getattr(args, 'use_fsdp_inference', False):
        raise ValueError('EraserDiT W8A8 cannot wrap FSDP models')
    if (args.operator_fusion_backend != 'disabled' and getattr(c, 'sp_degree', 1) > 1
            and getattr(c, 'dit_parallel_backend', None) != 'nccl'):
        raise ValueError('EraserDiT W8A8 with SP requires operator fusion disabled')
    if c.cfg_parallel_device:
        raise ValueError('use cfg_degree for composable W8A8 CFG')


def selected_names(model, scope):
    suffixes = ['ff.net.0.proj', 'ff.net.2']
    if scope == 'blocks':
        suffixes += ['attn1.to_q','attn1.to_k','attn1.to_v','attn1.to_out.0',
                     'attn2.to_q','attn2.to_out.0']
    elif scope == 'ffn_up':
        suffixes = ['ff.net.0.proj']
    elif scope != 'ffn':
        raise ValueError('quantization scope must be blocks, ffn or ffn_up')
    return [f'transformer_blocks.{i}.{s}' for i in range(len(model.transformer_blocks)) for s in suffixes]


def quantize_transformer(model, scope='blocks', *, execution_device=None, mode='int8_w8a8_native'):
    from layers.quantization.eraserdit_int8 import NativeInt8Linear
    if mode not in ('int8_w8a8_native', 'fp8_w8a8_native', 'fp8_w8a8_tensorwise', 'fp8_w8a8_static'):
        raise ValueError('unsupported quantization mode')
    from layers.quantization.eraserdit_fp8 import NativeFp8Linear, NativeTensorwiseFp8Linear, NativeStaticFp8Linear
    linear_class = {'int8_w8a8_native': NativeInt8Linear,
                    'fp8_w8a8_native': NativeFp8Linear,
                    'fp8_w8a8_tensorwise': NativeTensorwiseFp8Linear,
                    'fp8_w8a8_static': NativeStaticFp8Linear}[mode]
    existing = getattr(model, '_eraserdit_quantization_report', None)
    if existing is not None:
        if existing['scope'] != scope or existing['mode'] != mode:
            raise ValueError('model already quantized with a different scope or mode')
        return existing
    if (getattr(model, '_layerwise_offload_manager', None) is not None
            or getattr(model, 'layerwise_offload_managers', None)
            or hasattr(model, '_block_compile_report')):
        raise ValueError('quantize before registering offload or compile')
    if getattr(model,'peft_config',None):
        raise ValueError('W8A8 conversion with LoRA adapters is not supported')
    names = selected_names(model, scope)
    for name in names:
        module = model.get_submodule(name)
        if not isinstance(module, nn.Linear) or module.in_features%32 or module.out_features%32:
            raise ValueError(f'unsupported W8A8 target: {name}')
        if module.weight.dtype != torch.bfloat16 or module.weight.device.type not in ('cuda', 'cpu'):
            raise ValueError(f'W8A8 target must be CPU/CUDA BF16: {name}')
    device = model.get_submodule(names[0]).weight.device
    source_device = device
    device = torch.device(execution_device) if execution_device is not None else device
    if device.type != 'cuda':
        raise ValueError('CPU quantization needs a CUDA execution_device')
    if torch.cuda.get_device_capability(device) < (8,0):
        raise ValueError('this INT8 implementation requires CUDA capability >= 8.0')
    # Execute the real backend before mutating any model layer.
    if mode.startswith('fp8_'):
        if torch.cuda.get_device_capability(device) < (8, 9):
            raise ValueError('FP8 requires CUDA capability >= 8.9')
        probe = torch.zeros((32, 32), device=device, dtype=torch.float8_e4m3fn)
        unit = torch.ones((), device=device)
        torch._scaled_mm(probe, probe.t(), scale_a=unit, scale_b=unit,
                         out_dtype=(torch.bfloat16 if linear_class.tensorwise else torch.float32),
                         use_fast_accum=linear_class.use_fast_accum)
    else:
        probe = torch.zeros((32,32), device=device,dtype=torch.int8)
        torch._int_mm(probe, probe.t())
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    # Only move GELU across an identity dropout. Other activation types and
    # nonzero dropout retain their existing semantics and execution path.
    gelu_targets = []
    if scope in ('blocks', 'ffn'):
        from diffusers.models.activations import GELU
        from layers.quantization.gelu import make_gelu_lut, ProjectionBeforeFusedGelu
        for block in model.transformer_blocks:
            net = block.ff.net
            if (type(net[0]) is GELU and net[0].approximate == 'tanh'
                    and (type(net[1]) is nn.Identity
                         or (type(net[1]) is nn.Dropout and net[1].p == 0))):
                gelu_targets.append(net)
        lut = make_gelu_lut(device) if gelu_targets else None
    original_bytes = quantized_bytes = 0
    for name in names:
        original = model.get_submodule(name)
        original_bytes += sum(t.numel()*t.element_size() for t in original.parameters())
        # CPU and CUDA FP32 division can round scales differently; at FP8 ties
        # this changes many packed weights. Always convert on the execution
        # GPU, one layer at a time, then restore the requested storage location.
        # Offload changes residency, not the quantized model's numerical path.
        replacement = linear_class.from_linear(original, execution_device=device).to(original.weight.device)
        quantized_bytes += sum(t.numel()*t.element_size() for t in replacement.buffers())
        parent, _, child = name.rpartition('.')
        setattr(model.get_submodule(parent), child, replacement)
    for net in gelu_targets:
        # Offload managers enumerate buffers by block and mutate tensor.data;
        # sharing one Tensor would deduplicate names and couple block residency.
        net[2].gelu_lut = lut.to(net[2].weight_scale.device, copy=True)
        net[0] = ProjectionBeforeFusedGelu(net[0].proj)
    torch.cuda.synchronize(device)
    report = dict(mode=mode,scope=scope,quantized_count=len(names),
                  selected_names=names,source_linear_bytes=original_bytes,
                  quantized_linear_bytes=quantized_bytes,bf16_duplicate_weight_count=0,
                  backend=('torch._int_mm / triton fused expansion GEMM' if mode == 'int8_w8a8_native'
                           else 'torch._scaled_mm E4M3 scaled BF16 output with bias, use_fast_accum=True' if mode == 'fp8_w8a8_static'
                           else 'torch._scaled_mm E4M3 scaled BF16 output with bias' if mode == 'fp8_w8a8_tensorwise'
                           else 'torch._scaled_mm E4M3 FP32 output, use_fast_accum=False') + (' + triton dynamic tensor quantization' if mode == 'fp8_w8a8_tensorwise'
                                else ' + triton static activation quantization' if mode == 'fp8_w8a8_static'
                                else ' + triton quantization/epilogue')
                          + (' + exact BF16 GELU lookup fusion' if gelu_targets else ''),
                  scale_granularity=('fixed-tensor/tensor' if mode == 'fp8_w8a8_static'
                                     else 'tensor/tensor' if mode == 'fp8_w8a8_tensorwise' else 'token/channel'),
                  activation_scale=0.125 if mode == 'fp8_w8a8_static' else None,
                  activation_saturation=56. if mode == 'fp8_w8a8_static' else None,
                  use_fast_accum=mode == 'fp8_w8a8_static',
                  fused_expansion_min_rows=1024 if mode == 'int8_w8a8_native' else None,
                  fused_gelu_count=len(gelu_targets),
                  gelu_lut_bytes_per_module=131072 if gelu_targets else 0,
                  conversion_seconds=time.perf_counter()-started, conversion_device=str(device),
                  storage_device=str(source_device))
    model._eraserdit_quantization_report = report
    if mode == 'int8_w8a8_native':
        model._eraserdit_int8_report = report
    return report


def runtime_report(model):
    base = getattr(model,'_eraserdit_quantization_report',None)
    if base is None:
        return dict(mode='none',runtime_call_count=0)
    from layers.quantization.eraserdit_int8 import NativeInt8Linear
    from layers.quantization.eraserdit_fp8 import NativeFp8Linear
    modules = [m for m in model.modules() if isinstance(m, (NativeInt8Linear, NativeFp8Linear))]
    return dict(base,runtime_call_count=sum(m.calls for m in modules),
                runtime_counts_include_compiled=False,
                fused_gemm_call_count=sum(getattr(m, "fused_calls", 0) for m in modules),
                fused_gelu_call_count=sum(getattr(m, "gelu_calls", 0) for m in modules),
                executed_module_count=sum(m.calls>0 for m in modules), fallback_count=0)
