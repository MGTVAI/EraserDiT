"""Install explicit extensions before offload hooks enumerate module buffers."""
from dataclasses import asdict
import torch
from config.q6_extensions import q6_extensions


def configure_q6_extensions(model, args):
    options = q6_extensions()
    if not options.active:
        return
    device = torch.device(args.device)
    if torch.cuda.get_device_capability(device) != (8,9):
        raise ValueError('Q6 extensions are validated only on SM89')
    report = dict(options=asdict(options))
    if options.qkv_calibration:
        from layers.quantization.qkv_fp8 import convert_qkv
        report['qkv'] = convert_qkv(model, options.qkv_calibration, device)
    if options.ffn_epilogue:
        from layers.quantization.ffn_epilogue import convert_ffn
        report['ffn_epilogue'] = convert_ffn(model)
    model._q6_extensions_report = report


def runtime_report(model):
    base = getattr(model, '_q6_extensions_report', None)
    if base is None:
        return None
    from layers.audited_allocations import snapshot
    qkv = [getattr(b.attn1, 'qkv_fp8', None) for b in model.transformer_blocks]
    return dict(base, qkv_calls=sum(getattr(m, 'calls', 0) for m in qkv),
                ffn_epilogue_calls=sum(getattr(b.ff, 'calls', 0) for b in model.transformer_blocks),
                allocations=snapshot())
