"""Explicit, opt-in single-worker Q6 extensions; legacy defaults stay unchanged."""
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


@dataclass(frozen=True)
class Q6Extensions:
    sage_peripheral: bool = False
    targeted_fill: bool = False
    qkv_calibration: str = ''
    ffn_epilogue: bool = False
    mask_sta: bool = False
    mask_sta_library: str = ''

    @property
    def active(self):
        return (self.sage_peripheral or self.targeted_fill or bool(self.qkv_calibration)
                or self.ffn_epilogue or self.mask_sta)


@lru_cache(maxsize=1)
def q6_extensions():
    def flag(name):
        value = os.environ.get(name, '0')
        if value not in ('0', '1'):
            raise ValueError(f'{name} must be 0 or 1')
        return value == '1'
    return Q6Extensions(
        sage_peripheral=flag('MGERASE_Q6_SAGE_PERIPHERAL'),
        targeted_fill=flag('MGERASE_Q6_TARGETED_FILL'),
        qkv_calibration=os.environ.get('MGERASE_Q6_QKV_CALIBRATION', ''),
        ffn_epilogue=flag('MGERASE_Q6_FFN_EPILOGUE'),
        mask_sta=flag('MGERASE_Q6_MASK_STA'),
        mask_sta_library=os.environ.get('MGERASE_Q6_MASK_STA_LIBRARY', ''))


def validate_q6_extensions(args):
    options = q6_extensions()
    if not options.active:
        return
    config = args.pipeline_config
    if (getattr(config, 'dit_parallel_backend', None) == 'nccl'
            or getattr(config, 'cfg_degree', 1) != 1
            or getattr(config, 'sp_degree', 1) != 1
            or getattr(config, 'tp_degree', 1) != 1
            or getattr(config, 'cfg_parallel_device', None)
            or getattr(args, 'dit_layerwise_offload', False)
            or getattr(args, 'use_fsdp_inference', False)
            or getattr(args, 'enable_torch_compile', False)
            or getattr(args, 'compile_components', None)):
        raise ValueError('Q6 extensions require one eager, standalone inference worker (no parallel/compile/layerwise offload)')
    if args.attention_backend != 'sage_fp8' or args.transformer_quantization != 'fp8_w8a8_static' or config.quantization_scope != 'ffn':
        raise ValueError('Q6 extensions require Sage FP8 and static FP8 FFN')
    if getattr(args, 'sage_fp8_qk_quant_gran', 'per_thread') != 'per_thread' or getattr(args, 'sage_fp8_accum_dtype', 'fp32+fp32') != 'fp32+fp32':
        raise ValueError('Q6 extensions preserve per_thread / fp32+fp32 Sage')
    if options.ffn_epilogue and not options.targeted_fill:
        raise ValueError('FFN epilogue currently requires the audited Q6b allocation path')
    if options.mask_sta:
        if not options.sage_peripheral:
            raise ValueError('mask STA requires MGERASE_Q6_SAGE_PERIPHERAL=1')
        if not options.mask_sta_library or not Path(options.mask_sta_library).is_file():
            raise ValueError('set MGERASE_Q6_MASK_STA_LIBRARY to the built mask STA adapter')
