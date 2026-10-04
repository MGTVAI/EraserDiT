"""Compare BF16 SDPA and SM89 quantized Attention, including dynamic quantization.

Synthetic Q/K/V only: video timing and quality require the erase CLI separately.
"""
import argparse
import json
from pathlib import Path

import torch

from entrypoints.cli.benchmark_fp16_fp8 import measure
from layers.attention.backends.attention_backend import AttentionMetadata
from layers.attention.backends.sage_fp8 import SageFP8AttentionImpl
from layers.attention.backends.sdpa import SDPAImpl


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sage-fp8-accum-dtype', choices=('fp32+fp32', 'fp32+fp16'), default='fp32+fp32')
    parser.add_argument('--sage-fp8-qk-quant-gran', choices=('per_thread', 'per_warp'), default='per_thread')
    parser.add_argument('--tokens', default='1200,32640')
    parser.add_argument('--heads', type=int, default=32)
    parser.add_argument('--head-dim', type=int, choices=(64, 128), default=64)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=20)
    args = parser.parse_args()
    tokens = [int(v) for v in args.tokens.split(',')]
    if min(*tokens, args.heads, args.repeats, args.iterations) < 1:
        parser.error('all sizes and repeat counts must be positive')
    if args.output.exists():
        parser.error('output already exists; preserve previous measurements')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        parser.error('this Sage FP8 backend requires SM89')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    results = []
    for count in tokens:
        q, k, v = [torch.randn(1, count, args.heads, args.head_dim,
                              device='cuda', dtype=torch.bfloat16) for _ in range(3)]
        scale = args.head_dim ** -.5
        metadata = AttentionMetadata()
        sdpa = SDPAImpl(args.heads, args.head_dim, False, scale)
        sage = SageFP8AttentionImpl(args.heads, args.head_dim, scale,
                                   pv_accum_dtype=args.sage_fp8_accum_dtype,
                                   qk_quant_gran=args.sage_fp8_qk_quant_gran)
        functions = dict(sdpa=lambda: sdpa.forward(q, k, v, metadata),
                         sage_fp8=lambda: sage.forward(q, k, v, metadata))
        reference = functions['sdpa']().float()
        actual = functions['sage_fp8']().float()
        if not actual.isfinite().all():
            raise RuntimeError('quantized Attention produced nonfinite values')
        row = dict(shape=list(q.shape),
                   relative_rmse=((actual-reference).square().mean() /
                                  reference.square().mean()).sqrt().item(),
                   max_abs=(actual-reference).abs().max().item(),
                   timing=measure(functions, args.repeats, args.iterations, False),
                   backend=sage.report())
        results.append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(device=torch.cuda.get_device_name(),
        torch=torch.__version__, dtype='bfloat16', cuda_graph=False,
        scope='Synthetic Q/K/V; includes quantization and layout; not video acceptance.',
        results=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
