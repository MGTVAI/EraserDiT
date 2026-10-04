"""Instrument a CLI run to count static FP8 activation clipping.

Usage: python -m entrypoints.cli.audit_static_fp8 --audit-report report.json
       [normal erase_eraserdit arguments]
This diagnostic synchronizes each quantized Linear and is NOT a timing run.
It observes the same post-GELU values as the quantizer without changing them.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys

import torch
import triton
import triton.language as tl

from layers.quantization.eraserdit_fp8 import NativeStaticFp8Linear, _load_activation


@triton.jit
def _statistics(X, P, TOTAL: tl.constexpr, BLOCK: tl.constexpr,
                LUT, HAS_GELU: tl.constexpr, LIMIT: tl.constexpr):
    block = tl.program_id(0)
    idx = block * BLOCK + tl.arange(0, BLOCK)
    valid = idx < TOTAL
    x = _load_activation(X, idx, valid, LUT, HAS_GELU)
    magnitude = tl.abs(x)
    finite = magnitude < float('inf')
    tl.store(P + block * 3, tl.max(tl.where(valid & finite, magnitude, 0.), 0))
    tl.store(P + block * 3 + 1, tl.sum((valid & finite & (magnitude > LIMIT)).to(tl.int32), 0))
    tl.store(P + block * 3 + 2, tl.sum((valid & ~finite).to(tl.int32), 0))


def activation_statistics(value, gelu_lut=None, *, limit=56.):
    x = value.contiguous()
    if not x.numel():
        return dict(elements=0, max_abs=0., clipped=0, nonfinite=0)
    partial = torch.empty((triton.cdiv(x.numel(), 8192), 3), device=x.device, dtype=torch.float32)
    _statistics[(partial.shape[0],)](x, partial, x.numel(), 8192, gelu_lut,
                                   gelu_lut is not None, limit, num_warps=8)
    maximum = partial[:, 0].max().item()
    clipped, nonfinite = partial[:, 1:].to(torch.int64).sum(0).tolist()
    return dict(elements=x.numel(), max_abs=maximum, clipped=clipped, nonfinite=nonfinite)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-report', type=Path, required=True)
    args, cli = parser.parse_known_args()
    if '--enable-torch-compile' in cli:
        parser.error('activation auditing requires torch compile disabled')
    output = args.audit_report.resolve()
    if output.exists():
        parser.error('audit report already exists')
    rows = {}
    original = NativeStaticFp8Linear.forward

    def audited(module, value):
        # Save no activations or module references between calls.
        key = id(module)
        if key not in rows:
            rows[key] = dict(index=len(rows), in_features=module.in_features,
                             out_features=module.out_features, fused_gelu=module.gelu_lut is not None,
                             calls=0, elements=0, clipped=0, nonfinite=0, max_abs=0.)
        sample = activation_statistics(value, module.gelu_lut, limit=448. * module.static_scale)
        row = rows[key]
        row['calls'] += 1
        for field in ('elements', 'clipped', 'nonfinite'):
            row[field] += sample[field]
        row['max_abs'] = max(row['max_abs'], sample['max_abs'])
        return original(module, value)

    NativeStaticFp8Linear.forward = audited
    previous_argv = sys.argv
    failed = False
    try:
        sys.argv = ['erase_eraserdit', *cli]
        runpy.run_module('entrypoints.cli.erase_eraserdit', run_name='__main__')
    except BaseException:
        failed = True
        raise
    finally:
        NativeStaticFp8Linear.forward = original
        sys.argv = previous_argv
        totals = {key: sum(r[key] for r in rows.values()) for key in ('calls','elements','clipped','nonfinite')}
        totals['max_abs'] = max((r['max_abs'] for r in rows.values()), default=0.)
        totals['clipped_fraction'] = totals['clipped'] / totals['elements'] if totals['elements'] else None
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(dict(failed=failed or not rows or bool(totals['nonfinite']), observed=bool(rows),
            scope='all calls including warmup; diagnostic synchronization excludes this run from timing comparisons',
            limit=56., totals=totals, linears=list(rows.values()), command=cli), indent=2)+'\n')
    if not rows:
        raise RuntimeError('no static FP8 Linear calls were observed')
    if totals['nonfinite']:
        raise RuntimeError('nonfinite static FP8 activations observed; see audit report')


if __name__ == '__main__':
    main()
