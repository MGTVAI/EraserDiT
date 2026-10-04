"""Isolated FP16 vs E4M3 FP8 vs INT8 GEMM/Linear benchmark on real DiT weights.

GEMM uses prequantized tensorwise FP8 inputs and FP16 output. Full FP8 Linear
uses the project's row/channel quantization and FP32 intermediate, adapted
to FP16 input/output only inside this benchmark. No pipeline dtype changes.
"""
import argparse
import json
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F
import triton
from safetensors import safe_open

from layers.quantization.eraserdit_fp8 import _quant_fp8_rows
from layers.quantization.eraserdit_int8 import _epilogue, _quant_rows
from layers.quantization.gemm import _gemm as _fused_int8_gemm


def tensor_quantize(value):
    value = value.float()
    scale = (value.abs().amax() / 448).clamp_min(1e-12)
    return (value / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale


def measure(functions, repeats, iterations, graph):
    runners, captures = {}, []
    for name, fn in functions.items():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        if graph:
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                output = fn()
            captures.append((capture, output))
            runners[name] = capture.replay
        else:
            runners[name] = fn
    for fn in runners.values():
        for _ in range(10):
            fn()
    samples = {name: [] for name in runners}
    for repeat in range(repeats):
        names = list(runners)
        if repeat % 2:
            names.reverse()
        for name in names:
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            torch.cuda.synchronize()
            start.record()
            for _ in range(iterations):
                runners[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / iterations)
    return {name: dict(median_ms=statistics.median(values), min_ms=min(values),
                       max_ms=max(values), samples_ms=values)
            for name, values in samples.items()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, default=Path('data/model'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', default='1200,32640')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--fp16-reduced-precision-reduction', action=argparse.BooleanOptionalAction,
                        default=True, help='PyTorch default is enabled; disable for a conservative accumulation comparison')
    args = parser.parse_args()
    rows_list = [int(x) for x in args.rows.split(',')]
    if min(rows_list) < 1 or min(args.repeats, args.iterations) < 1:
        parser.error('positive rows, repeats and iterations required')
    if args.output.exists():
        parser.error('output already exists')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        parser.error('FP8 requires SM89 or newer')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = args.fp16_reduced_precision_reduction
    names = ['attn1.to_q', 'ff.net.0.proj', 'ff.net.2']
    weights = {}
    for path in sorted((args.model_path / 'transformer').glob('*.safetensors')):
        with safe_open(path, framework='pt', device='cpu') as handle:
            for name in names:
                for suffix in ('weight', 'bias'):
                    key = f'transformer_blocks.0.{name}.{suffix}'
                    if key in handle.keys():
                        weights[name, suffix] = handle.get_tensor(key)
    results = []
    for name in names:
        weight = weights[name, 'weight'].to(device='cuda', dtype=torch.float16)
        bias = weights.get((name, 'bias'))
        bias = bias.to(device='cuda', dtype=torch.float16) if bias is not None else None
        n, k = weight.shape
        wt, ws = tensor_quantize(weight)
        row_ws = (weight.float().abs().amax(dim=1) / 448).clamp_min(1e-12)
        row_w = (weight.float() / row_ws[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        int_ws = (weight.float().abs().amax(dim=1) / 127).clamp_min(1e-12)
        int_w = (weight.float() / int_ws[:, None]).round().clamp(-127, 127).to(torch.int8)
        unit = torch.ones((), device='cuda', dtype=torch.float32)
        for m in rows_list:
            x = torch.randn(m, k, device='cuda', dtype=torch.float16)
            xt, xs = tensor_quantize(x)
            padded_m = max(32, triton.cdiv(m, 8) * 8)
            int_x = x if padded_m == m else F.pad(x, (0, 0, 0, padded_m - m))
            int_quant = torch.empty_like(int_x, dtype=torch.int8)
            int_scale = torch.empty(padded_m, device='cuda', dtype=torch.float32)
            _quant_rows[(padded_m,)](int_x, int_quant, int_scale, k,
                                   triton.next_power_of_2(k), num_warps=8)

            def int8_output(quant, scale, add_bias, fused=False):
                output = torch.empty((padded_m, n), device='cuda', dtype=torch.float16)
                if fused:
                    # Same production kernel, FP16 destination for this comparison.
                    _fused_int8_gemm[(triton.cdiv(padded_m, 64), triton.cdiv(n, 128))](
                        quant, int_w, scale, int_ws, bias, output, padded_m, n, k,
                        add_bias and bias is not None, 64, 128, 64, num_warps=4, num_stages=3)
                else:
                    accum = torch._int_mm(quant, int_w.t())
                    _epilogue[(triton.cdiv(padded_m * n, 1024),)](
                        accum, scale, int_ws, bias, output, n, padded_m * n,
                        add_bias and bias is not None, 1024)
                return output[:m]

            def full_int8():
                value = x if padded_m == m else F.pad(x, (0, 0, 0, padded_m - m))
                quant = torch.empty_like(value, dtype=torch.int8)
                scale = torch.empty(padded_m, device='cuda', dtype=torch.float32)
                _quant_rows[(padded_m,)](value, quant, scale, k,
                                       triton.next_power_of_2(k), num_warps=8)
                return int8_output(quant, scale, True, fused=m >= 1024 and n >= 2 * k)

            def full_fp8():
                quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
                scale = torch.empty(m, device='cuda', dtype=torch.float32)
                _quant_fp8_rows[(m,)](x, quant, scale, k, triton.next_power_of_2(k), num_warps=8)
                accum = torch._scaled_mm(quant, row_w.t(), scale_a=unit, scale_b=unit,
                                        out_dtype=torch.float32, use_fast_accum=False)
                output = torch.empty((m, n), device='cuda', dtype=torch.float16)
                _epilogue[(triton.cdiv(m * n, 1024),)](
                    accum, scale, row_ws, bias, output, n, m * n, bias is not None, 1024)
                return output

            functions = {
                'fp16_gemm': lambda: torch.mm(x, weight.t()),
                'fp8_gemm': lambda: torch._scaled_mm(xt, wt.t(), scale_a=xs, scale_b=ws,
                                                    out_dtype=torch.float16, use_fast_accum=False),
                'int8_raw_gemm': lambda: torch._int_mm(int_quant, int_w.t()),
                'int8_gemm_fp16out': lambda: int8_output(int_quant, int_scale, False),
                'fp16_linear': lambda: F.linear(x, weight, bias),
                'fp8_linear': full_fp8,
                'int8_linear': full_int8,
            }
            errors = {}
            for candidate, reference in [('fp8_gemm', 'fp16_gemm'), ('fp8_linear', 'fp16_linear'),
                                         ('int8_gemm_fp16out', 'fp16_gemm'), ('int8_linear', 'fp16_linear')]:
                expected = functions[reference]().float()
                actual = functions[candidate]().float()
                finite = bool(actual.isfinite().all() and expected.isfinite().all())
                if not finite:
                    raise RuntimeError(f'nonfinite output: {name}, {m}, {candidate}')
                errors[candidate] = dict(finite=finite, relative_rmse=(
                    (actual - expected).square().mean() / expected.square().mean()).sqrt().item())
                del expected, actual
            # Ensure the FP16 adaptation preserves the native INT8 epilogue.
            if m >= 1024 and n >= 2 * k:
                torch.testing.assert_close(int8_output(int_quant, int_scale, True, fused=True),
                                           int8_output(int_quant, int_scale, True), rtol=0, atol=0)
            row = dict(layer=name, shape_mkn=[m, k, n], int8_padded_rows=padded_m, errors=errors,
                       graph=measure(functions, args.repeats, args.iterations, True),
                       eager=measure(functions, args.repeats, args.iterations, False))
            results.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(device=torch.cuda.get_device_name(),
        torch=torch.__version__, cuda=torch.version.cuda, seed=42,
        repeats=args.repeats, iterations=args.iterations, input_output_dtype='float16',
        fp8_format='e4m3fn', use_fast_accum=False,
        fp16_reduced_precision_reduction=args.fp16_reduced_precision_reduction,
        gemm_scope='Prequantized tensorwise FP8; excludes quantization; no bias; FP16 output.',
        linear_scope='Dynamic per-token FP8 activation + per-channel FP8 weight; FP32 intermediate; FP16 output with bias.',
        int8_scope='Per-token activation/per-channel weight. raw_gemm excludes scaling, outputs INT32; gemm_fp16out includes dequantization, no bias; linear includes dynamic quantization/bias and fused expansion dispatch, FP16 output.',
        input_scope='Real layer-0 checkpoint weights; synthetic normal FP16 activations. No video quality or speed claim.',
        results=results), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
