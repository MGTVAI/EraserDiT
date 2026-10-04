"""Measure exact GELU+W8A8 fusion on real FFN weights and synthetic inputs."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from entrypoints.cli.benchmark_fp16_fp8 import measure
from layers.quantization.eraserdit_int8 import NativeInt8Linear
from layers.quantization.eraserdit_fp8 import NativeFp8Linear, NativeTensorwiseFp8Linear, NativeStaticFp8Linear
from layers.quantization.gelu import make_gelu_lut


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, default=Path('data/model'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', default='1200,32640')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--mode', choices=('int8', 'fp8', 'fp8_tensorwise', 'fp8_static'), default='int8')
    parser.add_argument('--compare-baseline', action='store_true',
                        help='also measure BF16 and unfused dynamic tensorwise FP8')
    args = parser.parse_args()
    try:
        rows_list = [int(value) for value in args.rows.split(',')]
    except ValueError:
        parser.error('rows must be comma-separated positive integers')
    if min(*rows_list, args.repeats, args.iterations) < 1:
        parser.error('rows, repeats and iterations must be positive')
    if args.output.exists():
        parser.error('output already exists; preserve earlier measurements')
    if args.compare_baseline and args.mode not in ('fp8_tensorwise', 'fp8_static'):
        parser.error('--compare-baseline requires tensorwise or static FP8')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0):
        parser.error('requires CUDA capability >= 8.0')
    if args.mode.startswith('fp8') and torch.cuda.get_device_capability() < (8, 9):
        parser.error('FP8 requires CUDA capability >= 8.9')
    linear_class = {'int8': NativeInt8Linear, 'fp8': NativeFp8Linear,
                    'fp8_tensorwise': NativeTensorwiseFp8Linear,
                    'fp8_static': NativeStaticFp8Linear}[args.mode]
    torch.set_num_threads(1)
    torch.manual_seed(42)
    weights = {}
    names = ('ff.net.0.proj', 'ff.net.2')
    for path in sorted((args.model_path / 'transformer').glob('*.safetensors')):
        with safe_open(path, framework='pt', device='cpu') as handle:
            for name in names:
                for suffix in ('weight', 'bias'):
                    key = f'transformer_blocks.0.{name}.{suffix}'
                    if key in handle.keys():
                        weights[name, suffix] = handle.get_tensor(key)
    linears, reference_linears = [], []
    for name in names:
        if (name, 'weight') not in weights:
            parser.error(f'missing checkpoint weight: {name}')
        weight = weights[name, 'weight']
        bias = weights.get((name, 'bias'))
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None,
                                 device='cuda', dtype=torch.bfloat16)
        linear.weight.copy_(weight)
        if bias is not None:
            linear.bias.copy_(bias)
        linears.append(linear_class.from_linear(linear))
        if args.compare_baseline:
            reference_linears.append(linear)
    up, down = linears
    weight = down.weight_int8 if args.mode == 'int8' else down.weight_fp8
    fused_down = linear_class(weight, down.weight_scale, down.bias)
    fused_down.gelu_lut = make_gelu_lut('cuda')
    if args.compare_baseline:
        baseline_up = NativeTensorwiseFp8Linear(up.weight_fp8, up.weight_scale, up.bias)
        baseline_down = NativeTensorwiseFp8Linear(down.weight_fp8, down.weight_scale, down.bias)
    results = []
    for rows in rows_list:
        x = torch.randn(rows, up.in_features, device='cuda', dtype=torch.bfloat16)
        functions = {
            'separate': lambda: down(torch.nn.functional.gelu(up(x), approximate='tanh')),
            'fused': lambda: fused_down(up(x)),
        }
        if args.compare_baseline:
            functions['dynamic_baseline'] = lambda: baseline_down(torch.nn.functional.gelu(baseline_up(x), approximate='tanh'))
            functions['bf16'] = lambda: reference_linears[1](torch.nn.functional.gelu(reference_linears[0](x), approximate='tanh'))
        expected, actual = functions['separate'](), functions['fused']()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if not torch.isfinite(actual).all():
            raise RuntimeError('non-finite FFN output')
        row = dict(rows=rows, exact=True,
                   eager=measure(functions, args.repeats, args.iterations, False),
                   graph=measure(functions, args.repeats, args.iterations, True))
        if args.compare_baseline:
            reference = functions['bf16']().float()
            row['errors_vs_bf16'] = {}
            for name, fn in functions.items():
                output = fn().float()
                row['errors_vs_bf16'][name] = dict(finite=bool(output.isfinite().all()),
                    relative_rmse=((output-reference).square().mean()/reference.square().mean()).sqrt().item())
        results.append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        device=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        mode=args.mode,
        scope='Full W8A8 FFN, real block 0 weights, synthetic BF16 inputs; not video performance.',
        results=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
