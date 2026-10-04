"""Real-weight Linear screening; synthetic activations are not video acceptance."""
import argparse
import json
from pathlib import Path
import statistics

import torch
from safetensors import safe_open


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, default=Path('data/model'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', default='128,1280,10240,32640')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    rows_list = [int(v) for v in args.rows.split(',')]
    if args.repeats < 1 or not rows_list or min(rows_list) < 1:
        parser.error('rows and repeats must be positive')
    if args.output.exists():
        parser.error('output already exists; preserve earlier measurements')
    from layers.quantization.eraserdit_int8 import NativeInt8Linear
    from layers.quantization.eraserdit_fp8 import NativeFp8Linear, NativeTensorwiseFp8Linear
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
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
        weight = weights[name, 'weight'].to(device='cuda', dtype=torch.bfloat16)
        bias = weights.get((name, 'bias'))
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None,
                                 device='cuda', dtype=torch.bfloat16).eval()
        linear.weight.copy_(weight)
        if bias is not None:
            linear.bias.copy_(bias)
        modules = dict(bf16=linear, int8=NativeInt8Linear.from_linear(linear),
                       fp8=NativeFp8Linear.from_linear(linear),
                       fp8_tensorwise=NativeTensorwiseFp8Linear.from_linear(linear))
        for rows in rows_list:
            x = torch.randn(rows, weight.shape[1], device='cuda', dtype=torch.bfloat16)
            reference = linear(x).float()
            errors, times = {}, {mode: [] for mode in modules}
            for mode, module in modules.items():
                for _ in range(3):
                    output = module(x)
                delta = output.float() - reference
                errors[mode] = dict(relative_rmse=(delta.square().mean() / reference.square().mean()).sqrt().item(),
                                    max_abs=delta.abs().max().item(), finite=bool(output.isfinite().all()))
            # Alternate order; include activation quantization, GEMM and epilogue.
            for repeat in range(args.repeats):
                order = list(modules) if repeat % 2 == 0 else list(reversed(modules))
                for mode in order:
                    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    torch.cuda.synchronize()
                    start.record()
                    for _ in range(10):
                        output = modules[mode](x)
                    end.record()
                    end.synchronize()
                    times[mode].append(start.elapsed_time(end) / 10)
            row = dict(layer=name, shape=[rows, weight.shape[1], weight.shape[0]],
                       modes={mode: dict(median_ms=statistics.median(times[mode]),
                                         samples_ms=times[mode], **errors[mode]) for mode in modules})
            results.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
        scope='Real checkpoint weights; synthetic normal activations; not end-to-end speed or quality.',
        results=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
