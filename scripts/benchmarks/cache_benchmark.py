#!/usr/bin/env python3
"""Resident-session cache A/B with interleaved repeats and decoded video metrics.

Example: CUDA_VISIBLE_DEVICES=0 python scripts/benchmarks/cache_benchmark.py --model-path MODEL
--video VIDEO --mask MASK --prompt PROMPT --directory results/cache_prediction --repeats 5
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--mask', type=Path, required=True)
    parser.add_argument('--prompt', required=True, help='Match the prompt used by the reference video')
    parser.add_argument('--reference', type=Path, help='Also compare against this external reference video')
    parser.add_argument('--attention-backend', default='sdpa')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--infer-len', type=int, default=121)
    parser.add_argument('--overlap', type=int, default=9)
    parser.add_argument('--tea-threshold', type=float, default=0.005)
    parser.add_argument('--dbc-threshold', type=float, default=0.03)
    parser.add_argument('--back-blocks', type=int, default=0)
    parser.add_argument('--configs', nargs='+', choices=['off', 'text', 'tea_none', 'tea_linear', 'dbc_none', 'dbc_linear', 'dbc_text'],
                        help='Subset of candidates; off is always included')
    parser.add_argument('--resource-policy', choices=['fullgpu', 'dynamic_offload', 'component_offload'], default='fullgpu')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    for path in (args.model_path, args.video, args.mask, *([args.reference] if args.reference else [])):
        if not path.exists():
            parser.error(f'missing input: {path}')
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    configs = {'off': {'transformer_cache_mode': 'off', 'cache_text_projections': False}}
    for mode, prefix in [('teacache', 'tea'), ('cache_dit', 'dbc')]:
        for predictor in ('none', 'linear'):
            configs[f'{prefix}_{predictor}'] = {
                'transformer_cache_mode': mode, 'cache_residual_predictor': predictor,
                'cache_text_projections': False,
                'teacache_threshold': args.tea_threshold,
                'cache_dit_residual_diff_threshold': args.dbc_threshold,
                'cache_dit_back_blocks': args.back_blocks,
            }
    configs['text'] = {'transformer_cache_mode': 'off', 'cache_text_projections': True}
    configs['dbc_text'] = {**configs['dbc_none'], 'cache_text_projections': True}
    if args.configs:
        configs = {name: config for name, config in configs.items() if name == 'off' or name in args.configs}
    def task(name, config):
        return {'id': name, 'output': str(directory / f'{name}.mp4'), **config}
    tasks = [task('warmup', configs['off'])]
    names = list(configs)
    for repeat in range(args.repeats):
        # Rotate execution order; no model-load or cold-start cost in samples.
        order = names[repeat % len(names):] + names[:repeat % len(names)]
        tasks.extend(task(f'{name}_{repeat}', configs[name]) for name in order)
    task_file = directory / 'tasks.json'
    task_file.write_text(json.dumps(tasks, indent=2) + '\n')
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, '-m', 'entrypoints.cli.erase_eraserdit',
               '--model-path', str(args.model_path.resolve()), '--video-input', str(args.video.resolve()),
               '--mask-input', str(args.mask.resolve()), '--task-file', str(task_file),
               '--num-inference-steps', str(args.steps), '--infer-len', str(args.infer_len),
               '--overlap', str(args.overlap), '--resource-policy', args.resource_policy,
               '--attention-backend', args.attention_backend, '--prompt', args.prompt, '--seed', str(args.seed)]
    env = dict(os.environ, HF_HUB_OFFLINE='1', OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '4'))
    (directory / 'command.json').write_text(json.dumps({
        'command': command, 'cuda_visible_devices': env.get('CUDA_VISIBLE_DEVICES'),
        'configs': configs, 'repeats': args.repeats,
    }, indent=2) + '\n')
    with (directory / 'run.log').open('w') as log:
        subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    log = (directory / 'run.log').read_text()
    runs = json.JSONDecoder().raw_decode(log[log.index('{\n  "tasks"'):])[0]
    (directory / 'runs.json').write_text(json.dumps(runs, indent=2) + '\n')
    with (directory / 'quality.log').open('w') as log:
        subprocess.run([sys.executable, str(root / 'scripts/validation/cache_compare.py'),
                        '--directory', str(directory), '--mask', str(args.mask.resolve()),
                        '--baseline', 'off_0', '--names', *[t['id'] for t in tasks if t['id'] != 'warmup']],
                       cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    quality = json.loads((directory / 'quality.json').read_text())['outputs']
    if args.reference:
        with (directory / 'reference_quality.log').open('w') as log:
            subprocess.run([sys.executable, str(root / 'scripts/validation/cache_compare.py'),
                            '--directory', str(directory), '--mask', str(args.mask.resolve()),
                            '--reference', str(args.reference.resolve()), '--report-name', 'reference_quality.json',
                            '--names', *[t['id'] for t in tasks if t['id'] != 'warmup']],
                           cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = {}
    for name in names:
        samples = [r for r in runs['tasks'] if r['id'].rsplit('_', 1)[0] == name]
        times = [r['e2e_seconds_excluding_warmup'] for r in samples]
        summary[name] = {
            'seconds_median': statistics.median(times), 'seconds_min': min(times), 'seconds_max': max(times),
            'repeat_output_identical': len({quality[r['id']]['sha256'] for r in samples}) == 1,
            'quality': quality[samples[0]['id']]['regions'],
        }
    for result in summary.values():
        result['speedup_vs_off'] = summary['off']['seconds_median'] / result['seconds_median']
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
