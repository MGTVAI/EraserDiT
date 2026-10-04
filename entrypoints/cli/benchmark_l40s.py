"""Run reproducible L40S profiles, including process-tree NVML peak memory.

The allocator cap applies per process, including NCCL workers. NVML sums the
owner and workers sharing a GPU. Successful execution is not a quality pass.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time


PROFILES = {
    'bf16': (1, ['--dit-layerwise-offload', '--attention-backend', 'sdpa']),
    'bf16_component': (1, ['--no-dit-layerwise-offload', '--dit-cpu-offload', '--attention-backend', 'sdpa']),
    'bf16_fused': (1, ['--no-dit-layerwise-offload', '--dit-cpu-offload', '--attention-backend', 'sdpa',
                       '--operator-fusion-backend', 'triton',
                       '--operator-fusion-ops', 'qk_rmsnorm_rope,rmsnorm_adaln']),
    'sage_bf16_native_fusion': (1, ['--no-dit-layerwise-offload', '--dit-cpu-offload',
                                    '--attention-backend', 'sage_fp8', '--operator-fusion-backend', 'triton',
                                    '--operator-fusion-ops', 'qk_rmsnorm_rope,rmsnorm_adaln']),
    'sdpa_fp8_dynamic_native_fusion': (1, ['--no-dit-layerwise-offload', '--dit-cpu-offload',
                                           '--attention-backend', 'sdpa', '--transformer-quantization', 'fp8_w8a8_native',
                                           '--quantization-scope', 'ffn', '--operator-fusion-backend', 'triton',
                                           '--operator-fusion-ops', 'qk_rmsnorm_rope,rmsnorm_adaln']),
    'fast': (1, ['--no-dit-layerwise-offload']),
    'fast_offload': (1, ['--dit-layerwise-offload']),
    'fast_component': (1, ['--no-dit-layerwise-offload', '--dit-cpu-offload']),
    'sage_bf16': (1, ['--no-dit-layerwise-offload', '--dit-cpu-offload',
                      '--attention-backend', 'sage_fp8', '--operator-fusion-backend', 'triton',
                      '--operator-fusion-ops', 'qk_rmsnorm_rope_fast,rmsnorm_adaln_fast,gated_residual']),
    'sage_fp8_dynamic': (1, ['--no-dit-layerwise-offload', '--dit-cpu-offload',
                             '--attention-backend', 'sage_fp8', '--transformer-quantization', 'fp8_w8a8_native',
                             '--quantization-scope', 'ffn', '--operator-fusion-backend', 'triton',
                             '--operator-fusion-ops', 'qk_rmsnorm_rope_fast,rmsnorm_adaln_fast,gated_residual']),
    'sdpa_fast_fusion': (1, ['--no-dit-layerwise-offload', '--dit-cpu-offload',
                             '--attention-backend', 'sdpa', '--operator-fusion-backend', 'triton',
                             '--operator-fusion-ops', 'qk_rmsnorm_rope_fast,rmsnorm_adaln_fast,gated_residual']),
    'cfg2': (2, ['--dit-parallel-backend', 'nccl', '--cfg-degree', '2']),
    'sp2': (2, ['--dit-parallel-backend', 'nccl', '--sp-degree', '2']),
    'cfg2_sp2': (4, ['--dit-parallel-backend', 'nccl', '--cfg-degree', '2', '--sp-degree', '2']),
    'cfg2_sp2_reference': (4, ['--dit-parallel-backend', 'nccl', '--cfg-degree', '2',
                               '--sp-degree', '2', '--sp-linear-mode', 'reference']),
    'sp4': (4, ['--dit-parallel-backend', 'nccl', '--sp-degree', '4']),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read_report(path):
    text = Path(path).read_text()
    marker = '{\n  "tasks"'
    if marker not in text:
        raise ValueError(f'CLI report missing: {path}')
    report = json.JSONDecoder().raw_decode(text[text.rindex(marker):])[0]
    if not report['tasks']:
        raise ValueError('empty CLI report')
    return report


def summarize(report):
    rows = []
    for task in report['tasks']:
        timing = task['timing']
        output = task['output_file_path']
        row = dict(id=task['id'], output=output, output_sha256=sha256(output),
                   metadata=task['runtime_video_metadata'],
                   request_seconds=task['e2e_seconds_excluding_warmup'],
                   warmup_seconds=float(task.get('warmup', {}).get('duration_seconds') or 0),
                   pure_seconds=timing['pure_inference_seconds'],
                   denoise_seconds=sum(v for k, v in timing['pure_inference_stage_breakdown_ms'].items()
                                       if 'DenoisingStage' in k) / 1000,
                   owner_peak_allocated_gib=timing['extra']['peak_allocated_gib'],
                   owner_peak_reserved_gib=timing['extra']['peak_reserved_gib'])
        rows.append(row)
    return dict(samples=rows, load_seconds=report['tasks'][0]['timing']['extra']['load_seconds'],
                statistics={key: dict(median=statistics.median(r[key] for r in rows),
                                      min=min(r[key] for r in rows), max=max(r[key] for r in rows))
                            for key in ('request_seconds', 'pure_seconds', 'denoise_seconds')})


def monitored_run(command, env, devices, directory, *, interval=.1, cwd=None):
    import psutil
    import pynvml as nv
    nv.nvmlInit()
    process = None
    started = time.monotonic()
    measurements = dict(sample_interval_seconds=interval, samples=0,
                        scope='process tree, from launch through exit, including load and warmup',
                        cpu_peak_rss_sum_gib=0., devices={}, sampling_errors=[])
    try:
        handles = {index: nv.nvmlDeviceGetHandleByIndex(int(index)) for index in devices}
        for index, handle in handles.items():
            existing = nv.nvmlDeviceGetComputeRunningProcesses(handle)
            if existing:
                raise RuntimeError(f'GPU {index} already has compute processes: {[p.pid for p in existing]}')
            measurements['devices'][index] = dict(name=nv.nvmlDeviceGetName(handle),
                uuid=nv.nvmlDeviceGetUUID(handle), task_peak_gib=0., board_peak_gib=0.,
                initial_board_gib=nv.nvmlDeviceGetMemoryInfo(handle).used / 1024**3,
                observed_pids=[], external_pids=[], per_process_peak_gib={})
        with (directory / 'run.log').open('x') as log, (directory / 'memory.jsonl').open('x') as samples:
            process = subprocess.Popen(command, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
            parent = psutil.Process(process.pid)
            known = {process.pid}
            while True:
                rss = 0
                try:
                    children = [parent, *parent.children(recursive=True)]
                    known.update(p.pid for p in children)
                    for child in children:
                        try:
                            rss += child.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass
                sample = dict(seconds=time.monotonic() - started, cpu_rss_sum_gib=rss / 1024**3, devices={})
                measurements['cpu_peak_rss_sum_gib'] = max(measurements['cpu_peak_rss_sum_gib'], rss / 1024**3)
                for index, handle in handles.items():
                    row = measurements['devices'][index]
                    try:
                        running = nv.nvmlDeviceGetComputeRunningProcesses(handle)
                        # A worker can launch between the descendant scan and
                        # NVML query. Check ancestry before calling it external.
                        for item in running:
                            if item.pid not in known:
                                try:
                                    if any(p.pid == process.pid for p in psutil.Process(item.pid).parents()):
                                        known.add(item.pid)
                                except psutil.NoSuchProcess:
                                    pass
                        used = sum(p.usedGpuMemory for p in running if p.pid in known) / 1024**3
                        board = nv.nvmlDeviceGetMemoryInfo(handle).used / 1024**3
                        row['task_peak_gib'] = max(row['task_peak_gib'], used)
                        row['board_peak_gib'] = max(row['board_peak_gib'], board)
                        row['observed_pids'] = sorted(set(row['observed_pids']) | {p.pid for p in running if p.pid in known})
                        row['external_pids'] = sorted(set(row['external_pids']) | {p.pid for p in running if p.pid not in known})
                        for item in running:
                            if item.pid in known:
                                key = str(item.pid)
                                row['per_process_peak_gib'][key] = max(row['per_process_peak_gib'].get(key, 0), item.usedGpuMemory / 1024**3)
                        sample['devices'][index] = dict(task_gib=used, board_gib=board)
                    except nv.NVMLError as error:
                        measurements['sampling_errors'].append(str(error))
                samples.write(json.dumps(sample) + '\n')
                measurements['samples'] += 1
                if process.poll() is not None:
                    break
                time.sleep(interval)
            measurements['returncode'] = process.returncode
        return measurements
    finally:
        if process is not None and process.poll() is None:
            # Interrupting the benchmark must not orphan inference workers.
            import signal
            children = psutil.Process(process.pid).children(recursive=True)
            for child in reversed(children):
                try:
                    child.send_signal(signal.SIGTERM)
                except psutil.NoSuchProcess:
                    pass
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            _, alive = psutil.wait_procs(children, timeout=5)
            for child in alive:
                child.kill()
        measurements['process_seconds'] = time.monotonic() - started
        (directory / 'memory.json').write_text(json.dumps(measurements, indent=2) + '\n')
        nv.nvmlShutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--profile', choices=PROFILES, required=True)
    parser.add_argument('--devices', default='0')
    parser.add_argument('--video', type=Path, default=Path('data/113000356.mp4'))
    parser.add_argument('--mask', type=Path, default=Path('data/113000356_mask.mp4'))
    parser.add_argument('--model-path', type=Path, default=Path('data/model'))
    parser.add_argument('--prompt', default='There is a rooftop terrace overlooking the city at sunset.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--dp-degree', type=int, choices=(1, 2, 4), default=1,
                        help='Disjoint resident workers; repeats are total submitted requests')
    parser.add_argument('--cases', type=Path, help='JSON array of named request overrides; alternate case order each repeat')
    parser.add_argument('--allocator-gib', type=float, default=22.)
    parser.add_argument('--memory-target-gib', type=float, default=24.)
    parser.add_argument('--sage-source', type=Path, help='Optional isolated compatible SageAttention build')
    parser.add_argument('--source-root', type=Path, help='Reuse a previous run/source snapshot for an unchanged experiment batch')
    parser.add_argument('--cache', choices=('off', 'teacache', 'cache_dit'), default='off')
    parser.add_argument('--cache-threshold', type=float, default=.3)
    parser.add_argument('--cache-back-blocks', type=int, default=0)
    parser.add_argument('--cache-max-skip', type=int, default=1)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    devices = args.devices.split(',')
    degree, options = PROFILES[args.profile]
    total_devices = degree * args.dp_degree
    if args.repeats < 1 or len(devices) != total_devices or len(set(devices)) != total_devices or not all(d.isdecimal() for d in devices):
        parser.error('use repeats >= 1 and exactly the required number of distinct physical GPU indices')
    if not 0 < args.allocator_gib <= args.memory_target_gib:
        parser.error('require 0 < allocator-gib <= memory-target-gib')
    for source in (args.video, args.mask):
        if not source.is_file():
            parser.error(f'missing input: {source}')
    cases = json.loads(args.cases.read_text()) if args.cases else [{'id': args.profile}]
    allowed = {'id', 'video', 'mask', 'prompt', 'seed', 'num_inference_steps', 'strength', 'guidance_scale',
               'infer_len', 'overlap', 'compact_tail_padding', 'transformer_cache_mode', 'cache_text_projections',
               'cache_probe_metric', 'teacache_threshold', 'cache_dit_residual_diff_threshold',
               'cache_dit_front_blocks', 'cache_dit_back_blocks', 'max_teacache_consecutive_skip',
               'cache_dit_max_consecutive_cached_steps', 'cache_end_guard_steps'}
    if not isinstance(cases, list) or not cases or any(
            not isinstance(c, dict) or not isinstance(c.get('id'), str)
            or not re.fullmatch(r'[A-Za-z0-9_-]+', c['id']) or set(c)-allowed for c in cases):
        parser.error('cases must be nonempty named request overrides with safe unique IDs')
    if len({c['id'] for c in cases}) != len(cases):
        parser.error('duplicate case IDs')
    if args.repeats * len(cases) < args.dp_degree:
        parser.error('provide at least one request per DP worker')
    for case in cases:
        for key in ('video', 'mask'):
            path = Path(case.get(key, getattr(args, key))).resolve()
            if not path.is_file():
                parser.error(f'missing {key}: {path}')
            case[key] = str(path)
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    tasks, groups = [], {}
    for i in range(args.repeats):
        for case in (cases if i % 2 == 0 else list(reversed(cases))):
            task_id = f'{case["id"]}_r{i}'
            output = f'{task_id}.mp4' if args.cases else f'output_{i}.mp4'
            tasks.append({'prompt': args.prompt, 'seed': args.seed, **case,
                          'id': task_id, 'output': str(root/output)})
            groups[task_id] = case['id']
    (root / 'tasks.json').write_text(json.dumps(tasks, indent=2) + '\n')
    cli = ['--model-path', str(args.model_path.resolve()), '--task-file', str(root / 'tasks.json'),
           '--no-dit-cpu-offload', '--text-encoder-cpu-offload', '--vae-cpu-offload', '--vae-low-memory',
           '--runtime-mode', 'windowed_streaming', '--streaming-cache-dtype', 'uint8',
           '--infer-len', '121', '--overlap', '9', '--no-compact-tail-padding',
           '--num-inference-steps', '50', '--strength', '.8', '--guidance-scale', '3',
           '--no-cache-text-projections', '--warmup', '--warmup-steps', '2',
           '--transformer-cache-mode', args.cache, '--cache-probe-metric', 'mask_frame_max',
           '--teacache-threshold', str(args.cache_threshold),
           '--cache-dit-residual-diff-threshold', str(args.cache_threshold),
           '--cache-dit-back-blocks', str(args.cache_back_blocks),
           '--max-teacache-consecutive-skip', str(args.cache_max_skip),
           '--cache-dit-max-consecutive-cached-steps', str(args.cache_max_skip)]
    if args.profile.startswith('fast'):
        cli += ['--attention-backend', 'sage_fp8', '--transformer-quantization', 'fp8_w8a8_static',
                '--quantization-scope', 'ffn', '--operator-fusion-backend', 'triton',
                '--operator-fusion-ops', 'qk_rmsnorm_rope_fast,rmsnorm_adaln_fast,gated_residual']
    elif degree > 1:
        cli += ['--no-dit-layerwise-offload', '--attention-backend', 'sdpa', '--sp-linear-mode', 'sharded',
                '--operator-fusion-backend', 'triton', '--operator-fusion-ops', 'qk_rmsnorm_rope,rmsnorm_adaln']
    if args.compile:
        cli += ['--enable-torch-compile', '--torch-compile-scope', 'ffn']
    cli += options
    cli += ['--cuda-memory-limit-gib', str(args.allocator_gib)]
    entrypoint = 'entrypoints.cli.erase_eraserdit'
    if args.dp_degree > 1:
        entrypoint = 'entrypoints.cli.erase_parallel'
        cli += ['--dp-degree', str(args.dp_degree), '--parallel-run-dir', str(root/'dp')]
    command = [sys.executable, '-u', '-m', entrypoint, *cli]
    source_root = args.source_root.resolve() if args.source_root else Path.cwd()
    frozen = root / 'source'
    packages = ('cache', 'config', 'distributed', 'entrypoints', 'layers', 'loader',
                'memory', 'models', 'nodes', 'parallel', 'pipelines', 'utils')
    paths = [p for package in packages for p in (source_root/package).rglob('*.py')]
    paths += list(source_root.glob('*.py'))
    if not (source_root/'entrypoints/cli/erase_eraserdit.py').is_file():
        parser.error('source-root is not an EraserDiT source tree')
    source = {}
    for path in sorted(paths):
        relative = path.relative_to(source_root)
        target = frozen/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source[str(relative)] = sha256(target)
    pythonpath = [str(frozen)]
    if args.sage_source:
        pythonpath.append(str(args.sage_source.resolve()))
    overrides = dict(CUDA_VISIBLE_DEVICES=args.devices, HF_HUB_OFFLINE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                     PYTHONPATH=os.pathsep.join(pythonpath), PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                     MGERASE_NCCL_PACKING='direct', MGERASE_POSTPROCESS_CHUNKED_FP32='1',
                     MGERASE_VAE_INPLACE_ACTIVATIONS='1', MGERASE_VAE_OFFLOAD_MODE='cached',
                     MGERASE_FFMPEG_THREADS='auto')
    # Preserve and record benchmark-relevant optional transport switches.
    # Do not copy unrelated environment variables (for example service secrets).
    for name, default in (('MGERASE_ULYSSES_HEAD_CHUNKS', '1'),
                          ('MGERASE_ULYSSES_OUTPUT_OVERLAP', '0'),
                          ('MGERASE_DIT_BOUNDARY_TRANSPORT', 'cpu')):
        overrides[name] = os.environ.get(name, default)
    manifest = dict(profile=args.profile, command=command, env=overrides, cases=cases, repeats=args.repeats,
                    dp_degree=args.dp_degree,
                    cwd=str(frozen), source_root=str(source_root),
                    allocator_limit_scope='per process execution device; aggregate physical usage measured separately',
                    memory_target_gib=args.memory_target_gib,
                    inputs={p: sha256(p) for case in cases for p in (case['video'], case['mask'])},
                    source_sha256=source, git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()})
    if not args.prepare_only:
        manifest['nvidia_smi_start'] = subprocess.check_output(['nvidia-smi'], text=True)
        manifest['nvidia_smi_topology'] = subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if args.prepare_only:
        print(f'Prepared {root}', flush=True)
        return
    print(f'Starting {args.profile}; log: {root / "run.log"}', flush=True)
    memory = monitored_run(command, {**os.environ, **overrides}, devices, root, cwd=frozen)
    changed = [p for p, digest in source.items() if not (frozen/p).is_file() or sha256(frozen/p) != digest]
    (root / 'source_changes_during_run.json').write_text(json.dumps(changed, indent=2) + '\n')
    if memory['returncode']:
        raise RuntimeError(f'CLI failed with code {memory["returncode"]}; see {root / "run.log"}')
    worker_reports = []
    if args.dp_degree > 1:
        dispatcher = json.loads((root/'dp/report.json').read_text())
        if not dispatcher['passed']:
            raise RuntimeError('DP dispatcher did not pass')
        worker_reports = [read_report(root/'dp'/f'worker{i}.log') for i in range(args.dp_degree)]
        by_id = {t['id']: t for worker in worker_reports for t in worker['tasks']}
        report = {'tasks': [by_id[t['id']] for t in tasks]}
    else:
        report = read_report(root / 'run.log')
    if len(report['tasks']) != len(tasks):
        raise ValueError('incomplete CLI report')
    (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    result = summarize(report)
    if worker_reports:
        result.pop('load_seconds')
        result['load_seconds_by_worker'] = [w['tasks'][0]['timing']['extra']['load_seconds'] for w in worker_reports]
        request_sums = [sum(t['e2e_seconds_excluding_warmup'] for t in w['tasks']) for w in worker_reports]
        result['dp'] = dict(degree=args.dp_degree, worker_request_seconds=request_sums,
            cold_requests_per_minute=len(tasks)*60/memory['process_seconds'],
            estimated_warm_requests_per_minute=len(tasks)*60/max(request_sums),
            warm_rate_scope='estimate from longest worker request sum; excludes loading, warmup, scheduling and process exit')
    if len(cases) > 1:
        # Different requests are not interchangeable timing samples.
        result.pop('statistics')
        result['cases'] = {case['id']: summarize({'tasks': [t for t in report['tasks'] if groups[t['id']] == case['id']]})
                           for case in cases}
    result['memory'] = memory
    result['sampled_memory_pass'] = (not memory['sampling_errors'] and all(
        d['observed_pids'] and not d['external_pids'] and d['task_peak_gib'] <= args.memory_target_gib
        for d in memory['devices'].values()))
    result['quality_status'] = 'not evaluated; compare outputs separately'
    result['source_changes_during_run'] = changed
    (root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
