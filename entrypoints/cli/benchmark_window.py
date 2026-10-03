"""Reproducible single-window NCCL benchmark; model loading and warmup excluded."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

CONFIGS = {'sp1': (1, 1), 'sp2': (1, 2), 'cfg2': (2, 1),
           'sp4': (1, 4), 'cfg2_sp2': (2, 2)}


def probe(path):
    data = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_read_frames',
        '-of', 'json', str(path)], text=True))
    return data['streams'][0]


def prepare(source, target):
    # Lossless H.264 keeps the source codec supported by VideoEncodingProfile.
    # Source YUV samples and color metadata are retained without resampling.
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-i', str(source),
                    '-map', '0:v:0', '-frames:v', '121', '-an',
                    '-c:v', 'libx264', '-crf', '0', '-preset', 'ultrafast',
                    '-threads', '4', str(target)], check=True)
    metadata = probe(target)
    if int(metadata['nb_read_frames']) != 121:
        raise ValueError(f'{source} must contain at least 121 frames')
    if frame_hashes(source) != frame_hashes(target):
        raise ValueError(f'{target} does not preserve the first 121 decoded RGB frames')
    return metadata


def frame_hashes(path):
    text = subprocess.check_output([
        'ffmpeg', '-v', 'error', '-nostdin', '-i', str(path), '-map', '0:v:0', '-an',
        '-frames:v', '121', '-pix_fmt', 'rgb24', '-f', 'framemd5', '-'], text=True)
    return [line.rsplit(',', 1)[1].strip() for line in text.splitlines() if not line.startswith('#')]


def summarize(log):
    text = log.read_text()
    report = json.JSONDecoder().raw_decode(text[text.index('{\n  "tasks"'):])[0]
    rows = []
    for task in report['tasks']:
        timing = task['timing']
        counts = timing['pure_inference_stage_counts']
        denoise_names = [name for name in counts if 'DenoisingStage' in name]
        if sum(counts[name] for name in denoise_names) != 1:
            raise ValueError('benchmark must execute exactly one denoising window')
        if task['runtime_video_metadata']['num_frames'] != 121:
            raise ValueError('benchmark output must contain 121 frames')
        row = {'pure_seconds': timing['pure_inference_seconds'],
               'denoise_seconds': sum(timing['pure_inference_stage_breakdown_ms'][name]
                                      for name in denoise_names) / 1000}
        row['id'] = task.get('id')
        row['e2e_seconds'] = task.get('e2e_seconds_excluding_warmup')
        row['parallel_history'] = task.get('parallel_history', [])
        output = task.get('output_file_path')
        if output and Path(output).is_file():
            row['output_sha256'] = hashlib.sha256(Path(output).read_bytes()).hexdigest()
        runtime = task.get('memory_runtime') or {}
        # Eight phase events per window. Keep only this request, excluding
        # warmup and earlier requests retained in the adapter's bounded history.
        events = runtime.get('component_transfers', [])[-8:]
        vae_events = [event for event in events if event['component'] == 'vae']
        if vae_events:
            row['vae_placement_wall_seconds'] = sum(event['seconds'] for event in vae_events)
            for direction in ('h2d_bytes', 'd2h_bytes'):
                row['vae_' + direction] = sum(event.get('weight_transfers', {}).get(direction, 0)
                                             for event in vae_events)
            row['owner_rss_gib'] = events[-1]['memory'].get('cpu_rss_bytes', 0) / 1024**3
        for name in ('peak_allocated_gib', 'peak_reserved_gib'):
            if name in timing.get('extra', {}):
                row['owner_' + name] = timing['extra'][name]
        rows.append(row)
    diagnostic = any(r.get('diagnostic_profiles') for row in rows
                     for window in row['parallel_history']
                     for r in window.get('dit_parallel', {}).get('rank_reports', []))
    result = {'diagnostic_only': diagnostic, 'samples': rows, **{
        key: {'median': statistics.median(row[key] for row in rows),
              'min': min(row[key] for row in rows), 'max': max(row[key] for row in rows)}
        for key in ('pure_seconds', 'denoise_seconds')}}
    if all(row['e2e_seconds'] is not None for row in rows):
        result['e2e_seconds'] = {'median': statistics.median(row['e2e_seconds'] for row in rows),
                                 'min': min(row['e2e_seconds'] for row in rows),
                                 'max': max(row['e2e_seconds'] for row in rows)}
    for enabled in (False, True):
        selected = [row for row in rows if row['id'] and
                    row['id'].endswith('_text_on' if enabled else '_text_off')]
        if selected:
            result['text_on' if enabled else 'text_off'] = {
                key: {'median': statistics.median(row[key] for row in selected),
                      'min': min(row[key] for row in selected), 'max': max(row[key] for row in selected)}
                for key in ('pure_seconds', 'denoise_seconds', 'e2e_seconds')
                if all(row[key] is not None for row in selected)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='New directory; existing runs are preserved')
    parser.add_argument('--devices', default='1,2,3,6', help='Allocated physical GPU IDs in order')
    parser.add_argument('--configs', default=','.join(CONFIGS))
    parser.add_argument('--packing', choices=('reference', 'packed', 'direct'), default='reference')
    parser.add_argument('--operator-fusion-backend', choices=('disabled', 'auto', 'triton'), default='disabled')
    parser.add_argument('--operator-fusion-ops', default='qk_rmsnorm_rope,rmsnorm_adaln')
    parser.add_argument('--vae-offload-mode', choices=('full', 'split', 'cached'), default='full')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--text-cache-ab', action='store_true',
                        help='Alternate off/on then on/off within one resident session; repeats per mode')
    parser.add_argument('--profile-step', type=int, default=0,
                        help='Diagnostic only: capture this one-based step per window (0 disables)')
    parser.add_argument('--prepare-only', action='store_true', help='Prepare lossless fixtures, tasks and commands without CUDA')
    parser.add_argument('--model-path', type=Path, default=Path('data/model'))
    parser.add_argument('--video', type=Path, default=Path('data/113000356.mp4'))
    parser.add_argument('--mask', type=Path, default=Path('data/113000356_mask.mp4'))
    parser.add_argument('--prompt', default='There is a rooftop terrace overlooking the city at sunset.')
    args = parser.parse_args()
    configs = args.configs.split(',')
    devices = args.devices.split(',')
    if args.repeats < 1 or len(set(configs)) != len(configs) or any(c not in CONFIGS for c in configs):
        parser.error('use unique known configs and repeats >= 1')
    if args.profile_step < 0 or args.profile_step > 40:
        parser.error('profile-step must be 0..40')
    if len(set(devices)) != len(devices) or any(CONFIGS[c][0] * CONFIGS[c][1] > len(devices) for c in configs):
        parser.error('provide enough distinct allocated GPUs for every configuration')
    if not args.prepare_only:
        import torch
        if not torch.cuda.is_available():
            parser.error('CUDA is unavailable; use --prepare-only to prepare without running inference')
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    video, mask = root / 'video_121.mp4', root / 'mask_121.mp4'
    video_meta, mask_meta = prepare(args.video, video), prepare(args.mask, mask)
    # The bundled mask has a rounded frame-rate tag (23.98 vs 24000/1001).
    # Pair by frame index, as the erase pipeline does; never resample frames.
    if any(video_meta[key] != mask_meta[key] for key in ('width', 'height', 'nb_read_frames')):
        raise ValueError('video and mask dimensions and frame count must match')
    jobs = []
    for name in configs:
        cfg, sp = CONFIGS[name]
        tasks = [{'id': f'{name}_{i}', 'video': str(video), 'mask': str(mask),
                  'output': str(root / f'{name}_{i}.mp4'), 'prompt': args.prompt, 'seed': 42}
                  for i in range(args.repeats)]
        if args.text_cache_ab:
            tasks = [dict(task, id=task['id'] + ('_text_on' if enabled else '_text_off'),
                          output=str(root / (task['id'] + ('_text_on.mp4' if enabled else '_text_off.mp4'))),
                          cache_text_projections=enabled)
                     for i, task in enumerate(tasks)
                     for enabled in ((False, True) if i % 2 == 0 else (True, False))]
        task_file = root / f'{name}_tasks.json'
        task_file.write_text(json.dumps(tasks, indent=2) + '\n')
        command = [sys.executable, '-u', '-m', 'entrypoints.cli.erase_eraserdit',
                   '--model-path', str(args.model_path.resolve()), '--task-file', str(task_file),
                   '--dit-parallel-backend', 'nccl', '--cfg-degree', str(cfg),
                   '--sp-degree', str(sp), '--ulysses-degree', str(sp), '--sp-linear-mode', 'sharded',
                   '--attention-backend', 'sdpa', '--transformer-cache-mode', 'off', '--no-cache-text-projections',
                   '--operator-fusion-backend', args.operator_fusion_backend,
                   '--operator-fusion-ops', args.operator_fusion_ops,
                   '--no-dit-layerwise-offload', '--no-dit-cpu-offload',
                   '--text-encoder-cpu-offload', '--vae-cpu-offload', '--vae-low-memory',
                   '--infer-len', '121', '--overlap', '9', '--no-compact-tail-padding',
                   '--num-inference-steps', '50', '--strength', '0.8', '--guidance-scale', '3.0',
                   '--warmup', '--warmup-steps', '2']
        jobs.append({'name': name, 'command': command, 'log': str(root / f'{name}.log'),
                     'env': {'CUDA_VISIBLE_DEVICES': ','.join(devices[:cfg * sp]),
                             'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
                             'MGERASE_NCCL_PACKING': args.packing,
                             'MGERASE_VAE_OFFLOAD_MODE': args.vae_offload_mode,
                             'MGERASE_DIT_PROFILE_DIR': str(root / 'profiles' / name) if args.profile_step else '',
                             'MGERASE_DIT_PROFILE_STEP': str(args.profile_step or 2)}})
    source_files = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'])
    source_paths = [Path(p) for p in source_files.decode().split('\0') if p.endswith('.py') and Path(p).is_file()]
    manifest = {'window_frames': 121, 'repeats': args.repeats, 'packing': args.packing,
                'text_cache_ab': args.text_cache_ab, 'diagnostic_only': bool(args.profile_step),
                'metadata': {'video': video_meta, 'mask': mask_meta}, 'jobs': jobs,
                'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (args.video, args.mask, *source_paths)}}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if args.prepare_only:
        print(f'Prepared {root / "manifest.json"}; no GPU jobs started.', flush=True)
        return
    summary = {}
    for job in jobs:
        print(f'Starting {job["name"]}; log: {job["log"]}', flush=True)
        with open(job['log'], 'w') as log:
            subprocess.run(job['command'], env={**os.environ, **job['env']},
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        summary[job['name']] = summarize(Path(job['log']))
        (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({job['name']: summary[job['name']]}), flush=True)


if __name__ == '__main__':
    main()
