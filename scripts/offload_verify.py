#!/usr/bin/env python3
"""Verify real-model offload across repeated requests and an injected failure.

Uses the EraserDiT CLI options; --output-path names a JSON report. Each successful
request writes a sibling MP4. Run with PYTHONPATH=. and HF_HUB_OFFLINE=1.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import torch

from entrypoints.cli.erase_eraserdit import (
    _build_parser, _build_server_args, _task_to_sampling_params,
)
from pipelines.session import EraseSession
from utils.determinism import enable_deterministic_mode
from utils.distributed_runtime import destroy_runtime_distributed


class InjectedFailure(RuntimeError):
    pass


def assert_idle(session):
    pipeline = session.pipeline
    adapter = pipeline._memory_adapter
    snapshot = adapter.snapshot()
    if adapter.active_component_name is not None:
        raise AssertionError('component lease leaked after request')
    state = snapshot.get('flexible_state', {})
    for key in ('flexible_usage_bytes', 'resident_bytes', 'event_queue_size'):
        if state.get(key, 0) != 0:
            raise AssertionError(f'{key} did not return to zero: {state}')
    if state.get('peak_flexible_usage_bytes', 0) > session.server_args.max_weight_usage:
        raise AssertionError('managed weights exceeded configured budget')
    for name in ('transformer', 'text_encoder', 'vae'):
        module = pipeline.get_module(name)
        if any(t.device.type != 'cpu' for t in list(module.parameters()) + list(module.buffers())):
            raise AssertionError(f'{name} still has GPU weights after request')
    text = pipeline.get_module('text_encoder')
    if text.shared.weight is not text.encoder.embed_tokens.weight:
        raise AssertionError('T5 shared embedding weight tie was broken')
    return snapshot


def main():
    parser = _build_parser()
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--inject-failure', action='store_true')
    args = parser.parse_args()
    if not args.output_path or not args.model_path:
        parser.error('--model-path and --output-path are required')
    if args.resource_policy not in ('dynamic_offload', 'component_offload'):
        parser.error('verification requires an offload policy')
    if args.repeat < 2:
        parser.error('--repeat must be at least 2')
    if args.inject_failure and args.resource_policy != 'dynamic_offload':
        parser.error('--inject-failure requires dynamic_offload')
    os.environ.setdefault('MGERASE_FFMPEG_THREADS', 'auto')
    determinism = enable_deterministic_mode()
    report_path = Path(args.output_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    server_args = _build_server_args(args)
    session = None
    report = {'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'determinism': determinism, 'resource_policy': server_args.resolve_resource_policy().as_dict(),
              'runs': [], 'failure_recovered': False, 'passed': False}
    try:
        started = time.perf_counter()
        session = EraseSession(server_args)
        report['load_seconds'] = time.perf_counter() - started
        for index in range(args.repeat):
            if index == 1 and args.inject_failure:
                extent = session.pipeline.get_module('transformer').transformer_blocks[0].flexible_extent
                original = extent._original_forward
                def fail(*values, **kwargs):
                    original(*values, **kwargs)
                    raise InjectedFailure('after first transformer block')
                params = _task_to_sampling_params({'output': str(report_path.with_suffix('.failed.mp4'))}, args)
                with patch.object(extent, '_original_forward', side_effect=fail):
                    try:
                        session.run(params)
                    except InjectedFailure:
                        assert_idle(session)
                        report['failure_recovered'] = True
                    else:
                        raise AssertionError('failure injection was not reached')
            output = report_path.with_suffix(f'.run{index}.mp4')
            params = _task_to_sampling_params({'output': str(output)}, args)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            result = session.run(params)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            snapshot = assert_idle(session)
            report['runs'].append({
                'output': str(output), 'seconds': elapsed,
                'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
                'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
                'memory_runtime': snapshot,
                'phase_events': result.extra.get('runtime_phase_events'),
            })
        if len({run['sha256'] for run in report['runs']}) != 1:
            raise AssertionError('repeated requests produced different output bytes')
        report['passed'] = True
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            destroy_runtime_distributed()
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'report': str(report_path), 'passed': report['passed']}, indent=2))


if __name__ == '__main__':
    main()
