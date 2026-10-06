"""Paired real-weight NCCL forward screening, separate from video acceptance."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


VARIANTS = {'reference': ('disabled', 'reference'),
            'fusion': ('triton', 'reference'),
            'direct': ('disabled', 'direct'),
            'fusion_direct': ('triton', 'direct'),
            'compiled': ('triton', 'direct'),
            'heads2_serial': ('triton', 'direct'),
            'heads2': ('triton', 'direct'),
            'heads4': ('triton', 'direct'),
            'heads2_output': ('triton', 'direct'),
            'heads4_output': ('triton', 'direct')}
VARIANTS.update({name: ('triton', 'direct') for name in
                 ('aligned', 'aligned_heads2_output', 'aligned_heads4_output')})
VARIANTS.update({f'aligned_heads4_output_{suffix}': ('triton', 'direct')
                 for suffix in ('native_qk', 'native_adaln', 'native_rms')})
VARIANTS.update({f'aligned_heads{chunks}_output_native_rms': ('triton', 'direct')
                 for chunks in (1, 2)})
VARIANTS['aligned_auto_output_native_rms'] = ('triton', 'direct')
VARIANTS.update({f'{name}_compact': VARIANTS[name] for name in tuple(VARIANTS)
                 if name.endswith('native_rms')})


def _rank(rank, options, rendezvous):
    from config.dit_parallel import DiTTopology
    from config.eraserdit import EraserDiTPipelineConfig
    from config.server_args import ServerArgs, set_global_server_args
    from distributed.dit_groups import DiTGroups
    from layers.operator_fusion.registry import resolve_operator_fusion_decision
    from loader.meta_load import load_safetensors_model
    from models.adapters.eraserdit.nccl_runner import DiTRankRunner
    from models.dits.eraserdit_transformer import EraserDiTLTXVideoTransformer3DModel
    from utils.determinism import enable_deterministic_mode
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    enable_deterministic_mode()
    topology = DiTTopology(cfg=options['cfg'], ulysses=options['sp'])
    dist.init_process_group('nccl', init_method=rendezvous, rank=rank,
                            world_size=topology.world_size, timeout=timedelta(seconds=180))
    try:
        config = EraserDiTPipelineConfig(dit_parallel_backend='nccl', cfg_degree=topology.cfg,
                                         sp_degree=topology.sp, sp_linear_mode=options['sp_linear_mode'])
        set_global_server_args(ServerArgs(device=f'cuda:{rank}', pipeline_config=config,
                                          dit_cpu_offload=False, dit_layerwise_offload=False))
        groups = DiTGroups(topology)
        model_path = Path(options['model_path']) / 'transformer'
        model_config = json.loads((model_path / 'config.json').read_text())
        model, _ = load_safetensors_model(lambda: EraserDiTLTXVideoTransformer3DModel.from_config(model_config),
                                         model_path, dtype=torch.bfloat16, device=f'cuda:{rank}')
        runner = DiTRankRunner(model, groups, config)
        results = []
        for frames in options['latent_frames']:
            torch.manual_seed(42)
            hidden = torch.randn(1, 128, frames, 34, 60, device=rank, dtype=torch.bfloat16)
            common = dict(cond_latents=torch.randn_like(hidden),
                mask_values=torch.ones(1, 1, frames, 34, 60, device=rank, dtype=torch.bfloat16),
                encoder_attention_mask=torch.ones(1, 128, device=rank), num_frames=frames,
                height=34, width=60, return_dict=False)
            positive = dict(common, encoder_hidden_states=torch.randn(1, 128, 4096, device=rank, dtype=torch.bfloat16))
            negative = dict(common, encoder_hidden_states=positive['encoder_hidden_states'] + .125)
            initial = dict(static=dict(positive=positive, negative=negative), hidden=hidden,
                           timestep=torch.tensor([500.], device=rank))
            follow = dict(initial, static=None)
            expected = None
            for repeat in range(options['repeats']):
                names = options['variants'] if repeat % 2 == 0 else list(reversed(options['variants']))
                for name in names:
                    backend, packing = VARIANTS[name]
                    variant = name.removesuffix('_compact')
                    fusion_ops = None
                    if '_native_' in variant:
                        fusion_ops = (
                            'qk_rmsnorm_rope_native' if variant.endswith(('native_qk', 'native_rms')) else 'qk_rmsnorm_rope',
                            'rmsnorm_adaln_native' if variant.endswith(('native_adaln', 'native_rms')) else 'rmsnorm_adaln')
                    decision = resolve_operator_fusion_decision(SimpleNamespace(
                        operator_fusion_backend=backend, operator_fusion_ops=fusion_ops, sp_degree=topology.sp))
                    # Benchmark-only mutation on an idle rank; serving keeps
                    # this decision fixed for the lifetime of the process pool.
                    model.operator_fusion_decision = decision
                    for block in model.transformer_blocks:
                        block.attn1.processor.operator_fusion_decision = decision
                    if runner.sequence is not None:
                        runner.sequence.packing = packing
                        runner.sequence.compact_ffn_down = name.endswith('_compact')
                        runner.sequence.head_chunk_policy = 'auto' if '_auto_' in name else 'fixed'
                        chunk_name = name.removeprefix('aligned_')
                        runner.sequence.head_chunks = int(chunk_name[5]) if chunk_name.startswith('heads') else 1
                        runner.sequence.head_overlap = not name.endswith('_serial')
                        runner.sequence.output_overlap = '_output' in name
                    runner.config.sp_linear_mode = 'aligned' if name.startswith('aligned') else options['sp_linear_mode']
                    from layers.block_compile import configure_block_compile, remove_block_compile
                    if name == 'compiled':
                        configure_block_compile(model, mode='default')
                    else:
                        remove_block_compile(model)
                    runner.reset()
                    warm_started = time.perf_counter()
                    warm = runner.predict(initial, owner_only=True)
                    if rank == 0 and expected is None:
                        expected = tuple(t.clone() for t in warm)
                    torch.cuda.synchronize()
                    warm_seconds = time.perf_counter() - warm_started
                    del warm
                    dist.barrier()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    before = time.perf_counter()
                    start.record()
                    output = runner.predict(follow, owner_only=True)
                    end.record()
                    end.synchronize()
                    elapsed = time.perf_counter() - before
                    if rank == 0:
                        for value, reference in zip(output, expected):
                            torch.testing.assert_close(value, reference, atol=0, rtol=0)
                    report = dict(rank=rank, wall_seconds=elapsed, device_ms=start.elapsed_time(end),
                                  details=runner.report())
                    reports = [None] * topology.world_size
                    dist.all_gather_object(reports, report, group=groups.control)
                    if rank == 0:
                        result = dict(variant=name, repeat=repeat, tokens=frames * 34 * 60,
                                      warm_seconds=warm_seconds, wall_seconds=max(r['wall_seconds'] for r in reports), ranks=reports)
                        results.append(result)
                        print(json.dumps({k: v for k, v in result.items() if k != 'ranks'}), flush=True)
                        Path(options['run_dir'], 'samples.json').write_text(json.dumps(results, indent=2) + '\n')
                    del output
            del expected
        if rank == 0:
            summary = {}
            for tokens in sorted({r['tokens'] for r in results}):
                summary[tokens] = {}
                for name in options['variants']:
                    values = [r['wall_seconds'] for r in results if r['tokens'] == tokens and r['variant'] == name]
                    summary[tokens][name] = dict(median=statistics.median(values), min=min(values), max=max(values))
            Path(options['run_dir'], 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, default=Path('data/model'))
    parser.add_argument('--cfg', type=int, choices=(1, 2), default=2)
    parser.add_argument('--sp', type=int, choices=(1, 2, 4), default=2)
    parser.add_argument('--sp-linear-mode', choices=('reference', 'sharded', 'aligned'), default='reference',
                        help='Match the serving GEMM policy; sharded changes BF16 rounding')
    parser.add_argument('--variants', default='reference,fusion,direct,fusion_direct')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--latent-frames', default='16,5')
    args = parser.parse_args()
    if args.repeats < 1 or args.cfg * args.sp > torch.cuda.device_count():
        parser.error('positive repeats and enough visible CUDA devices required')
    variants = args.variants.split(',')
    if not variants or len(set(variants)) != len(variants) or set(variants) - VARIANTS.keys():
        parser.error('variants must be unique supported names: ' + ','.join(VARIANTS))
    if any('heads' in v or v.startswith('aligned') for v in variants) and args.sp not in (2, 4):
        parser.error('head chunk variants require SP2 or SP4')
    frames = [int(v) for v in args.latent_frames.split(',')]
    if not frames or min(frames) < 1:
        parser.error('latent frame counts must be positive')
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    options = dict(run_dir=str(root), model_path=str(args.model_path.resolve()),
                   cfg=args.cfg, sp=args.sp, sp_linear_mode=args.sp_linear_mode,
                   repeats=args.repeats, latent_frames=frames, variants=variants)
    files = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z']).decode().split('\0')
    hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files if p.endswith('.py') and Path(p).is_file()}
    (root / 'manifest.json').write_text(json.dumps(dict(options=options, source_sha256=hashes,
        torch=torch.__version__, cuda=torch.version.cuda, devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        scope='Synthetic fixed input, real weights. Forward screening only; no video quality/speed claim.'), indent=2) + '\n')
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_rank, args=(options, f'file://{directory}/store'), nprocs=args.cfg * args.sp, join=True)


if __name__ == '__main__':
    main()
