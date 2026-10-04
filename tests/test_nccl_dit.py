"""Orthogonal topology and real multiprocess DiT collective correctness."""
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from config.dit_parallel import DiTTopology, resolve_dit_topology
from config.eraserdit import EraserDiTPipelineConfig


def _rank_check(rank, topology, rendezvous, cuda, fsdp, ring_mode, tp_mode, text_cache=False, fusion=False, residual_cache=False):
    from config.server_args import ServerArgs, set_global_server_args
    from distributed.dit_groups import DiTGroups
    from models.dits.eraserdit_transformer import EraserDiTLTXVideoTransformer3DModel
    from models.adapters.eraserdit.nccl_runner import DiTRankRunner
    device = torch.device('cuda', rank) if cuda else torch.device('cpu')
    if cuda:
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group('nccl' if cuda else 'gloo', init_method=rendezvous,
                            world_size=topology.world_size, rank=rank, timeout=timedelta(seconds=45))
    try:
        groups = DiTGroups(topology)
        axes = ('tp', 'ulysses', 'ring', 'sp', 'cfg')
        for axis in axes:
            group, ranks, _ = groups.get(axis)
            if len(ranks) > 1:
                for other in axes:
                    if groups.get(other)[1] == ranks:
                        assert groups.get(other)[0] is group
                if len(ranks) == topology.world_size:
                    assert group is dist.group.WORLD
        if topology.tp > 1:
            from distributed.dit_groups import gather_variable
            group, ranks, trank = groups.get('tp')
            lengths = tuple(range(1, len(ranks) + 1))
            local = torch.full((1, trank + 1), float(trank), device=device)
            actual = gather_variable(local, group, len(ranks), lengths=lengths)
            expected = torch.tensor([[float(i) for i, n in enumerate(lengths) for _ in range(n)]], device=device)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            gathered = gather_variable(local, group, len(ranks), lengths=lengths, dst=ranks[-1])
            if rank == ranks[-1]:
                torch.testing.assert_close(gathered, expected, atol=0, rtol=0)
            else:
                assert gathered is None
        set_global_server_args(ServerArgs(device=str(device), attention_backend='sdpa'))
        torch.manual_seed(123)
        model = EraserDiTLTXVideoTransformer3DModel(in_channels=3, out_channels=4,
            num_attention_heads=32 if fusion else 4, attention_head_dim=64 if fusion else 16,
            cross_attention_dim=2048 if fusion else 64,
            num_layers=2, caption_channels=16).eval().requires_grad_(False)
        if cuda:
            model.to(dtype=torch.bfloat16)
        from copy import deepcopy
        reference = deepcopy(model).to(device)
        if topology.tp > 1:
            from layers.dit_tensor_parallel import shard_linear_weights
            group, ranks, trank = groups.get('tp')
            reference_names = tuple(f'transformer_blocks.{i}.ff.net.2' for i in range(len(model.transformer_blocks))) \
                if tp_mode == 'aligned' and topology.sp > 1 else ()
            shard_linear_weights(model, group, trank, len(ranks),
                'sharded' if tp_mode == 'aligned' else tp_mode,
                replicated_names=('caption_projection.linear_1', 'proj_out') if tp_mode == 'aligned' else (),
                reference_names=reference_names)
            if tp_mode == 'aligned':
                assert isinstance(model.caption_projection.linear_1, torch.nn.Linear)
                assert isinstance(model.proj_out, torch.nn.Linear)
                if topology.sp > 1:
                    assert all(block.ff.net[2].mode == 'reference' for block in model.transformer_blocks)
        model.to(device)
        if fsdp:
            from memory.backends.fsdp_offload import shard_model
            from torch.distributed.device_mesh import init_device_mesh
            mesh = init_device_mesh('cuda', (topology.world_size,))
            shard_model(model, cpu_offload=False, mesh=mesh, fsdp_shard_conditions=[
                lambda name, module: name.startswith('transformer_blocks.') and name.count('.') == 1])
        config = EraserDiTPipelineConfig(sp_linear_mode='sharded', ring_attention_mode=ring_mode,
                                        tp_linear_mode=tp_mode)
        cached = DiTRankRunner(deepcopy(model), groups, config) if text_cache else None
        fused = None
        if fusion:
            from layers.operator_fusion.registry import resolve_operator_fusion_decision
            optimized = deepcopy(model)
            decision = resolve_operator_fusion_decision(SimpleNamespace(
                operator_fusion_backend='triton', operator_fusion_ops=None, sp_degree=topology.sp))
            optimized.operator_fusion_decision = decision
            for block in optimized.transformer_blocks:
                block.attn1.processor.operator_fusion_decision = decision
            fused = DiTRankRunner(optimized, groups, config)
        runner = DiTRankRunner(model, groups, config)
        for frames in ((1, 2, 65) if ring_mode == 'streaming' else (1, 2)):
            # Nine tokens exercises unequal sequence and ring shards.
            torch.manual_seed(42 + frames)
            hidden = torch.randn(1, 1, frames, 3, 3, device=device).to(torch.bfloat16 if cuda else torch.float32)
            common = dict(cond_latents=torch.randn_like(hidden), mask_values=torch.ones_like(hidden),
                          encoder_attention_mask=torch.ones(1, 5, device=device),
                          num_frames=frames, height=3, width=3, return_dict=False)
            positive = dict(common, encoder_hidden_states=torch.randn(1, 5, 16, device=device).to(hidden.dtype))
            negative = dict(common, encoder_hidden_states=positive['encoder_hidden_states'] + .7)
            for step in (1., 2.):
                timestep = torch.tensor([step], device=device)
                packet = dict(static=dict(negative=negative, positive=positive) if step == 1 else None,
                              hidden=hidden, timestep=timestep)
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda):
                    expected = [reference(**v, hidden_states=hidden, timestep=timestep)[0].float()
                                for v in (negative, positive)]
                    actual = runner.predict(packet)
                    if fused is not None:
                        fused_actual = fused.predict(dict(packet, cache_text_projections=True))
                        for a, b in zip(actual, fused_actual):
                            torch.testing.assert_close(a, b, atol=0, rtol=0)
                    if cached is not None:
                        cached_actual = cached.predict(dict(packet, cache_text_projections=True))
                        for a, b in zip(actual, cached_actual):
                            torch.testing.assert_close(a, b, atol=0, rtol=0)
                    owner_actual = runner.predict(packet, owner_only=True)
                    if topology.ulysses > 1:
                        packing = runner.sequence.packing
                        try:
                            for alternative in (('reference', 'packed', 'direct') if cuda else ('reference', 'packed')):
                                if alternative == packing:
                                    continue
                                runner.sequence.packing = alternative
                                alternate = runner.predict(packet, owner_only=True)
                                if rank == 0:
                                    for a, b in zip(owner_actual, alternate):
                                        torch.testing.assert_close(a, b, atol=0, rtol=0)
                        finally:
                            runner.sequence.packing = packing
                if rank == 0:
                    for a, b in zip(owner_actual, actual):
                        torch.testing.assert_close(a, b, atol=0, rtol=0)
                else:
                    assert owner_actual is None
                for a, b in zip(actual, expected):
                    if topology.tp > 1 and topology.sp == 1 and tp_mode == 'reference':
                        torch.testing.assert_close(a, b, atol=0, rtol=0)
                    else:
                        torch.testing.assert_close(a, b, atol=.035 if cuda else 2e-6, rtol=.035 if cuda else 2e-5)
            if cached is not None:
                for stats in cached.report()['text_cache'].values():
                    assert stats['projection_hits'] == 1
                    assert stats['kv_hits'] == len(model.transformer_blocks)
                cached.reset()
                assert not cached.text_caches and cached.static is None
            if fused is not None:
                stats = fused.report()['operator_fusion']
                forwards = 2 * (2 if topology.cfg == 1 else 1)
                assert stats['fused_calls']['qk_rmsnorm_rope'] == 2 * forwards
                assert stats['fused_calls']['rmsnorm_adaln'] == 4 * forwards
                assert not stats['runtime_fallback_reasons']
                fused.reset()
            runner.reset()
        if residual_cache:
            cached_runner = DiTRankRunner(deepcopy(reference), groups, config)
            for mode in ('teacache', 'cache_dit'):
                for force in (True, False):
                    for metric in ('global', 'mask_frame_max'):
                        options = dict(transformer_cache_mode=mode, transformer_cache_force_compute=force,
                            cache_probe_metric=metric, teacache_warmup_steps=1, cache_dit_warmup_steps=1)
                        for index in range(6):
                            packet = dict(hidden=hidden, timestep=timestep,
                                static=dict(negative=negative, positive=positive) if index == 0 else None)
                            expected = runner.predict(packet)
                            actual = cached_runner.predict(dict(packet, residual_cache=options,
                                step=index, total_steps=6, cache_text_projections=True))
                            for a, b in zip(actual, expected):
                                torch.testing.assert_close(a, b, atol=0 if force else .035, rtol=0 if force else .035)
                        report = cached_runner.report()['residual_cache']
                        total = report['total']
                        skipped = total.get('skip_steps', total.get('cached_middle_steps', 0))
                        assert skipped == (0 if force else 2 * (2 if topology.cfg == 1 else 1)), report
                        assert not report['communication']['mismatch_count'] if 'communication' in report else True
                        cached_runner.reset(); runner.reset()
                        assert cached_runner.cache_window is None
        if topology.tp > 1 or fsdp:
            original = sum(p.numel() * p.element_size() for p in reference.parameters())
            assert runner.report()['local_parameter_bytes'] < original
    finally:
        dist.destroy_process_group()


class DistributedDiTTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in real NCCL GPUs')
    def test_residual_cache_same_topology(self):
        for topology in (DiTTopology(cfg=2), DiTTopology(ulysses=2)):
            self.run_ranks(topology, cuda=True, residual_cache=True)
        if torch.cuda.device_count() >= 4:
            self.run_ranks(DiTTopology(cfg=2, ulysses=2), cuda=True, residual_cache=True)

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in real NCCL GPUs')
    def test_fusion_matches_same_topology(self):
        for topology in (DiTTopology(cfg=2), DiTTopology(ulysses=2)):
            self.run_ranks(topology, cuda=True, fusion=True)
        if torch.cuda.device_count() >= 4:
            self.run_ranks(DiTTopology(cfg=2, ulysses=2), cuda=True, fusion=True)

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in real process-pool GPUs')
    def test_fusion_process_pool(self):
        self._check_process_pool(sp_degree=2 if torch.cuda.device_count() >= 4 else 1,
                                 text_cache=True, fusion=True)

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in real NCCL GPUs')
    def test_text_cache_matches_same_topology(self):
        for topology in (DiTTopology(cfg=2), DiTTopology(ulysses=2)):
            self.run_ranks(topology, cuda=True, text_cache=True)
        if torch.cuda.device_count() >= 4:
            self.run_ranks(DiTTopology(cfg=2, ulysses=2), cuda=True, text_cache=True)

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in real process-pool GPUs')
    def test_text_cache_process_pool(self):
        self._check_process_pool(sp_degree=2 if torch.cuda.device_count() >= 4 else 1, text_cache=True)

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in real NCCL GPUs')
    def test_ulysses_packing_matches_reference(self):
        self.run_ranks(DiTTopology(ulysses=2), cuda=True)
        if torch.cuda.device_count() >= 4:
            self.run_ranks(DiTTopology(ulysses=4), cuda=True)
            self.run_ranks(DiTTopology(ulysses=2, cfg=2), cuda=True)

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in real process-pool GPUs')
    def test_process_pool_parent_group_and_peer_failure(self):
        self._check_process_pool(sp_degree=1)

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in real process-pool GPUs')
    def test_process_pool_cfg_sp_guided(self):
        if torch.cuda.device_count() < 4:
            self.skipTest('CFG2 x SP2 requires four GPUs')
        self._check_process_pool(sp_degree=2)

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in real process-pool GPUs')
    def test_residual_cache_process_pool(self):
        self._check_process_pool(sp_degree=2 if torch.cuda.device_count() >= 4 else 1,
                                 text_cache=True, residual_cache=True)

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in real NCCL GPUs')
    def test_cuda_ipc_process_pool(self):
        from unittest.mock import patch
        with patch.dict(os.environ, MGERASE_DIT_BOUNDARY_TRANSPORT='cuda_ipc'):
            self._check_process_pool(sp_degree=2 if torch.cuda.device_count() >= 4 else 1,
                                     text_cache=True, residual_cache=True)

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in real NCCL GPUs')
    def test_cuda_ipc_normal_close(self):
        from unittest.mock import patch
        with patch.dict(os.environ, MGERASE_DIT_BOUNDARY_TRANSPORT='cuda_ipc'):
            self._check_process_pool(sp_degree=1, kill_peer=False)

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in real NCCL GPUs')
    def test_cuda_ipc_owner_rank_failure(self):
        from unittest.mock import patch
        with patch.dict(os.environ, MGERASE_DIT_BOUNDARY_TRANSPORT='cuda_ipc'):
            self._check_process_pool(sp_degree=1, failed_rank=0)

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in real NCCL GPUs')
    def test_head_overlap_process_pool(self):
        if torch.cuda.device_count() < 4:
            self.skipTest('CFG2 x SP2 requires four GPUs')
        from unittest.mock import patch
        with patch.dict(os.environ, MGERASE_NCCL_PACKING='direct', MGERASE_ULYSSES_HEAD_CHUNKS='2'):
            self._check_process_pool(sp_degree=2, text_cache=True, fusion=True, residual_cache=True)

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in real process-pool GPUs')
    def test_budgeted_process_pool_idle_residency(self):
        self._check_process_pool(sp_degree=2 if torch.cuda.device_count() >= 4 else 1,
                                 text_cache=True, residual_cache=True, memory_limit=22)

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in real NCCL GPUs')
    def test_input_output_overlap_process_pool(self):
        if torch.cuda.device_count() < 4:
            self.skipTest('CFG2 x SP2 requires four GPUs')
        from unittest.mock import patch
        with patch.dict(os.environ, MGERASE_NCCL_PACKING='direct',
                        MGERASE_ULYSSES_HEAD_CHUNKS='2', MGERASE_ULYSSES_OUTPUT_OVERLAP='1'):
            self._check_process_pool(sp_degree=2, text_cache=True, fusion=True,
                                     residual_cache=True, memory_limit=22)

    def _check_process_pool(self, sp_degree, text_cache=False, fusion=False, residual_cache=False,
                            kill_peer=True, failed_rank=1, memory_limit=None):
        from copy import deepcopy
        from config.server_args import ServerArgs, set_global_server_args
        from models.dits.eraserdit_transformer import EraserDiTLTXVideoTransformer3DModel
        from pipelines.runtime.dit_executor import DiTProcessPool, DiTProcessWindow
        config = EraserDiTPipelineConfig(dit_parallel_backend='nccl', cfg_degree=2, sp_degree=sp_degree,
                                        sp_linear_mode='sharded')
        args = ServerArgs(device='cuda:0', pipeline_config=config, cuda_memory_limit_gib=memory_limit,
                          operator_fusion_backend='triton' if fusion else 'disabled')
        set_global_server_args(args)
        torch.manual_seed(42)
        model = EraserDiTLTXVideoTransformer3DModel(in_channels=3, out_channels=1,
            num_attention_heads=32 if fusion else 4, attention_head_dim=64 if fusion else 16,
            cross_attention_dim=2048 if fusion else 64,
            num_layers=2, caption_channels=16).to(device='cuda:0', dtype=torch.bfloat16).eval()
        reference = deepcopy(model)
        topology = resolve_dit_topology(config)
        plan = dict(topology=topology, devices=[torch.device('cuda', i) for i in range(topology.world_size)],
                    ring_attention_mode='reference')
        hidden = torch.randn(1, 1, 1, 3, 3, device='cuda:0', dtype=torch.bfloat16)
        values = dict(hidden_states=hidden, cond_latents=torch.randn_like(hidden), mask_values=torch.ones_like(hidden),
                      encoder_hidden_states=torch.randn(1, 5, 16, device=hidden.device, dtype=hidden.dtype),
                      encoder_attention_mask=torch.ones(1, 5, device=hidden.device), timestep=torch.ones(1, device=hidden.device),
                      num_frames=1, height=3, width=3, return_dict=False)
        negative = dict(values, encoder_hidden_states=values['encoder_hidden_states'] + .7)
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group('nccl', init_method=f'file://{directory}/parent', rank=0, world_size=1)
            pool = None
            try:
                pool = DiTProcessPool(model, plan, args)
                for frames in (1, 2):
                    # Change window shape and static conditions on a resident
                    # pool; no prior-window output or packet may be reused.
                    values = dict(values, num_frames=frames)
                    for key in ('hidden_states', 'cond_latents', 'mask_values'):
                        values[key] = values[key][:, :, :1].repeat(1, 1, frames, 1, 1)
                    negative = dict(values, encoder_hidden_states=values['encoder_hidden_states'] + .7)
                    request = SimpleNamespace(cache_text_projections=text_cache,
                        transformer_cache_mode='teacache' if residual_cache else 'off',
                        transformer_cache_force_compute=True, cache_probe_metric='mask_frame_max', extra={})
                    with DiTProcessWindow(model, plan, pool=pool, batch=request, total_steps=6) as window:
                        held = []
                        for scale in (0., 1., 7.5):
                            # Exercise non-default producer/consumer streams.
                            with torch.cuda.stream(torch.cuda.Stream()):
                                hidden = torch.randn_like(values['hidden_states']).transpose(-1, -2)
                                values = dict(values, hidden_states=hidden)
                                negative = dict(negative, hidden_states=hidden)
                                actual = window.predict(negative, values)
                                held.append((actual, tuple(a.clone() for a in actual)))
                            torch.cuda.synchronize()
                            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                                expected = [reference(**v)[0].float() for v in (negative, values)]
                            for a,b in zip(actual, expected):
                                # SP changes GEMM M even without this output
                                # optimization. Dense-vs-SP keeps the existing
                                # small-model tolerance; guided-vs-pair must be
                                # bitwise equal on the very same topology.
                                tolerance = .035 if sp_degree > 1 else 0
                                torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
                            guided = window.predict_guided(negative, values, scale)
                            torch.testing.assert_close(guided, actual[0] + scale * (actual[1] - actual[0]),
                                                       atol=0, rtol=0)
                        # Retained results must not alias reusable IPC buffers.
                        for pair, snapshot in held:
                            for a, b in zip(pair, snapshot):
                                torch.testing.assert_close(a, b, atol=0, rtol=0)
                        report = window.report()
                        self.assertEqual(report['transport'], 'nccl')
                        self.assertEqual(report['boundary_transport'],
                            'cuda_tensor_ipc' if pool.boundary_transport == 'cuda_ipc' else 'cpu_tensor_ipc')
                        self.assertEqual(report['output_assembly'], 'owner_only')
                        self.assertEqual(report['cache_text_projections'], text_cache)
                        for rank_report in report['rank_reports']:
                            if os.environ.get('MGERASE_ULYSSES_OUTPUT_OVERLAP') == '1':
                                self.assertTrue(rank_report['ulysses_output_overlap'])
                            if memory_limit is not None and rank_report['rank'] == 0:
                                residency = rank_report['weight_residency']
                                self.assertEqual(residency['mode'], 'shared_rank_idle_cpu')
                                self.assertTrue(residency['active'])
                                self.assertEqual(residency['acquisitions'], frames)
                                self.assertEqual(residency['d2h_weight_bytes'], 0)
                            if fusion:
                                stats = rank_report['operator_fusion']
                                self.assertEqual(stats['sp_degree'], sp_degree)
                                self.assertEqual(stats['fused_calls']['qk_rmsnorm_rope'], 12)
                                self.assertEqual(stats['fused_calls']['rmsnorm_adaln'], 24)
                            self.assertEqual(bool(rank_report['text_cache']), text_cache)
                            for stats in rank_report['text_cache'].values():
                                self.assertEqual(stats['projection_hits'], 5)
                                self.assertEqual(stats['kv_hits'], 10)
                        self.assertEqual(report['boundary_tensor_bytes']['output'], 9 * expected[0].numel() * 4)
                        self.assertTrue(all(v >= 0 for v in report['boundary_wall_seconds'].values()))
                        self.assertTrue(all(r['window_gpu_seconds']['forward_and_output_gather'] > 0
                                            for r in report['rank_reports']))
                held_before_close = tuple(a.clone() for a in actual)
                self.assertEqual(dist.get_world_size(), 1)
                if kill_peer:
                    pool.processes[failed_rank].terminate()
                    pool.processes[failed_rank].join(timeout=5)
                    with self.assertRaises(RuntimeError):
                        pool.call('report')
                else:
                    pool.close()
                self.assertTrue(pool.closed)
                for a, b in zip(actual, held_before_close):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                self.assertTrue(all(not p.is_alive() for p in pool.processes))
                self.assertEqual(dist.get_world_size(), 1)
            finally:
                if pool:
                    pool.close(force=True)
                dist.destroy_process_group()

    def test_invalid_boundary_transport(self):
        from unittest.mock import patch
        from pipelines.runtime.dit_executor import DiTProcessPool
        with patch.dict(os.environ, MGERASE_DIT_BOUNDARY_TRANSPORT='invalid'):
            with self.assertRaisesRegex(ValueError, 'BOUNDARY_TRANSPORT'):
                DiTProcessPool(torch.nn.Linear(1, 1).bfloat16(), {}, None)

    def test_topology_noncontiguous_sp_groups(self):
        topology = DiTTopology(tp=2, ulysses=2, ring=2, cfg=2)
        self.assertEqual(topology.groups('sp')[0], (0, 2, 4, 6))
        self.assertEqual(topology.groups('ulysses')[0], (0, 2))
        self.assertEqual(topology.groups('ring')[0], (0, 4))
        self.assertEqual(topology.groups('cfg')[0], (0, 8))
        for rank in range(topology.world_size):
            self.assertEqual(topology.rank(topology.coordinates(rank)), rank)

    def test_validation(self):
        from config.server_args import ServerArgs
        config = EraserDiTPipelineConfig(dit_parallel_backend='nccl', sp_degree=4,
                                         ulysses_degree=2, ring_degree=2)
        self.assertEqual(resolve_dit_topology(config).world_size, 4)
        ServerArgs(pipeline_config=config)
        for changed in (dict(sp_degree=2), dict(tp_degree=2, dit_fsdp_shard_degree=8),
                        dict(sp_degree=True), dict(sp_degree=0)):
            with self.assertRaises(ValueError):
                resolve_dit_topology(replace(config, **changed))
        with self.assertRaises(ValueError):
            ServerArgs(pipeline_config=config, enable_torch_compile=True)
        with self.assertRaises(ValueError):
            ServerArgs(pipeline_config=config, compile_components=('vae_decoder',))
        with self.assertRaises(ValueError):
            ServerArgs(pipeline_config=replace(config, sp_linear_mode='unknown'))
        from config.service_contracts.eraserdit import _validate_eraserdit_request
        args = ServerArgs(pipeline_config=config)
        _validate_eraserdit_request({'sampling': {'transformer_cache_mode': 'off'}}, args)
        with self.assertRaises(ValueError):
            _validate_eraserdit_request({'sampling': {'transformer_cache_mode': 'teacache'}}, args)
        from config.dit_parallel import validate_nccl_dit
        for allowed in (EraserDiTPipelineConfig(dit_parallel_backend='nccl', cfg_degree=2),
                        EraserDiTPipelineConfig(dit_parallel_backend='nccl', sp_degree=2)):
            validate_nccl_dit(ServerArgs(pipeline_config=allowed), {'cache_text_projections': True})
        for blocked in (config,
                        EraserDiTPipelineConfig(dit_parallel_backend='nccl', tp_degree=2),
                        EraserDiTPipelineConfig(dit_parallel_backend='nccl', dit_fsdp_shard_degree=2)):
            with self.assertRaisesRegex(ValueError, 'resident CFG/Ulysses'):
                validate_nccl_dit(ServerArgs(pipeline_config=blocked), {'cache_text_projections': True})
        for sp in (1, 2, 4):
            allowed = EraserDiTPipelineConfig(dit_parallel_backend='nccl', sp_degree=sp)
            args = ServerArgs(pipeline_config=allowed, operator_fusion_backend='auto')
            validate_nccl_dit(args)
            from layers.operator_fusion.registry import resolve_operator_fusion_decision
            self.assertEqual(resolve_operator_fusion_decision(args).sp_degree, sp)
        for blocked in (config,
                        EraserDiTPipelineConfig(dit_parallel_backend='nccl', tp_degree=2),
                        EraserDiTPipelineConfig(dit_parallel_backend='nccl', dit_fsdp_shard_degree=2)):
            with self.assertRaisesRegex(ValueError, 'operator fusion requires resident'):
                ServerArgs(pipeline_config=blocked, operator_fusion_backend='auto')
        with self.assertRaisesRegex(ValueError, 'qk_rmsnorm_rope and rmsnorm_adaln only'):
            ServerArgs(pipeline_config=EraserDiTPipelineConfig(dit_parallel_backend='nccl'),
                       operator_fusion_backend='triton', operator_fusion_ops='gated_residual')

    def run_ranks(self, topology, cuda=False, fsdp=False, ring_mode='reference', tp_mode='reference', text_cache=False, fusion=False, residual_cache=False):
        print('checking', topology, 'cuda', cuda, 'fsdp', fsdp, flush=True)
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_rank_check, args=(topology, f'file://{directory}/store', cuda, fsdp, ring_mode, tp_mode, text_cache, fusion, residual_cache),
                     nprocs=topology.world_size, join=True)

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_PROCESSES') == '1', 'opt-in multiprocessing test')
    def test_cpu_collective_matrix(self):
        for topology in (DiTTopology(cfg=2), DiTTopology(ulysses=2), DiTTopology(ring=2),
                         DiTTopology(tp=2), DiTTopology(ulysses=2, ring=2),
                         DiTTopology(ulysses=2, cfg=2), DiTTopology(tp=2, ulysses=2),
                         DiTTopology(tp=2, cfg=2), DiTTopology(replicas=2)):
            self.run_ranks(topology)

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in real NCCL GPUs')
    def test_streaming_flash_profile(self):
        from layers.attention.ring_accumulator import piece
        from utils.determinism import enable_deterministic_mode
        enable_deterministic_mode()
        if torch.cuda.get_device_capability(0) != (8, 0) or not torch.__version__.startswith('2.6.'):
            self.skipTest('bitwise profile is validated on A100 / PyTorch 2.6')
        with torch.no_grad():
            for length in (10200, 32640):
                for seed in (42, 123):
                    torch.manual_seed(seed)
                    q, k, v = [torch.randn(1, length // 2 if i == 0 else length, 32, 64,
                                          device='cuda:0', dtype=torch.bfloat16) for i in range(3)]
                    expected = torch.nn.functional.scaled_dot_product_attention(
                        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
                    split = ((length // 2 + 127) // 128) * 128
                    state = piece(q, k[:, split:], v[:, split:])
                    actual = piece(q, k[:, :split], v[:, :split], state, final=True)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in real NCCL GPUs')
    def test_streaming_ring_uneven_shards(self):
        self.run_ranks(DiTTopology(ring=2), cuda=True, ring_mode='streaming')
        if torch.cuda.device_count() >= 4:
            self.run_ranks(DiTTopology(ulysses=2, ring=2), cuda=True, ring_mode='streaming')

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in real NCCL GPUs')
    def test_tp_aligned_partitions_weights(self):
        self.run_ranks(DiTTopology(tp=2), cuda=True, tp_mode='aligned')
        if torch.cuda.device_count() >= 4:
            self.run_ranks(DiTTopology(tp=2, ulysses=2), cuda=True, tp_mode='aligned')

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in real NCCL GPUs')
    def test_tp_reference_matches_dense(self):
        self.run_ranks(DiTTopology(tp=2), cuda=True)

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in real NCCL GPUs')
    def test_online_ring_uneven_shards(self):
        self.run_ranks(DiTTopology(ring=2), cuda=True, ring_mode='online')
        if torch.cuda.device_count() >= 4:
            self.run_ranks(DiTTopology(ulysses=2, ring=2), cuda=True, ring_mode='online')

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in real NCCL GPUs')
    def test_cuda_collective_matrix(self):
        for topology in (DiTTopology(cfg=2), DiTTopology(ulysses=2), DiTTopology(ring=2), DiTTopology(tp=2)):
            self.run_ranks(topology, cuda=True)
        self.run_ranks(DiTTopology(ulysses=2), cuda=True, fsdp=True)
        self.run_ranks(DiTTopology(replicas=2), cuda=True, fsdp=True)
        if torch.cuda.device_count() >= 4:
            for topology in (DiTTopology(ulysses=2, ring=2), DiTTopology(ulysses=2, cfg=2),
                             DiTTopology(tp=2, ulysses=2)):
                self.run_ranks(topology, cuda=True)


if __name__ == '__main__':
    unittest.main()
