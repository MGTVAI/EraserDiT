"""Two-device composition, cache consensus and persistent offload ownership."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.test_mesh_gpu import GPUParallelTests
from config.eraserdit import EraserDiTPipelineConfig
from config.server_args import ServerArgs
from memory.adapters.layerwise_memory_adapter import LayerwiseMemoryAdapter
from models.adapters.eraserdit.mesh import EraserDiTMeshWindow
from models.adapters.eraserdit.replicas import EraserDiTReplicaPool


@unittest.skipUnless(os.environ.get('ERASERDIT_TEST_TWO_GPU') == '1', 'explicit two-GPU opt-in')
class ComposableTests(unittest.TestCase):
    setUp = GPUParallelTests.setUp
    reference = GPUParallelTests.reference

    def test_compile_offload_cfg_sp_cache_and_reuse(self):
        from layers.block_compile import configure_block_compile
        expected = self.reference(self.values)
        self.model.to('cpu')
        adapter = LayerwiseMemoryAdapter(prefetch_size=1)
        adapter.register(modules={'transformer': self.model}, device='cuda:0', max_weight_usage=2**20)
        configure_block_compile(self.model, mode='default')
        try:
            for sp, cfg, attention_mode in ((1, 2, "ulysses"), (2, 1, "ulysses"), (2, 1, "ring")):
                args = ServerArgs(device='cuda:0', resource_policy='dynamic_offload',
                    enable_torch_compile=True, max_weight_usage=2**20,
                    pipeline_config=EraserDiTPipelineConfig(sp_degree=sp, cfg_degree=cfg))
                plan = dict(sp=sp, cfg=cfg, devices=[torch.device('cuda', i) for i in range(2)],
                            linear_mode='sharded', attention_mode=attention_mode)
                with patch.dict(os.environ, {'MGERASE_TORCH_COMPILE_MODE': 'default'}):
                    pool = EraserDiTReplicaPool(self.model, plan, args)
                replica_id = id(pool.models[1])
                try:
                    for mode in ('off', 'teacache', 'cache_dit'):
                        batch = SimpleNamespace(extra={}, transformer_cache_mode=mode,
                            teacache_threshold=.9, teacache_warmup_steps=1,
                            cache_dit_residual_diff_threshold=.9, cache_dit_warmup_steps=1,
                            cache_dit_front_blocks=1, cache_dit_back_blocks=0)
                        adapter.acquire_component_residency('transformer', reason='test')
                        try:
                            with EraserDiTMeshWindow(self.model, plan, pool=pool,
                                    batch=batch, total_steps=4) as window:
                                for step in range(4):
                                    neg, pos = window.predict(self.values, self.values)
                                    torch.testing.assert_close(pos, neg, atol=0, rtol=0)
                                    torch.testing.assert_close(pos, expected, atol=.04, rtol=.04)
                                self.assertEqual(id(pool.models[1]), replica_id)
                        finally:
                            adapter.release_component_residency('transformer', reason='test')
                        for model in pool.models:
                            self.assertTrue(all(p.device.type == 'cpu' for p in model.parameters()))
                        report = batch.extra['transformer_cache']
                        self.assertEqual(len(report['ranks']), 2)
                        if mode != 'off':
                            key = 'skip_steps' if mode == 'teacache' else 'cached_middle_steps'
                            self.assertGreater(report['ranks'][0]['total'][key], 0)
                    adapter.acquire_component_residency('transformer', reason='failure')
                    try:
                        with self.assertRaises(Exception):
                            with EraserDiTMeshWindow(self.model, plan, pool=pool) as window:
                                with patch.object(pool.models[1], 'forward', side_effect=RuntimeError('injected')):
                                    window.predict(self.values, self.values)
                    finally:
                        adapter.release_component_residency('transformer', reason='failure')
                    self.assertFalse(pool.active)
                    self.assertTrue(all(p.device.type == 'cpu' for p in pool.models[1].parameters()))
                finally:
                    pool.close()
        finally:
            adapter.shutdown()

    def test_cpu_int8_offload_compile_cfg(self):
        from layers.block_compile import configure_block_compile
        from models.dits.eraserdit_quantization import quantize_transformer
        self.model.to('cpu')
        report = quantize_transformer(self.model, execution_device='cuda:0')
        self.assertEqual(report['conversion_device'], 'cpu')
        self.assertTrue(all(t.device.type == 'cpu' for t in self.model.buffers()))
        self.model.to('cuda:0')
        expected = self.reference(self.values)
        self.model.to('cpu')
        adapter = LayerwiseMemoryAdapter()
        adapter.register(modules={'transformer': self.model}, device='cuda:0', max_weight_usage=2**20)
        configure_block_compile(self.model, mode='default')
        args = ServerArgs(device='cuda:0', resource_policy='dynamic_offload', enable_torch_compile=True,
                          transformer_quantization='int8_w8a8_native', max_weight_usage=2**20)
        plan = dict(sp=1, cfg=2, devices=[torch.device('cuda', i) for i in range(2)])
        with patch.dict(os.environ, {'MGERASE_TORCH_COMPILE_MODE': 'default'}):
            pool = EraserDiTReplicaPool(self.model, plan, args)
        try:
            adapter.acquire_component_residency('transformer', reason='test')
            with EraserDiTMeshWindow(self.model, plan, pool=pool) as window:
                for _ in range(3):
                    neg, pos = window.predict(self.values, self.values)
                    torch.testing.assert_close(pos, neg, atol=0, rtol=0)
                    torch.testing.assert_close(pos, expected, atol=.02, rtol=.02)
        finally:
            pool.close()
            adapter.shutdown()

    def test_cancelled_stage_releases_all_ranks(self):
        from nodes.control import CancellationToken, RequestCancelled
        from memory.policies.memory_phase_controller import MemoryPhaseController
        from pipelines.stages.eraserdit_erase.denoising import EraserDiTEraseDenoisingStage
        self.model.to('cpu')
        adapter = LayerwiseMemoryAdapter()
        adapter.register(modules={'transformer': self.model}, device='cuda:0', max_weight_usage=2**20)
        args = ServerArgs(device='cuda:0', resource_policy='dynamic_offload', max_weight_usage=2**20,
                          pipeline_config=EraserDiTPipelineConfig(cfg_degree=2))
        stage = EraserDiTEraseDenoisingStage(self.model, None, args)
        token = CancellationToken('cancelled-mesh')
        token.request()
        v = self.values
        batch = SimpleNamespace(modules={'transformer': self.model, 'scheduler': None,
            'vae': SimpleNamespace(temporal_compression_ratio=8)},
            latents=v['hidden_states'], cond_latents=v['cond_latents'], mask_values=v['mask_values'],
            timesteps=torch.ones(1, device='cuda:0'), prompt_embeds=v['encoder_hidden_states'],
            prompt_attention_mask=v['encoder_attention_mask'], negative_prompt_embeds=v['encoder_hidden_states'],
            negative_attention_mask=v['encoder_attention_mask'], guidance_scale=3., rope_interpolation_scale=None,
            padded_video=torch.empty(1, 3, 1, 3, 3), metrics=None,
            extra={'service_cancellation_token': token,
                   'memory_phase_controller': MemoryPhaseController(adapter, rank=0, device='cuda:0')})
        try:
            with self.assertRaises(RequestCancelled):
                stage.forward(batch, args)
            self.assertIsNone(adapter.active_component_name)
            self.assertFalse(stage._replica_pool.active)
            for model in stage._replica_pool.models:
                self.assertTrue(all(p.device.type == 'cpu' for p in model.parameters()))
        finally:
            stage.close()
            adapter.shutdown()
