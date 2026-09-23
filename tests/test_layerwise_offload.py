"""CUDA correctness and residency contracts for the SGLang-style backend."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from config.server_args import ServerArgs
from memory.backends.layerwise_offload import LayerwiseOffloadManager
from memory.adapters.layerwise_memory_adapter import LayerwiseMemoryAdapter
from memory.policies.memory_phase_controller import MemoryPhase, MemoryPhaseController
from memory.policies.component_offload import offload_component


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(16)
        self.scale_shift_table = nn.Parameter(torch.zeros(16))
        self.linear = nn.Linear(16, 16)
        self.register_buffer('strided', torch.randn(16, 16).t())
        self.register_buffer('scale', torch.tensor(0.01, dtype=torch.float64))
        self.fail = False

    def forward(self, x):
        if self.fail:
            raise RuntimeError('intentional block failure')
        return self.linear(self.norm1(x)) + x @ self.strided * self.scale.float() + self.scale_shift_table


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([Block() for _ in range(4)])

    def forward(self, x, order=(0, 1, 2, 3)):
        for i in order:
            x = self.transformer_blocks[i](x)
        return x


class ConfigurationTests(unittest.TestCase):
    def test_prefetch_validation(self):
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                ServerArgs(dit_offload_prefetch_size=value)
        self.assertEqual(ServerArgs(dit_offload_prefetch_size=0).dit_offload_prefetch_size, 0)

    def test_fsdp_conflict_is_rejected_before_loading(self):
        from pipelines.eraserdit_erase_pipeline import EraserDiTErasePipeline
        from pipelines.base import ComposedPipelineBase
        pipeline = object.__new__(EraserDiTErasePipeline)
        args = ServerArgs(resource_policy='dynamic_offload', device='cuda:0', use_fsdp_inference=True)
        with patch('torch.cuda.is_available', return_value=True), patch.object(ComposedPipelineBase, 'load_modules') as loader:
            with self.assertRaisesRegex(ValueError, 'FSDP'):
                pipeline.load_modules(args)
            loader.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
class LayerwiseTests(unittest.TestCase):
    def test_outputs_layout_budget_and_jump_recovery(self):
        torch.manual_seed(7)
        for depth in (0, 1, 4):
            for budget_layers in (1, 2, 4):
                model = Toy().eval()
                reference = copy.deepcopy(model).cuda()
                size = sum(t.numel()*t.element_size() for t in
                           list(model.transformer_blocks[0].parameters()) + list(model.transformer_blocks[0].buffers()))
                manager = LayerwiseOffloadManager(model, device='cuda:0',
                            max_weight_usage=size*budget_layers, prefetch_size=depth)
                self.assertTrue(all(t.device.type == 'cpu' for t in model.parameters()))
                self.assertEqual(manager.h2d_count, 0)
                x = torch.randn(2, 16, device='cuda')
                with torch.no_grad():
                    for order in ((0,1,2,3), (0,2,2,3), (3,1,0), (0,1,2,3)):
                        manager.begin()
                        torch.testing.assert_close(model(x, order), reference(x, order), rtol=0, atol=0)
                        manager.end()
                        self.assertEqual(manager.used, 0)
                        self.assertLessEqual(manager.peak, size*budget_layers)
                        self.assertEqual(model.transformer_blocks[0].strided.stride(), (1,16))
                manager.close()
                self.assertTrue(all(not p.is_pinned() for p in model.parameters()))
                self.assertEqual(manager.snapshot()['pinned_cpu_bytes'], 0)
                with torch.no_grad():
                    torch.testing.assert_close(model(x.cpu()), reference.cpu()(x.cpu()), rtol=0, atol=0)
                self.assertFalse(model.transformer_blocks[0]._forward_pre_hooks)

    def test_restored_packed_model_can_be_registered_again(self):
        model = Toy().eval()
        reference = copy.deepcopy(model).cuda()
        x = torch.randn(2,16,device='cuda')
        with torch.no_grad():
            for _ in range(2):
                manager = LayerwiseOffloadManager(model, device='cuda:0', max_weight_usage=10**6)
                manager.begin()
                torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)
                manager.end()
                manager.close()

    def test_range_does_not_prefetch_skipped_layers(self):
        model = Toy().eval()
        manager = LayerwiseOffloadManager(model, device='cuda:0', max_weight_usage=10**6, prefetch_size=4)
        x = torch.randn(2,16,device='cuda')
        with torch.no_grad():
            manager.begin()
            with manager.execution_range(0,1):
                model(x, (0,))
            self.assertEqual(manager.h2d_count, 1)
            with manager.execution_range(3,4):
                model(x, (3,))
            self.assertEqual(manager.h2d_count, 2)
            manager.end()
        manager.close()

    def test_failure_cleanup_and_retry(self):
        model = Toy().eval()
        reference = copy.deepcopy(model).cuda()
        manager = LayerwiseOffloadManager(model, device='cuda:0', max_weight_usage=10**6)
        model.transformer_blocks[1].fail = True
        x = torch.randn(2,16,device='cuda')
        with torch.no_grad():
            manager.begin()
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                model(x)
            manager.end()
            self.assertEqual(manager.used, 0)
            self.assertTrue(all(p.device.type=='cpu' for p in model.parameters()))
            model.transformer_blocks[1].fail = False
            manager.begin()
            torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)
            manager.end()
        manager.close()

    def test_partial_h2d_failure_rolls_back_and_retries(self):
        model = Toy().eval()
        reference = copy.deepcopy(model).cuda()
        manager = LayerwiseOffloadManager(model, device='cuda:0', max_weight_usage=10**6)
        original_copy = torch.Tensor.copy_
        transfers = []
        def fail_second(target, source, *args, **kwargs):
            if target.device.type == 'cuda' and source.device.type == 'cpu':
                transfers.append(1)
                if len(transfers) == 2:
                    raise RuntimeError('injected H2D failure')
            return original_copy(target, source, *args, **kwargs)
        x = torch.randn(2,16,device='cuda')
        with torch.no_grad():
            manager.begin()
            with patch.object(torch.Tensor, 'copy_', new=fail_second):
                with self.assertRaisesRegex(RuntimeError, 'H2D failure'):
                    model(x)
            manager.end()
            self.assertEqual(manager.used, 0)
            self.assertTrue(all(p.device.type == 'cpu' for p in model.parameters()))
            manager.begin()
            torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)
            manager.end()
        manager.close()

    def test_rejects_budget_and_alias_before_mutation(self):
        model = Toy()
        original = model.transformer_blocks[0].linear.weight.data_ptr()
        with self.assertRaisesRegex(ValueError, 'budget'):
            LayerwiseOffloadManager(model, device='cuda:0', max_weight_usage=1)
        self.assertEqual(original, model.transformer_blocks[0].linear.weight.data_ptr())
        model.transformer_blocks[1].linear.weight = model.transformer_blocks[0].linear.weight
        with self.assertRaisesRegex(ValueError, 'storage'):
            LayerwiseOffloadManager(model, device='cuda:0', max_weight_usage=10**6)

    def test_stage_switches_and_exception(self):
        modules = dict(transformer=Toy().eval(), text_encoder=nn.Linear(16,16),
                       vae=nn.ModuleDict(dict(encoder=nn.Linear(16,16), decoder=nn.Linear(16,16))))
        adapter = LayerwiseMemoryAdapter()
        adapter.register(modules=modules, device='cuda:0', max_weight_usage=10**6)
        controller = MemoryPhaseController(adapter, rank=0, device='cuda:0')
        phases = [(MemoryPhase.TEXT_ENCODE, 'text_encoder'), (MemoryPhase.VAE_ENCODE, 'vae.encoder'),
                  (MemoryPhase.DENOISE, 'transformer'), (MemoryPhase.VAE_DECODE, 'vae.decoder')]
        with torch.no_grad():
            for phase, name in phases:
                controller.enter(phase, component_name=name, window_key=(0,0))
                self.assertEqual(adapter.active_component_name, name)
                for other, tensors in adapter.components.items():
                    if other != name:
                        self.assertTrue(all(t.device.type=='cpu' for t in tensors))
                controller.exit(phase)
                self.assertEqual(adapter.snapshot()['resident_bytes'], 0)
        class Stage:
            @offload_component('transformer')
            def forward(self, batch, args):
                batch.modules['transformer'](torch.ones(2,16,device='cuda'))
                raise RuntimeError('stage failed')
        batch = SimpleNamespace(modules=modules, extra=dict(memory_phase_controller=controller,window_index=1))
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, 'stage failed'):
            Stage().forward(batch, ServerArgs(resource_policy='dynamic_offload',device='cuda:0'))
        self.assertIsNone(adapter.active_component_name)
        self.assertEqual(adapter.manager.used, 0)
        adapter.shutdown()
        adapter.shutdown()

    def test_real_transformer_and_caches(self):
        self._real_transformer_and_caches(compile_enabled=False)

    def test_real_transformer_compile_and_caches(self):
        self._real_transformer_and_caches(compile_enabled=True)

    def _real_transformer_and_caches(self, *, compile_enabled):
        from models.dits.eraserdit_transformer import EraserDiTLTXVideoTransformer3DModel
        from config.server_args import set_global_server_args
        from cache.eraserdit import EraserDiTCacheWindow
        from config.eraserdit_cache import CACHE_DEFAULTS
        set_global_server_args(ServerArgs(device='cuda:0', attention_backend='sdpa'))
        torch.manual_seed(19)
        model = EraserDiTLTXVideoTransformer3DModel(in_channels=3,out_channels=1,
                    num_attention_heads=2,attention_head_dim=16,cross_attention_dim=32,
                    num_layers=3,caption_channels=16).eval().to(dtype=torch.bfloat16)
        reference = copy.deepcopy(model).cuda()
        modules = dict(transformer=model,text_encoder=nn.Linear(16,16),
                       vae=nn.ModuleDict(dict(encoder=nn.Linear(16,16),decoder=nn.Linear(16,16))))
        adapter = LayerwiseMemoryAdapter(prefetch_size=2)
        adapter.register(modules=modules,device='cuda:0',max_weight_usage=10**6)
        if compile_enabled:
            from layers.block_compile import configure_block_compile
            configure_block_compile(model, mode='default')
        values = dict(hidden_states=torch.randn(1,1,1,2,2,device='cuda',dtype=torch.bfloat16),
                      cond_latents=torch.randn(1,1,1,2,2,device='cuda',dtype=torch.bfloat16),
                      mask_values=torch.ones(1,1,1,2,2,device='cuda',dtype=torch.bfloat16),
                      encoder_hidden_states=torch.randn(1,4,16,device='cuda',dtype=torch.bfloat16),
                      encoder_attention_mask=torch.ones(1,4,device='cuda'),
                      timestep=torch.ones(1,device='cuda',dtype=torch.bfloat16),
                      num_frames=1,height=2,width=2,return_dict=False)
        with torch.no_grad():
            for mode in ('off','teacache','cache_dit'):
                b1 = SimpleNamespace(**(CACHE_DEFAULTS | dict(transformer_cache_mode=mode)), extra={})
                b2 = SimpleNamespace(**(CACHE_DEFAULTS | dict(transformer_cache_mode=mode)), extra={})
                adapter.acquire_component_residency('transformer',reason='test')
                with EraserDiTCacheWindow(b1,total_steps=6,num_blocks=3) as c1, EraserDiTCacheWindow(b2,total_steps=6,num_blocks=3) as c2:
                    before = adapter.manager.h2d_count
                    for step in range(6):
                        actual = model(**values,**c1.kwargs('positive',step))[0]
                        expected = reference(**values,**c2.kwargs('positive',step))[0]
                        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    if mode != 'off':
                        self.assertLess(adapter.manager.h2d_count - before, 6 * 3)
                adapter.release_component_residency('transformer',reason='test')
        adapter.shutdown()


if __name__ == '__main__':
    unittest.main()
