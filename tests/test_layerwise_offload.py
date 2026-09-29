"""Migrated SGLang scheduling, repeated inference, layouts and stage cleanup."""
import copy
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from config.server_args import ServerArgs
from memory.backends.layerwise_offload import LayerwiseOffloadManager, OffloadableDiTMixin
from memory.adapters.sglang_memory_adapter import SGLangMemoryAdapter
from memory.validation import validate_memory_config


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)
        self.register_buffer('strided', torch.randn(16, 16).t())
        self.register_buffer('scale', torch.tensor(.01, dtype=torch.float64))
        self.fail = False

    def forward(self, x):
        if self.fail:
            raise RuntimeError('intentional block failure')
        return self.linear(x) + x @ self.strided * self.scale.float()


class Toy(OffloadableDiTMixin, nn.Module):
    layer_names = ['transformer_blocks']

    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([Block() for _ in range(4)])

    def forward(self, x):
        for block in self.transformer_blocks:
            x = block(x)
        return x


class ConfigurationTests(unittest.TestCase):
    def test_prefetch_ratio_and_count(self):
        for value in (-1, True, math.nan, math.inf):
            with self.assertRaises(ValueError):
                ServerArgs(dit_offload_prefetch_size=value)
        for value in (0, .5, 1, 4):
            self.assertEqual(ServerArgs(dit_offload_prefetch_size=value).dit_offload_prefetch_size, value)

    def test_legacy_parameters_are_removed(self):
        for key in ('resource_policy', 'dynamic_offload', 'max_weight_usage', 'pin_memory'):
            with self.assertRaises(TypeError):
                ServerArgs(**{key: 1})

    def test_mutually_exclusive_dit_offload(self):
        for kwargs in ({'dit_cpu_offload': True}, {'use_fsdp_inference': True}):
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                ServerArgs(dit_layerwise_offload=True, **kwargs)

    def test_unsupported_features_fail_before_execution(self):
        with patch('torch.cuda.is_available', return_value=True):
            for kwargs in ({'transformer_quantization': 'int8_w8a8_native'},):
                with self.assertRaises(ValueError):
                    validate_memory_config(ServerArgs(device='cuda', dit_layerwise_offload=True, **kwargs))
            validate_memory_config(ServerArgs(device='cuda', dit_layerwise_offload=True,
                                               enable_torch_compile=True))
            validate_memory_config(ServerArgs(device='cuda', dit_layerwise_offload=True),
                                   SimpleNamespace(transformer_cache_mode='teacache'))
            for mode in ('cache_dit',):
                with self.assertRaisesRegex(ValueError, 'cyclic'):
                    validate_memory_config(ServerArgs(device='cuda', dit_layerwise_offload=True),
                                           SimpleNamespace(transformer_cache_mode=mode))

    def test_cli_and_service_memory_options(self):
        from entrypoints.cli.erase_eraserdit import _build_parser, _build_server_args
        from entrypoints.server import serve
        from pipelines.eraserdit_erase_pipeline import EraserDiTErasePipeline
        parser = _build_parser()
        args = _build_server_args(parser.parse_args(['--model-path', 'data/model', '--dit-offload-prefetch-size', '.5']))
        self.assertTrue(args.dit_layerwise_offload)
        self.assertTrue(args.text_encoder_cpu_offload)
        self.assertEqual(args.dit_offload_prefetch_size, .5)
        server_parser = serve._build_parser()
        server_args = serve._build_server_args(server_parser.parse_args([
            '--model-path', 'data/model', '--task-root', '/tmp/eraserdit-test-tasks', '--no-dit-layerwise-offload',
            '--no-text-encoder-cpu-offload', '--no-vae-cpu-offload']), EraserDiTErasePipeline)
        self.assertFalse(server_args.resolve_resource_policy().enabled)



@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
class LayerwiseTests(unittest.TestCase):
    def test_execution_plan_probe_sparse_failure_and_recovery(self):
        for prefetch in (1, 3):
            torch.manual_seed(123)
            model = Toy().cuda().eval()
            reference = copy.deepcopy(model)
            manager = LayerwiseOffloadManager(model, layers_attr_str='transformer_blocks',
                                               num_layers=4, enabled=True, prefetch_size=prefetch)
            x = torch.randn(2, 16, device='cuda')
            try:
                with torch.no_grad():
                    for indices in ((0,), (0, 1, 2, 3), (0,), (0, 3)):
                        actual, expected = x, x
                        with manager.execution_plan(indices):
                            for i in indices:
                                actual = model.transformer_blocks[i](actual)
                                self.assertTrue(manager._gpu_layers.issubset(set(indices)))
                                expected = reference.transformer_blocks[i](expected)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        self.assertFalse(manager._gpu_layers)
                        self.assertIsNone(manager._execution_plan)
                    model.transformer_blocks[0].fail = True
                    with self.assertRaisesRegex(RuntimeError, 'intentional'):
                        with manager.execution_plan((0, 1)):
                            model.transformer_blocks[0](x)
                    self.assertFalse(manager._gpu_layers)
                    self.assertIsNone(manager._execution_plan)
                    model.transformer_blocks[0].fail = False
                    torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)
            finally:
                manager.release_all()
                torch.cuda.synchronize()
                manager.remove_forward_hooks()

    def test_upstream_prefetch_layout_and_repeated_forward(self):
        for prefetch in (0, .5, 4):
            torch.manual_seed(7)
            model = Toy().eval()
            reference = copy.deepcopy(model).cuda()
            args = ServerArgs(device='cuda:0', dit_layerwise_offload=True,
                              dit_offload_prefetch_size=prefetch)
            adapter = SGLangMemoryAdapter({'transformer': model}, args)
            try:
                manager = adapter.managers[0]
                expected_count = 1 if prefetch == 0 else (3 if prefetch == .5 else 4)
                self.assertEqual(manager.prefetch_size, expected_count)
                self.assertEqual(len(manager._gpu_layers), expected_count)
                x = torch.randn(2, 16, device='cuda')
                with torch.no_grad():
                    for _ in range(3):
                        adapter.acquire_component_residency('transformer', reason='test')
                        for _ in range(3):
                            torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)
                        adapter.release_component_residency('transformer', reason='test')
                        self.assertEqual(adapter.snapshot()['live_layers'], 0)
                        self.assertEqual(model.transformer_blocks[0].linear.weight.shape, (1,))
                weights = dict(manager.iter_cpu_weights())
                self.assertEqual(weights['transformer_blocks.0.strided'].stride(), (1, 16))
                self.assertTrue(weights['transformer_blocks.0.linear.weight'].is_pinned())
            finally:
                adapter.shutdown()
            self.assertFalse(model.transformer_blocks[0]._forward_pre_hooks)
            self.assertEqual(adapter.snapshot()['pinned_cpu_bytes'], 0)

    def test_exception_release_and_retry(self):
        model = Toy().eval()
        reference = copy.deepcopy(model).cuda()
        adapter = SGLangMemoryAdapter({'transformer': model}, ServerArgs(device='cuda:0', dit_layerwise_offload=True))
        try:
            x = torch.randn(2, 16, device='cuda')
            with torch.no_grad():
                adapter.acquire_component_residency('transformer', reason='test')
                model.transformer_blocks[1].fail = True
                with self.assertRaisesRegex(RuntimeError, 'intentional'):
                    model(x)
                adapter.release_component_residency('transformer', reason='failed')
                self.assertEqual(adapter.snapshot()['live_layers'], 0)
                model.transformer_blocks[1].fail = False
                adapter.acquire_component_residency('transformer', reason='retry')
                torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)
                adapter.release_component_residency('transformer', reason='retry')
        finally:
            adapter.shutdown()

    def test_t5_fsdp_outputs_shared_embedding_and_cleanup(self):
        from transformers import T5Config, T5EncoderModel
        import torch.distributed as dist
        model = T5EncoderModel(T5Config(vocab_size=64, d_model=32, d_kv=8, d_ff=64,
                               num_layers=2, num_heads=4, dropout_rate=0)).eval().to(torch.bfloat16)
        reference = copy.deepcopy(model).cuda()
        was_initialized = dist.is_initialized()
        adapter = SGLangMemoryAdapter({'text_encoder': model}, ServerArgs(device='cuda:0', text_encoder_cpu_offload=True))
        try:
            self.assertIs(model.shared.weight, model.encoder.embed_tokens.weight)
            x = torch.randint(0, 64, (1, 8), device='cuda')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                for _ in range(3):
                    adapter.acquire_component_residency('text_encoder', reason='test')
                    torch.testing.assert_close(model(x)[0], reference(x)[0], rtol=0, atol=0)
                    adapter.release_component_residency('text_encoder', reason='test')
                    self.assertTrue(all(p.to_local().device.type == 'cpu' for p in model.parameters()))
        finally:
            adapter.shutdown()
        self.assertEqual(dist.is_initialized(), was_initialized)

    def test_real_transformer_sequential_offload(self):
        from models.dits.eraserdit_transformer import EraserDiTLTXVideoTransformer3DModel
        from config.server_args import set_global_server_args
        set_global_server_args(ServerArgs(device='cuda:0', attention_backend='sdpa'))
        model = EraserDiTLTXVideoTransformer3DModel(in_channels=3, out_channels=1,
                 num_attention_heads=2, attention_head_dim=16, cross_attention_dim=32,
                 num_layers=3, caption_channels=16).eval().to(dtype=torch.bfloat16)
        reference = copy.deepcopy(model).cuda()
        adapter = SGLangMemoryAdapter({'transformer': model}, ServerArgs(device='cuda:0', dit_layerwise_offload=True))
        values = dict(hidden_states=torch.randn(1,1,1,2,2,device='cuda',dtype=torch.bfloat16),
                      cond_latents=torch.randn(1,1,1,2,2,device='cuda',dtype=torch.bfloat16),
                      mask_values=torch.ones(1,1,1,2,2,device='cuda',dtype=torch.bfloat16),
                      encoder_hidden_states=torch.randn(1,3,16,device='cuda',dtype=torch.bfloat16),
                      encoder_attention_mask=torch.ones(1,3,device='cuda',dtype=torch.bool),
                      timestep=torch.tensor([500.],device='cuda',dtype=torch.bfloat16),
                      num_frames=1, height=2, width=2, return_dict=False)
        try:
            with torch.no_grad():
                for _ in range(2):
                    adapter.acquire_component_residency('transformer', reason='test')
                    torch.testing.assert_close(model(**values)[0], reference(**values)[0], rtol=0, atol=0)
                    adapter.release_component_residency('transformer', reason='test')
        finally:
            adapter.shutdown()

    def test_t5_failure_cleanup_and_next_request(self):
        from transformers import T5Config, T5EncoderModel
        model = T5EncoderModel(T5Config(vocab_size=64, d_model=32, d_kv=8, d_ff=64,
                               num_layers=2, num_heads=4, dropout_rate=0)).eval()
        reference = copy.deepcopy(model).cuda()
        adapter = SGLangMemoryAdapter({'text_encoder': model}, ServerArgs(device='cuda:0', text_encoder_cpu_offload=True))
        try:
            ids = torch.randint(0,64,(1,8),device='cuda')
            with torch.no_grad():
                adapter.acquire_component_residency('text_encoder', reason='test')
                with patch.object(model.encoder.block[1], 'forward', side_effect=RuntimeError('T5 failure')):
                    with self.assertRaisesRegex(RuntimeError, 'T5 failure'):
                        model(ids)
                adapter.release_component_residency('text_encoder', reason='failed')
                self.assertTrue(all(p.to_local().device.type == 'cpu' for p in model.parameters()))
                adapter.acquire_component_residency('text_encoder', reason='retry')
                torch.testing.assert_close(model(ids)[0], reference(ids)[0], rtol=0, atol=0)
                adapter.release_component_residency('text_encoder', reason='retry')
        finally:
            adapter.shutdown()

    def test_partial_h2d_failure_can_release_and_restart(self):
        model = Toy().eval()
        reference = copy.deepcopy(model).cuda()
        adapter = SGLangMemoryAdapter({'transformer':model}, ServerArgs(device='cuda:0',dit_layerwise_offload=True))
        original = torch.Tensor.copy_
        transfers = []
        def fail(target, source, *args, **kwargs):
            if target.device.type == 'cuda' and source.device.type == 'cpu':
                transfers.append(1)
                if len(transfers) == 2:
                    raise RuntimeError('copy failure')
            return original(target, source, *args, **kwargs)
        try:
            x = torch.randn(2,16,device='cuda')
            with torch.no_grad():
                adapter.acquire_component_residency('transformer',reason='test')
                with patch.object(torch.Tensor,'copy_',new=fail):
                    with self.assertRaisesRegex(RuntimeError,'copy failure'):
                        model(x)
                adapter.release_component_residency('transformer',reason='failed')
                adapter.acquire_component_residency('transformer',reason='retry')
                torch.testing.assert_close(model(x),reference(x),rtol=0,atol=0)
                adapter.release_component_residency('transformer',reason='retry')
        finally:
            adapter.shutdown()

    def test_two_sessions_share_owned_process_group(self):
        from transformers import T5Config, T5EncoderModel
        import torch.distributed as dist
        args = ServerArgs(device='cuda:0', text_encoder_cpu_offload=True)
        config = T5Config(vocab_size=64, d_model=32, d_kv=8, d_ff=64,
                          num_layers=1, num_heads=4, dropout_rate=0)
        first = SGLangMemoryAdapter({'text_encoder': T5EncoderModel(config).eval()}, args)
        second = None
        try:
            model = T5EncoderModel(config).eval()
            second = SGLangMemoryAdapter({'text_encoder': model}, args)
            first.shutdown()
            self.assertTrue(dist.is_initialized())
            with torch.no_grad():
                second.acquire_component_residency('text_encoder', reason='test')
                self.assertTrue(torch.isfinite(model(torch.ones(1,4,dtype=torch.long,device='cuda'))[0]).all())
                second.release_component_residency('text_encoder', reason='test')
        finally:
            first.shutdown()
            if second is not None:
                second.shutdown()
        self.assertFalse(dist.is_initialized())
