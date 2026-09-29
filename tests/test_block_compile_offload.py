"""FFN compilation must tolerate weight eviction, new shapes and warmup errors."""
import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from layers.block_compile import configure_block_compile, prepare_block_compile, remove_block_compile
from layers.operator_fusion.registry import resolve_operator_fusion_decision
from memory.backends.layerwise_offload import LayerwiseOffloadManager


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(16, 32)
        self.down = nn.Linear(32, 16)

    def forward(self, value):
        return self.down(torch.nn.functional.gelu(self.up(value)))


class FFNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = FeedForward()

    def forward(self, value):
        return self.ff(value)


class FFNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=2, attention_head_dim=8)
        self.transformer_blocks = nn.ModuleList([FFNBlock() for _ in range(3)])

    def forward(self, value):
        for block in self.transformer_blocks:
            value = block(value)
        return value


class CompileFusionTests(unittest.TestCase):
    def test_ffn_compile_keeps_fusion_outside_compiled_region(self):
        with patch('layers.operator_fusion.registry._triton_is_available', return_value=True):
            for backend in ('auto', 'triton'):
                args = SimpleNamespace(operator_fusion_backend=backend, sp_degree=1,
                                       enable_torch_compile=False)
                expected = resolve_operator_fusion_decision(args)
                args.enable_torch_compile = True
                self.assertEqual(resolve_operator_fusion_decision(args), expected)
                args.sp_degree = 2
                if backend == 'triton':
                    with self.assertRaisesRegex(ValueError, 'SP torch.compile'):
                        resolve_operator_fusion_decision(args)
                else:
                    self.assertEqual(resolve_operator_fusion_decision(args).effective_ops, ())


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class CompileOffloadTests(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()

    def test_repeated_shapes_and_eviction_match_resident_compilation(self):
        for backend in ('native', 'inductor'):
            with self.subTest(backend=backend), patch.dict(os.environ, MGERASE_COMPILE_LINEAR_BACKEND=backend):
                torch.manual_seed(812)
                model = FFNModel().cuda().bfloat16().eval()
                reference = copy.deepcopy(model)
                manager = LayerwiseOffloadManager(model, layers_attr_str='transformer_blocks',
                                                 num_layers=3, enabled=True, prefetch_size=1)
                model.layerwise_offload_managers = [manager]
                try:
                    configure_block_compile(model, mode='default')
                    configure_block_compile(reference, mode='default')
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        for tokens in (33, 17, 33):
                            for target in (model, reference):
                                prepare_block_compile(target, batch_size=1, sequence_length=tokens,
                                                      device='cuda:0', dtype=torch.bfloat16)
                            self.assertFalse(manager._gpu_layers)
                            value = torch.randn(1, tokens, 16, device='cuda', dtype=torch.bfloat16)
                            graphs = torch._dynamo.utils.counters['stats']['unique_graphs']
                            for _ in range(3):
                                actual = model(value)
                                expected = reference(value)
                                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                                manager.release_all()
                            self.assertEqual(torch._dynamo.utils.counters['stats']['unique_graphs'], graphs)
                        self.assertEqual(len(model.transformer_blocks[0].ff._compile_warmed_shapes), 2)
                finally:
                    manager.release_all()
                    manager.remove_forward_hooks()
                    remove_block_compile(model)
                    remove_block_compile(reference)

    def test_warmup_failure_releases_weights_and_can_retry(self):
        model = FFNModel().cuda().bfloat16().eval()
        manager = LayerwiseOffloadManager(model, layers_attr_str='transformer_blocks',
                                         num_layers=3, enabled=True, prefetch_size=1)
        model.layerwise_offload_managers = [manager]
        try:
            with patch.dict(os.environ, MGERASE_COMPILE_LINEAR_BACKEND='native'):
                configure_block_compile(model, mode='default')
            ff = model.transformer_blocks[1].ff
            with patch.object(ff, 'forward', side_effect=RuntimeError('warmup failure')):
                with self.assertRaisesRegex(RuntimeError, 'warmup failure'):
                    prepare_block_compile(model, batch_size=1, sequence_length=23,
                                          device='cuda:0', dtype=torch.bfloat16)
            self.assertFalse(manager._gpu_layers)
            self.assertFalse(ff._compile_warmed_shapes)
            prepare_block_compile(model, batch_size=1, sequence_length=23,
                                  device='cuda:0', dtype=torch.bfloat16)
            self.assertFalse(manager._gpu_layers)
            self.assertEqual(len(ff._compile_warmed_shapes), 1)
        finally:
            manager.release_all()
            manager.remove_forward_hooks()
            remove_block_compile(model)
