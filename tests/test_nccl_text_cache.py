"""Resident rank cache invalidation and diagnostic trace lifecycle."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from config.dit_parallel import DiTTopology
from config.eraserdit import EraserDiTPipelineConfig
from config.server_args import ServerArgs, set_global_server_args
from models.adapters.eraserdit.nccl_runner import DiTRankRunner
from models.dits.eraserdit_transformer import EraserDiTLTXVideoTransformer3DModel
from utils.dit_profile import DiTStepProfiler


class NCCLTextCacheTests(unittest.TestCase):
    def test_rank_cache_invalidation(self):
        set_global_server_args(ServerArgs(device='cpu', attention_backend='sdpa'))
        torch.manual_seed(12)
        model = EraserDiTLTXVideoTransformer3DModel(in_channels=3, out_channels=1,
            num_attention_heads=4, attention_head_dim=16, cross_attention_dim=64,
            num_layers=2, caption_channels=16).eval().requires_grad_(False)
        groups = SimpleNamespace(topology=DiTTopology(), coordinates=(0, 0, 0, 0, 0), rank=0)
        config = EraserDiTPipelineConfig()
        runner = DiTRankRunner(model, groups, config)
        reference = DiTRankRunner(deepcopy(model), groups, config)
        hidden = torch.randn(1, 1, 1, 3, 3)
        values = dict(cond_latents=torch.randn_like(hidden), mask_values=torch.ones_like(hidden),
            encoder_hidden_states=torch.randn(1, 5, 16), encoder_attention_mask=torch.ones(1, 5),
            num_frames=1, height=3, width=3, return_dict=False)
        static = dict(positive=values, negative=dict(values,
            encoder_hidden_states=values['encoder_hidden_states'] + .7))

        def check(first=False):
            packet = dict(hidden=hidden, timestep=torch.ones(1), static=static if first else None)
            expected = reference.predict(packet)
            actual = runner.predict(dict(packet, cache_text_projections=True))
            for a, b in zip(expected, actual):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        check(first=True)
        check()
        self.assertEqual(runner.text_caches['positive'].projection_hits, 1)
        self.assertIsNot(runner.text_caches['positive'], runner.text_caches['negative'])
        with torch.no_grad():
            values['encoder_hidden_states'].add_(.2)
        check()  # In-place conditioning update must invalidate all layer K/V.
        for current in (model, reference.model):
            with torch.no_grad():
                current.transformer_blocks[0].attn2.to_k.weight.add_(.125)
        check()  # Normal weight edits invalidate retained projections.
        with self.assertRaisesRegex(ValueError, 'cannot change'):
            runner.predict(dict(hidden=hidden, timestep=torch.ones(1), static=None))
        check(first=True)
        self.assertEqual(runner.text_caches['positive'].projection_hits, 0)
        runner.reset()
        self.assertFalse(runner.text_caches)
        self.assertIsNone(runner.static)

    def test_profile_is_sampled_and_removes_hooks(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
        value = torch.ones(1, 4)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,
                MGERASE_DIT_PROFILE_DIR=directory, MGERASE_DIT_PROFILE_STEP='2'):
            profiler = DiTStepProfiler(0, 'cpu')
            for _ in range(3):
                with profiler.capture(model):
                    actual = model(value)
                torch.testing.assert_close(actual, model(value), atol=0, rtol=0)
            self.assertEqual(len(profiler.artifacts), 1)
            summary = json.loads(Path(profiler.artifacts[0]['summary']).read_text())
            self.assertIn('dit.module.0', {e['name'] for e in summary['events']})
            self.assertTrue(all('device_type' in e for e in summary['events']))
            self.assertTrue(Path(profiler.artifacts[0]['trace']).exists())
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
            profiler.reset()
            with profiler.capture(model):
                model(value)
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                with profiler.capture(model):
                    model(value)
                    raise RuntimeError('test failure')
            self.assertFalse(profiler.active)
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_disabled_profile_does_not_install_hooks(self):
        with patch.dict(os.environ, MGERASE_DIT_PROFILE_DIR=''):
            profiler = DiTStepProfiler(0, 'cpu')
            model = torch.nn.Sequential(torch.nn.Linear(2, 2))
            with patch('torch.profiler.profile', side_effect=AssertionError('unexpected profiler')):
                for _ in range(3):
                    with profiler.capture(model):
                        self.assertFalse(model[0]._forward_hooks)
                        model(torch.ones(1, 2))


if __name__ == '__main__':
    unittest.main()
