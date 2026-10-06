"""Selective SP protection, conservative fallback and wrapper lifetime."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from config.eraserdit import EraserDiTPipelineConfig
from config.server_args import ServerArgs
from layers.sequence_linear import aligned_sequence_lengths
from models.adapters.eraserdit.mesh import SequenceRank


class RecordingLinear(nn.Linear):
    def forward(self, value):
        self.seen = value.shape[1]
        return super().forward(value)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.ModuleList([RecordingLinear(4, 16), nn.GELU(approximate='tanh'),
                                  RecordingLinear(16, 4)])

    def forward(self, value):
        for module in self.net:
            value = module(value)
        return value


def model_fixture():
    def attention():
        return SimpleNamespace(to_q=RecordingLinear(4, 4).bfloat16(),
            to_k=RecordingLinear(4, 4).bfloat16(), to_v=RecordingLinear(4, 4).bfloat16(),
            to_out=[RecordingLinear(4, 4).bfloat16()])
    block = SimpleNamespace(attn1=attention(), attn2=attention(), ff=FeedForward().bfloat16())
    return SimpleNamespace(proj_out=RecordingLinear(4, 4).bfloat16(), transformer_blocks=[block])


class SequenceLinearTests(unittest.TestCase):
    def test_unknown_model_and_gradient_profile_fall_back(self):
        model = model_fixture()
        self.assertEqual(aligned_sequence_lengths(model, 2)[0], ())
        with torch.no_grad():
            self.assertEqual(aligned_sequence_lengths(model, 4)[0], ())

    def test_selective_shapes_and_unknown_length_fallback(self):
        # Substitute only the hardware/model profile; exercise the real wrapper
        # routing, including its nested FFN down projection, on small weights.
        for degree, length, allowed, down_length in (
                (2, 16, (16,), 8), (4, 16, (16,), 4),
                (4, 10200, (10200,), 3072), (4, 17, (16,), 17)):
            with self.subTest(degree=degree, length=length):
                model = model_fixture()
                rank = SequenceRank(SimpleNamespace(degree=degree), degree - 1)
                rank.partition(length)
                value = torch.randn(1, rank.end-rank.start, 4).bfloat16()
                block = model.transformer_blocks[0]
                with torch.no_grad(), patch('layers.sequence_linear.aligned_sequence_lengths',
                                             return_value=(allowed, None)):
                    with rank.linear_scope(model, 'aligned'):
                        block.attn1.to_q(value)
                        result = block.ff(value)
                        model.proj_out(value)
                    self.assertEqual(result.shape, value.shape)
                    self.assertEqual(block.attn1.to_q.seen,
                                     value.shape[1] if length in allowed else length)
                    self.assertEqual(block.ff.net[0].seen,
                                     value.shape[1] if length in allowed else length)
                    self.assertEqual(block.ff.net[2].seen, down_length)
                    self.assertEqual(model.proj_out.seen, length)
                    self.assertEqual(rank.linear_report['effective'],
                                     'aligned' if length in allowed else 'reference')
                for module in (model.proj_out, block.ff, block.ff.net[2], block.attn1.to_q):
                    self.assertNotIn('forward', module.__dict__)

    def test_compact_padding_all_rank_offsets_and_reference_control(self):
        model = model_fixture()
        block = model.transformer_blocks[0]
        with torch.no_grad(), patch('layers.sequence_linear.aligned_sequence_lengths',
                                   return_value=((10200,), None)):
            for index in range(4):
                rank = SequenceRank(SimpleNamespace(degree=4), index)
                rank.partition(10200)
                value = torch.randn(1, 2550, 4).bfloat16()
                rank.compact_ffn_down = False
                with rank.linear_scope(model, 'aligned'):
                    expected = block.ff(value)
                self.assertEqual(block.ff.net[2].seen, 10200)
                rank.compact_ffn_down = True
                with rank.linear_scope(model, 'aligned'):
                    actual = block.ff(value)
                self.assertEqual(block.ff.net[2].seen, 3072)
                self.assertEqual(rank.linear_report['compact_ffn_down_calls'], 1)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_fallback_matches_reference_with_batch_and_unequal_shards(self):
        model = model_fixture()
        rank = SequenceRank(SimpleNamespace(degree=4), 3)
        block = model.transformer_blocks[0]
        with torch.no_grad():
            for length in (17, 20, 17):
                rank.partition(length)
                value = torch.randn(2, rank.end-rank.start, 4).bfloat16()
                with rank.linear_scope(model, 'reference'):
                    expected = block.ff(block.attn1.to_q(value))
                with rank.linear_scope(model, 'aligned'):
                    actual = block.ff(block.attn1.to_q(value))
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_exception_restores_nested_and_preexisting_forward(self):
        model = model_fixture()
        block = model.transformer_blocks[0]
        previous = block.ff.net[2].forward
        block.ff.net[2].forward = previous
        rank = SequenceRank(SimpleNamespace(degree=4), 1)
        rank.partition(10200)
        with torch.no_grad(), patch('layers.sequence_linear.aligned_sequence_lengths',
                                     return_value=((10200,), None)):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with rank.linear_scope(model, 'aligned'):
                    block.ff(torch.randn(1, 2550, 4).bfloat16())
                    raise RuntimeError('injected')
        self.assertIs(block.ff.net[2].forward, previous)
        self.assertNotIn('forward', block.ff.__dict__)

    def test_protected_fp32_output_does_not_hide_local_execution(self):
        model = model_fixture()
        model.proj_out.float()
        rank = SequenceRank(SimpleNamespace(degree=2), 1)
        rank.partition(16)
        value = torch.randn(1, 8, 4).bfloat16()
        with torch.no_grad(), patch('layers.sequence_linear.aligned_sequence_lengths',
                                     return_value=((16,), None)):
            with rank.linear_scope(model, 'aligned'):
                model.transformer_blocks[0].attn1.to_q(value)
                model.proj_out(value.float())
        self.assertEqual(rank.linear_report['effective'], 'aligned')
        self.assertEqual(rank.linear_report['local_calls'], 1)
        self.assertEqual(rank.linear_report['reference_calls'], 1)
        self.assertIsNone(rank.linear_report['fallback_reason'])

    def test_topology_contract(self):
        for degree in (2, 4):
            ServerArgs(pipeline_config=EraserDiTPipelineConfig(
                dit_parallel_backend='nccl', sp_degree=degree, sp_linear_mode='aligned'))
        for kwargs in ({'dit_parallel_backend': 'peer', 'sp_degree': 2},
                       {'sp_degree': 1}, {'sp_degree': 2, 'ring_degree': 2},
                       {'sp_degree': 2, 'tp_degree': 2},
                       {'sp_degree': 2, 'dit_fsdp_shard_degree': 2}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, 'aligned SP'):
                ServerArgs(pipeline_config=EraserDiTPipelineConfig(
                    **dict({'dit_parallel_backend': 'nccl', 'sp_linear_mode': 'aligned'}, **kwargs)))


if __name__ == '__main__':
    unittest.main()
