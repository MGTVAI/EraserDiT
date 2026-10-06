"""Deterministic chunk selection and conservative profile fallback."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from layers.attention.ulysses_policy import select_head_chunks
from models.adapters.eraserdit.nccl_sequence import DistributedSequenceRank


class UlyssesPolicyTests(unittest.TestCase):
    def test_server_validation_accepts_auto_and_keeps_topology_constraints(self):
        from config.eraserdit import EraserDiTPipelineConfig
        from config.server_args import ServerArgs
        with patch.dict(os.environ, MGERASE_ULYSSES_HEAD_CHUNKS='auto',
                        MGERASE_ULYSSES_OUTPUT_OVERLAP='1', MGERASE_NCCL_PACKING='direct'):
            for degree in (2, 4):
                ServerArgs(pipeline_config=EraserDiTPipelineConfig(
                    dit_parallel_backend='nccl', sp_degree=degree, sp_linear_mode='aligned'))
            with self.assertRaisesRegex(ValueError, 'Ulysses head chunks require'):
                ServerArgs(pipeline_config=EraserDiTPipelineConfig(dit_parallel_backend='nccl', tp_degree=2))

    def test_profile_and_unknown_fallback(self):
        with torch.inference_mode(), patch('torch.__version__', '2.6.0+cu126'), \
                patch('torch.version.cuda', '12.6'), \
                patch('torch.cuda.get_device_name', return_value='NVIDIA L40S'):
            for degree, length, expected in ((2, 32640, 4), (2, 10200, 4),
                                              (4, 32640, 4), (4, 10200, 2), (4, 10000, 1)):
                q = SimpleNamespace(ndim=4, shape=(1, length // degree, 32, 64),
                                    dtype=torch.bfloat16, is_cuda=True, device='cuda:0')
                args = dict(degree=degree, length=length, packing='direct', ring_size=1)
                self.assertEqual(select_head_chunks(q, **args), expected)
                self.assertEqual(select_head_chunks(q, **dict(args, ring_size=2)), 1)
                self.assertEqual(select_head_chunks(q, **dict(args, packing='reference')), 1)
                with patch('torch.cuda.get_device_name', return_value='unvalidated GPU'):
                    self.assertEqual(select_head_chunks(q, **args), 1)
                q.dtype = torch.float32
                self.assertEqual(select_head_chunks(q, **args), 1)

    def test_configuration_is_opt_in(self):
        groups = SimpleNamespace(topology=SimpleNamespace(sp=4, ulysses=4), coordinates=(0, 0, 0))
        for config, policy, chunks in [('auto', 'auto', 1), ('1', 'fixed', 1),
                                      ('2', 'fixed', 2), ('4', 'fixed', 4)]:
            with patch.dict(os.environ, MGERASE_ULYSSES_HEAD_CHUNKS=config):
                sequence = DistributedSequenceRank(groups)
            self.assertEqual((sequence.head_chunk_policy, sequence.head_chunks), (policy, chunks))
        with patch.dict(os.environ, MGERASE_ULYSSES_HEAD_CHUNKS='3'):
            with self.assertRaisesRegex(ValueError, 'auto, 1, 2 or 4'):
                DistributedSequenceRank(groups)


if __name__ == '__main__':
    unittest.main()
