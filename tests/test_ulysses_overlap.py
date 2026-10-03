"""Real NCCL head-pipeline ordering, precision and stream-lifetime checks."""
import os
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _rank(rank, degree, cfg, rendezvous):
    from config.dit_parallel import DiTTopology
    from distributed.dit_groups import DiTGroups
    from models.adapters.eraserdit.nccl_sequence import DistributedSequenceRank
    from layers.attention.backends.sdpa import SDPAImpl
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group('nccl', init_method=rendezvous, rank=rank,
                           world_size=degree * cfg, timeout=timedelta(seconds=60))
    try:
        groups = DiTGroups(DiTTopology(ulysses=degree, cfg=cfg))
        with patch.dict(os.environ, MGERASE_NCCL_PACKING='direct', MGERASE_ULYSSES_HEAD_CHUNKS='1'):
            sequence = DistributedSequenceRank(groups)
        impl = SDPAImpl(16, 64, False, .125)
        metadata = SimpleNamespace(attn_mask=None)
        stream = torch.cuda.Stream()
        held = []
        with torch.no_grad(), torch.cuda.stream(stream):
            for length in (19, 24, 19):
                torch.manual_seed(100 + length + groups.coordinates[3])
                shard = sequence.partition(length)
                qkv = [torch.randn(2, length, 16, 64, device=rank, dtype=torch.bfloat16)[:, shard]
                       for _ in range(3)]
                qkv[1] = qkv[1].transpose(1, 2).contiguous().transpose(1, 2)
                sequence.head_chunks = 1
                reference = sequence.attention(*qkv, impl, metadata)
                for chunks, overlap in ((2, False), (2, True), (4, True)):
                    sequence.head_chunks, sequence.head_overlap = chunks, overlap
                    output = sequence.attention(*qkv, impl, metadata)
                    torch.testing.assert_close(output, reference, atol=0, rtol=0)
                    held.append((output, output.clone()))
                # All ranks fail after enqueueing the same input collectives;
                # a subsequent call must remain safe on the same stream/group.
                class Broken:
                    causal = dropout = False
                    def forward(self, *args):
                        raise RuntimeError('injected attention failure')
                try:
                    sequence.attention(*qkv, Broken(), metadata)
                except RuntimeError as error:
                    assert 'injected attention failure' in str(error)
                else:
                    raise AssertionError('injected failure not propagated')
                output = sequence.attention(*qkv, impl, metadata)
                torch.testing.assert_close(output, reference, atol=0, rtol=0)
                del qkv, reference, output
            for output, snapshot in held:
                torch.testing.assert_close(output, snapshot, atol=0, rtol=0)
        stream.synchronize()
    finally:
        dist.destroy_process_group()


class UlyssesOverlapTests(unittest.TestCase):
    def test_invalid_chunk_count(self):
        from models.adapters.eraserdit.nccl_sequence import DistributedSequenceRank
        groups = SimpleNamespace(topology=SimpleNamespace(sp=2, ulysses=2), coordinates=(0, 0, 0))
        for value in ('0', '3', '-1', 'invalid'):
            with patch.dict(os.environ, MGERASE_ULYSSES_HEAD_CHUNKS=value), self.assertRaises(ValueError):
                DistributedSequenceRank(groups)

    def test_unsupported_topology_rejected(self):
        from config.eraserdit import EraserDiTPipelineConfig
        from config.server_args import ServerArgs
        for options in ({}, {'sp_degree': 2, 'ring_degree': 2},
                        {'sp_degree': 2, 'tp_degree': 2}, {'sp_degree': 2, 'dit_fsdp_shard_degree': 2}):
            with patch.dict(os.environ, MGERASE_ULYSSES_HEAD_CHUNKS='2', MGERASE_NCCL_PACKING='direct'):
                with self.assertRaisesRegex(ValueError, 'head chunks'):
                    ServerArgs(pipeline_config=EraserDiTPipelineConfig(dit_parallel_backend='nccl', **options))

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_DIT_NCCL') == '1', 'opt-in NCCL GPUs')
    def test_head_pipeline(self):
        if torch.cuda.device_count() < 2:
            self.skipTest('at least two CUDA devices required')
        for degree, cfg in ((2, 1), (2, 2), (4, 1)):
            if torch.cuda.device_count() < degree * cfg:
                continue
            with tempfile.TemporaryDirectory() as directory:
                mp.spawn(_rank, args=(degree, cfg, f'file://{directory}/store'), nprocs=degree * cfg, join=True)


if __name__ == '__main__':
    unittest.main()
