"""Bitwise Ulysses wire-layout checks without a GPU or process group."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.adapters.eraserdit.nccl_sequence import DistributedSequenceRank


class UlyssesPackingTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for direct QKV pack')
    def test_direct_qkv_wire_layout_and_special_values(self):
        from layers.attention.qkv_packing import pack_qkv
        for batch, length in ((1, 7), (3, 16)):
            for dtype in (torch.bfloat16, torch.float32):
                for degree in (2, 4):
                    qkv = [torch.randn(batch, length, 8, 16, device='cuda', dtype=dtype) for _ in range(3)]
                    qkv[0].flatten()[:4] = torch.tensor([-0., float('inf'), float('-inf'), float('nan')], device='cuda')
                    # Q/K/V may have different layouts; padding is deliberately
                    # not sent and all payload bits must survive the copy.
                    qkv[1] = qkv[1].transpose(1, 2).contiguous().transpose(1, 2)
                    padded = torch.empty(batch, length, 8, 32, device='cuda', dtype=dtype)
                    padded[..., ::2].copy_(qkv[2])
                    qkv[2] = padded[..., ::2]
                    joined = torch.cat(qkv, dim=0)
                    expected = torch.cat([x.contiguous().flatten() for x in joined.split(8 // degree, dim=2)])
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        actual = pack_qkv(*qkv, degree)
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0)
                    width = 8 // degree
                    for offset, count in ((0, 1), (width - 1, 1)):
                        expected_chunk = torch.cat([part[:, :, offset:offset + count].contiguous().flatten()
                                                    for part in joined.split(width, dim=2)])
                        actual_chunk = pack_qkv(*qkv, degree, head_offset=offset, head_count=count)
                        torch.testing.assert_close(actual_chunk.view(torch.uint8), expected_chunk.view(torch.uint8),
                                                   atol=0, rtol=0)

    def test_wire_layout_and_roundtrip(self):
        # Batch=3 covers combined QKV. Noncontiguous layouts cover attention
        # transposes; odd lengths exercise the retained unequal-shard path.
        for degree in (2, 4):
            for length in (16, 19):
                for batch in (1, 3):
                    for dtype in (torch.float32, torch.bfloat16):
                        for strided in (False, True):
                            with self.subTest(degree=degree, length=length, batch=batch,
                                              dtype=dtype, strided=strided):
                                self.check_layout(degree, length, batch, dtype, strided)

    def check_layout(self, degree, length, batch, dtype, strided):
        heads, dim = 8, 4
        full = torch.arange(batch * length * heads * dim).reshape(batch, length, heads, dim).to(dtype)
        lengths = [length * (r + 1) // degree - length * r // degree for r in range(degree)]
        head_width = heads // degree
        for rank in range(degree):
            groups = SimpleNamespace(topology=SimpleNamespace(sp=degree, ulysses=degree),
                                     coordinates=(0, rank, 0),
                                     get=lambda name: (None, tuple(range(degree)), rank))
            for mode in ('reference', 'packed'):
                with patch.dict(os.environ, MGERASE_NCCL_PACKING=mode):
                    sequence = DistributedSequenceRank(groups)
                local = full[:, sequence.partition(length)]
                if strided:
                    local = local.transpose(1, 2).contiguous().transpose(1, 2)
                expected_wire = torch.cat([part.contiguous().flatten()
                                           for part in local.split(head_width, dim=2)])
                head_shard = full[:, :, rank * head_width:(rank + 1) * head_width]
                received = torch.cat([part.contiguous().flatten()
                                      for part in head_shard.split(lengths, dim=1)])
                self.exchange(sequence._ulysses_input, local, expected_wire, received,
                              [batch * n * head_width * dim for n in lengths],
                              [local.numel() // degree] * degree, head_shard)
                if strided:
                    head_shard = head_shard.transpose(1, 2).contiguous().transpose(1, 2)
                received = torch.cat([part.contiguous().flatten()
                                      for part in local.split(head_width, dim=2)])
                expected_wire = torch.cat([part.contiguous().flatten()
                                           for part in head_shard.split(lengths, dim=1)])
                self.exchange(sequence._ulysses_output, head_shard, expected_wire, received,
                              [local.numel() // degree] * degree,
                              [batch * n * head_width * dim for n in lengths], local)

    def exchange(self, function, value, wire, received, recv_counts, send_counts, expected):
        def collective(output, packed, output_splits, input_splits, group):
            self.assertEqual(output_splits, recv_counts)
            self.assertEqual(input_splits, send_counts)
            self.assertTrue(packed.is_contiguous())
            torch.testing.assert_close(packed, wire, atol=0, rtol=0)
            output.copy_(received)
        with patch('models.adapters.eraserdit.nccl_sequence.dist.all_to_all_single', collective):
            torch.testing.assert_close(function(value), expected, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
