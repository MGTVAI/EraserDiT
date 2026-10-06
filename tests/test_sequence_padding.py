"""Reference SP padding must preserve every payload bit and GEMM row offset."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from layers.sequence_padding import ReferenceSequencePadding


class SequencePaddingTests(unittest.TestCase):
    def test_cpu_autograd(self):
        pad_sequence = ReferenceSequencePadding()
        value = torch.randn(2, 3, 7, requires_grad=True)
        result = pad_sequence(value, 2, 8)
        torch.testing.assert_close(result, F.pad(value, (0, 0, 2, 3)), atol=0, rtol=0)
        result.sum().backward()
        torch.testing.assert_close(value.grad, torch.ones_like(value))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_layouts_bits_and_streams(self):
        stream = torch.cuda.Stream()
        pad_sequence = ReferenceSequencePadding()
        with torch.no_grad(), torch.cuda.stream(stream):
            held = []
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                for batch, local, width in ((1, 7, 2048), (3, 5, 37), (2, 0, 16)):
                    for start, length in ((0, local), (0, local + 7), (3, local + 8)):
                        value = torch.randn(batch, width * 2, local, device='cuda', dtype=dtype)
                        value = value[:, ::2].transpose(1, 2)
                        if local:
                            value[0, 0, :4] = torch.tensor(
                                [-0., float('nan'), float('inf'), float('-inf')], device='cuda')
                        expected = F.pad(value, (0, 0, start, length - local - start))
                        actual = pad_sequence(value, start, length)
                        torch.testing.assert_close(actual.flatten().view(torch.uint8), expected.flatten().view(torch.uint8),
                                                   atol=0, rtol=0)
                        # Projections consume scratch on the same stream and
                        # return independent tensors before subsequent reuse.
                        held.append((actual.clone(), expected))
            for actual, expected in held:
                torch.testing.assert_close(actual.flatten().view(torch.uint8), expected.flatten().view(torch.uint8),
                                           atol=0, rtol=0)
        stream.synchronize()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_autograd_fallback(self):
        pad_sequence = ReferenceSequencePadding()
        value = torch.randn(2, 3, 7, device='cuda', requires_grad=True)
        pad_sequence(value, 2, 8).sum().backward()
        torch.testing.assert_close(value.grad, torch.ones_like(value))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_reuse_preserves_projection_and_zero_rows(self):
        pad = ReferenceSequencePadding()
        weight = torch.randn(64, 32, device='cuda', dtype=torch.bfloat16)
        held = []
        with torch.no_grad():
            for step in range(4):
                value = torch.randn(2, 7, 32, device='cuda', dtype=torch.bfloat16)
                padded = pad(value, 3, 17)
                self.assertEqual(len(pad.buffers), 1)
                actual = F.linear(padded, weight)
                expected = F.linear(F.pad(value, (0, 0, 3, 7)), weight)
                held.append((actual, expected))
                torch.testing.assert_close(padded[:, :3], torch.zeros_like(padded[:, :3]))
                torch.testing.assert_close(padded[:, 10:], torch.zeros_like(padded[:, 10:]))
            other = torch.cuda.Stream()
            other.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(other):
                pad(value, 3, 17)
            torch.cuda.current_stream().wait_stream(other)
            self.assertEqual(len(pad.buffers), 2)
            for actual, expected in held:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_scope_releases_scratch_and_restores_forward_on_failure(self):
        from models.adapters.eraserdit.mesh import SequenceRank
        rank = SequenceRank(SimpleNamespace(degree=2), 1)
        rank.partition(17)
        linear = torch.nn.Linear(32, 64, device='cuda', dtype=torch.bfloat16)
        model = SimpleNamespace(proj_out=linear, transformer_blocks=[])
        value = torch.randn(2, 9, 32, device='cuda', dtype=torch.bfloat16)
        scratch = ReferenceSequencePadding()
        with torch.no_grad():
            expected = linear(F.pad(value, (0, 0, 8, 0)))[:, 8:].contiguous()
            with patch('layers.sequence_padding.ReferenceSequencePadding', return_value=scratch):
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    with rank.linear_scope(model, 'reference'):
                        actual = linear(value)
                        self.assertTrue(scratch.buffers)
                        raise RuntimeError('injected')
            self.assertFalse(scratch.buffers)
            self.assertNotIn('forward', linear.__dict__)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            with rank.linear_scope(model, 'reference'):
                torch.testing.assert_close(linear(value), expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
