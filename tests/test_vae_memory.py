"""Numerical/layout checks for bounded VAE normalization temporaries."""

import unittest

import torch
from diffusers.models.normalization import RMSNorm

from models.vaes.memory import ChunkedRMSNorm, configure_vae_memory
from models.vaes.eraserdit_vae import LTXVideoCausalConv3d, LTXVideoDownsampler3d


class VAEMemoryTests(unittest.TestCase):
    def check_normalization(self, device):
        for dtype in (torch.float32, torch.bfloat16):
            for affine in (False, True):
                for channels_last in (False, True):
                    with self.subTest(device=device, dtype=dtype, affine=affine, channels_last=channels_last):
                        reference = RMSNorm(32, eps=1e-8, elementwise_affine=affine).to(device, dtype)
                        chunked = ChunkedRMSNorm(32, eps=1e-8, elementwise_affine=affine).to(device, dtype)
                        chunked.load_state_dict(reference.state_dict())
                        chunked.chunk_size = 2 * 32 * 7 * 9 * 3
                        x = torch.randn(2, 32, 11, 7, 9, device=device, dtype=dtype).movedim(1, -1)
                        if channels_last:
                            x = x.contiguous()
                        with torch.no_grad():
                            expected, actual = reference(x), chunked(x)
                        # CPU strided FP32 reductions can change SIMD grouping.
                        tol = 5e-7 if device == 'cpu' and dtype == torch.float32 else 0
                        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
                        self.assertEqual(actual.stride(), expected.stride())
                        # Toggling on an existing model must restore native behavior.
                        configure_vae_memory(chunked, False)
                        with torch.no_grad():
                            torch.testing.assert_close(chunked(x), expected, rtol=0, atol=0)

    def test_cpu_normalization(self):
        self.check_normalization('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
    def test_cuda_normalization(self):
        self.check_normalization('cuda')

    def test_training_fallback(self):
        reference = RMSNorm(8, eps=1e-8)
        chunked = ChunkedRMSNorm(8, eps=1e-8)
        chunked.load_state_dict(reference.state_dict())
        chunked.chunk_size = 1
        x = torch.randn(1, 5, 3, 7, 8, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        reference(x).sum().backward()
        chunked(y).sum().backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)

    def test_convolution_context_and_boundaries(self):
        # Odd frame count exercises the last partial slice; grouped/dilated
        # convolutions and temporal strides check receptive-field indexing.
        for causal in (False, True):
            for kernel in (1, 3):
                for stride in (1, 2):
                    for dilation in (1, 2):
                        with self.subTest(causal=causal, kernel=kernel, stride=stride, dilation=dilation):
                            conv = LTXVideoCausalConv3d(4, 6, kernel_size=kernel,
                                stride=(stride, 1, 1), dilation=(dilation, 1, 1),
                                groups=2, is_causal=causal).double().eval()
                            x = torch.randn(2, 4, 13, 7, 9, dtype=torch.float64)
                            # Discontinuities make artificial chunk-boundary
                            # replication visible even far from global edges.
                            x[:, :, 3:6] += 10
                            with torch.no_grad():
                                expected = conv(x)
                                conv.chunk_size = x.numel() // 13 * 3
                                actual = conv(x)
                            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_downsampling_preserves_input(self):
        for stride in ((1, 2, 2), (2, 1, 1), (2, 2, 2)):
            with self.subTest(stride=stride):
                module = LTXVideoDownsampler3d(4, 8, stride=stride).double().eval()
                x = torch.randn(1, 4, 13, 8, 10, dtype=torch.float64)
                saved = x.clone()
                with torch.no_grad():
                    expected = module(x)
                    module.conv.chunk_size = 4 * 8 * 10 * 3
                    actual = module(x)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(x, saved, rtol=0, atol=0)

    def test_cli_and_service_options(self):
        from entrypoints.cli.erase_eraserdit import _build_parser, _build_server_args
        parser = _build_parser()
        args = parser.parse_args(['--model-path', 'data/model', '--vae-low-memory'])
        self.assertTrue(_build_server_args(args).pipeline_config.vae_low_memory)
        self.assertFalse(parser.parse_args([]).vae_low_memory)
        from entrypoints.server.serve import _build_parser as service_parser
        service_args = service_parser().parse_args([
            '--model-path', 'data/model', '--task-root', 'outputs', '--vae-low-memory'])
        self.assertTrue(service_args.vae_low_memory)


if __name__ == '__main__':
    unittest.main()
