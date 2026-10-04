"""Complete AdaLN fusion: rounding error, dispatch and block integration."""
import unittest
from types import SimpleNamespace

import torch
from diffusers.models.normalization import RMSNorm
from layers.operator_fusion.config import normalize_operator_fusion_ops
from layers.operator_fusion.registry import resolve_operator_fusion_decision
from layers.operator_fusion.rmsnorm_adaln_fast import apply_fused_rmsnorm_adaln_fast
from layers.operator_fusion.runtime import operator_fusion_request_scope


def decision(backend='triton'):
    return resolve_operator_fusion_decision(SimpleNamespace(
        operator_fusion_backend=backend, operator_fusion_ops='rmsnorm_adaln_fast', sp_degree=1))


class AdaLNFastTests(unittest.TestCase):
    def test_explicit_and_exclusive(self):
        self.assertNotIn('rmsnorm_adaln_fast', normalize_operator_fusion_ops(None))
        with self.assertRaisesRegex(ValueError, 'only one'):
            normalize_operator_fusion_ops('rmsnorm_adaln,rmsnorm_adaln_fast')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_numerics_noncontiguous_modulation_and_stats(self):
        torch.manual_seed(472)
        norm = RMSNorm(2048, eps=1e-6, elementwise_affine=False).cuda()
        with torch.inference_mode():
            for tokens, magnitude in [(1, 0.), (257, .001), (1025, 100.)]:
                h = torch.randn(1, tokens, 2048, device='cuda', dtype=torch.bfloat16) * magnitude
                table = torch.randn(1, 1, 6, 2048, device='cuda', dtype=torch.bfloat16)
                scale, shift = table[:, :, 1], table[:, :, 4]
                before = [v.clone() for v in (h, scale, shift)]
                expected = norm(h) * (1 + scale) + shift
                d = decision()
                with operator_fusion_request_scope(d) as stats:
                    actual = apply_fused_rmsnorm_adaln_fast(h, scale, shift, norm, decision=d)
                self.assertEqual(stats.fused_calls, {'rmsnorm_adaln_fast': 1})
                self.assertTrue(actual.isfinite().all())
                self.assertEqual(actual.dtype, expected.dtype)
                error = ((actual.float()-expected.float()).square().mean()
                         / expected.float().square().mean().clamp_min(1e-12)).sqrt()
                self.assertLess(float(error), .001)
                for a, b in zip((h, scale, shift), before):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_fallback_and_autograd(self):
        norm = RMSNorm(2048, eps=1e-6, elementwise_affine=False).cuda()
        for shape, requires_grad, reason in [((2, 3, 2048), False, 'batch_size'),
                                              ((1, 3, 2048), True, 'autograd')]:
            h = torch.randn(shape, device='cuda', dtype=torch.bfloat16, requires_grad=requires_grad)
            scale, shift = [torch.randn(shape[0], 1, 2048, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
            actual = apply_fused_rmsnorm_adaln_fast(h, scale, shift, norm, decision=decision('auto'))
            torch.testing.assert_close(actual, norm(h) * (1 + scale) + shift, rtol=0, atol=0)
            if requires_grad:
                actual.sum().backward()
                self.assertTrue(h.grad.isfinite().all())
            with self.assertRaisesRegex(RuntimeError, reason):
                apply_fused_rmsnorm_adaln_fast(h, scale, shift, norm, decision=decision())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_block_integration(self):
        from config.server_args import ServerArgs, set_global_server_args
        from models.dits.eraserdit_transformer import EraserDiTLTXVideoTransformer3DModel
        args = ServerArgs(device='cuda', attention_backend='sdpa')
        set_global_server_args(args)
        torch.manual_seed(713)
        model = EraserDiTLTXVideoTransformer3DModel(
            in_channels=3, out_channels=1, num_attention_heads=32, attention_head_dim=64,
            cross_attention_dim=2048, num_layers=2, caption_channels=16).cuda().bfloat16().eval()
        inputs = dict(hidden_states=torch.randn(1, 1, 1, 4, 8, device='cuda', dtype=torch.bfloat16),
                      cond_latents=torch.randn(1, 1, 1, 4, 8, device='cuda', dtype=torch.bfloat16),
                      mask_values=torch.ones(1, 1, 1, 4, 8, device='cuda', dtype=torch.bfloat16),
                      encoder_hidden_states=torch.randn(1, 8, 16, device='cuda', dtype=torch.bfloat16),
                      encoder_attention_mask=torch.ones(1, 8, device='cuda'),
                      timestep=torch.ones(1, device='cuda'), num_frames=1, height=4, width=8,
                      return_dict=False)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            reference = model(**inputs)[0].float()
            model.operator_fusion_decision = decision()
            with operator_fusion_request_scope(model.operator_fusion_decision) as stats:
                actual = model(**inputs)[0].float()
            self.assertEqual(stats.fused_calls, {'rmsnorm_adaln_fast': 4})
            self.assertTrue(actual.isfinite().all())
            self.assertLess(float(((actual-reference).square().mean()/reference.square().mean()).sqrt()), .005)


if __name__ == '__main__':
    unittest.main()
