"""Fused elementwise sites must preserve native normalization and rounding."""
import unittest
from unittest.mock import Mock

import torch

from layers.operator_fusion.qk_rmsnorm_rope import apply_fused_qk_rmsnorm_rope
from layers.operator_fusion.gated_residual import apply_fused_gated_residual
from layers.operator_fusion.registry import OperatorFusionDecision
from models.dits.eraserdit_attention import apply_rotary_emb


def decision(forced=True):
    return OperatorFusionDecision('triton' if forced else 'auto',
                                  ('qk_rmsnorm_rope',), ('qk_rmsnorm_rope',), (), 1, forced)


class FusionPrecisionTests(unittest.TestCase):
    def test_new_site_is_explicit_and_single_gpu_auto_only(self):
        from types import SimpleNamespace
        from layers.operator_fusion.config import normalize_operator_fusion_ops
        from layers.operator_fusion.registry import resolve_operator_fusion_decision
        self.assertNotIn('gated_residual', normalize_operator_fusion_ops(None))
        args = SimpleNamespace(operator_fusion_backend='auto',
                               operator_fusion_ops='gated_residual', sp_degree=2)
        from unittest.mock import patch
        with patch('layers.operator_fusion.registry._triton_is_available', return_value=True):
            d = resolve_operator_fusion_decision(args)
            self.assertEqual(d.effective_ops, ())
            self.assertIn('topology_not_signed', d.fallback_reasons)
            args.sp_degree = 1
            self.assertEqual(resolve_operator_fusion_decision(args).effective_ops,
                             ('gated_residual',))

    def test_gated_residual_cpu_fallback_and_forced_failure(self):
        args = (torch.randn(1, 5, 7), torch.randn(1, 5, 7), torch.randn(1, 1, 7))
        for forced in (False, True):
            d = OperatorFusionDecision('triton' if forced else 'auto',
                                       ('gated_residual',), ('gated_residual',), (), 1, forced)
            if forced:
                with self.assertRaisesRegex(RuntimeError, 'capability check failed'):
                    apply_fused_gated_residual(*args, decision=d)
            else:
                torch.testing.assert_close(apply_fused_gated_residual(*args, decision=d),
                                           args[0] + args[1] * args[2], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_gated_residual_rounding_layouts_and_autograd(self):
        d = OperatorFusionDecision('triton', ('gated_residual',), ('gated_residual',), (), 1, True)
        auto = OperatorFusionDecision('auto', d.requested_ops, d.effective_ops, (), 1, False)
        torch.manual_seed(719)
        for shape in ((1, 1, 2048), (1, 257, 2048), (2, 33, 129)):
            for scale in (0.001, 1., 100.):
                h, u = [torch.randn(shape, device='cuda', dtype=torch.bfloat16) * scale for _ in range(2)]
                # Match the non-contiguous batch stride of ada_values.unbind().
                g = torch.randn(shape[0], 1, 6, shape[-1], device='cuda', dtype=torch.bfloat16)[:, :, 2]
                h_before, u_before, g_before = h.clone(), u.clone(), g.clone()
                actual = apply_fused_gated_residual(h, u, g, decision=d)
                torch.testing.assert_close(actual, h + u * g, rtol=0, atol=0)
                for value, before in ((h, h_before), (u, u_before), (g, g_before)):
                    torch.testing.assert_close(value, before, rtol=0, atol=0)
                hv, uv, gv = h[..., ::2], u[..., ::2], g[..., ::2]
                torch.testing.assert_close(apply_fused_gated_residual(hv, uv, gv, decision=auto),
                                           hv + uv * gv, rtol=0, atol=0)
                with self.assertRaisesRegex(RuntimeError, 'tensor_layout'):
                    apply_fused_gated_residual(hv, uv, gv, decision=d)
        h = torch.randn(1, 3, 7, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        u, g = torch.ones_like(h), torch.ones(1, 1, 7, device='cuda', dtype=torch.bfloat16)
        apply_fused_gated_residual(h, u, g, decision=auto).sum().backward()
        torch.testing.assert_close(h.grad, torch.ones_like(h))
        with self.assertRaisesRegex(RuntimeError, 'autograd'):
            apply_fused_gated_residual(h, u, g, decision=d)

    def test_auto_unsupported_contract_uses_reference(self):
        value = torch.randn(1, 3, 32)
        norm = torch.nn.RMSNorm(32, eps=1e-5)
        expected = (value + 1, value - 1)
        reference = Mock(return_value=expected)
        actual = apply_fused_qk_rmsnorm_rope(
            value, value, norm, norm, (value, value),
            decision=decision(False), reference=reference)
        self.assertIs(actual, expected)
        reference.assert_called_once_with()
        with self.assertRaisesRegex(RuntimeError, 'capability check failed'):
            apply_fused_qk_rmsnorm_rope(value, value, norm, norm, (value, value),
                                       decision=decision(), reference=reference)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_qk_site_is_bit_exact_for_native_norms(self):
        from diffusers.models.normalization import RMSNorm
        torch.manual_seed(72)
        for norm_type in (torch.nn.RMSNorm, RMSNorm):
            qnorm = norm_type(2048, eps=1e-5).cuda().bfloat16()
            knorm = norm_type(2048, eps=1e-5).cuda().bfloat16()
            with torch.no_grad():
                qnorm.weight.uniform_(-2, 2)
                knorm.weight.uniform_(0.2, 1.8)
                for shape, scale in (((1, 1, 2048), 1.), ((1, 257, 2048), .001),
                                     ((2, 33, 2048), 100.)):
                    q, k = [torch.randn(shape, device='cuda', dtype=torch.bfloat16) * scale for _ in range(2)]
                    angles = torch.randn(shape, device='cuda')
                    freqs = (angles.cos(), angles.sin())
                    expected = (apply_rotary_emb(qnorm(q), freqs), apply_rotary_emb(knorm(k), freqs))
                    actual = apply_fused_qk_rmsnorm_rope(
                        q, k, qnorm, knorm, freqs, decision=decision(),
                        reference=Mock(side_effect=AssertionError('unexpected fallback')))
                    for value, ref in zip(actual, expected):
                        torch.testing.assert_close(value, ref, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_full_block_matches_and_offload_hooks_still_run(self):
        from config.server_args import ServerArgs, set_global_server_args
        from layers.operator_fusion.registry import resolve_operator_fusion_decision
        from models.dits.eraserdit_transformer import EraserDiTLTXVideoTransformer3DModel
        from memory.backends.layerwise_offload import LayerwiseOffloadManager
        args = ServerArgs(device='cuda', attention_backend='sdpa')
        set_global_server_args(args)
        torch.manual_seed(31)
        model = EraserDiTLTXVideoTransformer3DModel(
            in_channels=3, out_channels=1, num_attention_heads=32, attention_head_dim=64,
            cross_attention_dim=2048, num_layers=2, caption_channels=16,
        ).cuda().bfloat16().eval()
        inputs = dict(hidden_states=torch.randn(1, 1, 1, 4, 8, device='cuda', dtype=torch.bfloat16),
                      cond_latents=torch.randn(1, 1, 1, 4, 8, device='cuda', dtype=torch.bfloat16),
                      mask_values=torch.ones(1, 1, 1, 4, 8, device='cuda', dtype=torch.bfloat16),
                      encoder_hidden_states=torch.randn(1, 8, 16, device='cuda', dtype=torch.bfloat16),
                      encoder_attention_mask=torch.ones(1, 8, device='cuda'),
                      timestep=torch.ones(1, device='cuda'), num_frames=1, height=4, width=8,
                      return_dict=False)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            expected = model(**inputs)[0]
            args.operator_fusion_backend = 'triton'
            args.operator_fusion_ops = 'qk_rmsnorm_rope,rmsnorm_adaln,gated_residual'
            fused = resolve_operator_fusion_decision(args)
            model.operator_fusion_decision = fused
            for block in model.transformer_blocks:
                block.attn1.processor.operator_fusion_decision = fused
            actual = model(**inputs)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            manager = LayerwiseOffloadManager(model, layers_attr_str='transformer_blocks',
                                               num_layers=2, enabled=True)
            try:
                from cache.eraserdit_text import EraserDiTTextCache
                cache = EraserDiTTextCache()
                for _ in range(2):
                    torch.testing.assert_close(model(**inputs, text_cache=cache)[0], expected, rtol=0, atol=0)
                self.assertEqual(cache.projection_hits, 1)
                self.assertEqual(cache.kv_hits, 2)
                cache.clear()
                manager.release_all()
                self.assertFalse(manager._gpu_layers)
            finally:
                manager.release_all()
                torch.cuda.synchronize()
                manager.remove_forward_hooks()


if __name__ == '__main__':
    unittest.main()
