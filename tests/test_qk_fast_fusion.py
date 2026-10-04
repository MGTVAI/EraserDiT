"""Approximate full Q/K fusion: bounded error, explicit selection and fallback."""
import unittest
from unittest.mock import Mock
from types import SimpleNamespace

import torch
from diffusers.models.normalization import RMSNorm

from layers.operator_fusion.config import normalize_operator_fusion_ops
from layers.operator_fusion.registry import resolve_operator_fusion_decision
from layers.operator_fusion.qk_rmsnorm_rope import apply_fused_qk_rmsnorm_rope
from layers.operator_fusion.runtime import operator_fusion_request_scope
from models.dits.eraserdit_attention import apply_rotary_emb


def decision(backend='triton'):
    return resolve_operator_fusion_decision(SimpleNamespace(
        operator_fusion_backend=backend,
        operator_fusion_ops='qk_rmsnorm_rope_fast', sp_degree=1,
    ))


class QKFastFusionTests(unittest.TestCase):
    def test_explicit_selection_and_conflict(self):
        self.assertNotIn('qk_rmsnorm_rope_fast', normalize_operator_fusion_ops(None))
        self.assertEqual(normalize_operator_fusion_ops('qk_rmsnorm_rope_fast'),
                         ('qk_rmsnorm_rope_fast',))
        with self.assertRaisesRegex(ValueError, 'only one'):
            normalize_operator_fusion_ops('qk_rmsnorm_rope,qk_rmsnorm_rope_fast')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_error_and_telemetry(self):
        torch.manual_seed(813)
        qn, kn = [RMSNorm(2048, eps=1e-5).cuda().bfloat16() for _ in range(2)]
        with torch.inference_mode():
            qn.weight.uniform_(-2, 2)
            kn.weight.uniform_(.2, 1.8)
            for shape, scale in [((1, 1, 2048), 0.), ((1, 257, 2048), .001),
                                 ((2, 33, 2048), 100.)]:
                q, k = [torch.randn(shape, device='cuda', dtype=torch.bfloat16) * scale for _ in range(2)]
                angles = torch.randn(shape, device='cuda')
                freqs = (angles.cos(), angles.sin())
                expected = (apply_rotary_emb(qn(q), freqs), apply_rotary_emb(kn(k), freqs))
                before = (q.clone(), k.clone())
                d = decision()
                with operator_fusion_request_scope(d) as stats:
                    actual = apply_fused_qk_rmsnorm_rope(
                        q, k, qn, kn, freqs, decision=d,
                        reference=Mock(side_effect=AssertionError('unexpected fallback')))
                self.assertEqual(stats.fused_calls, {'qk_rmsnorm_rope_fast': 1})
                for got, ref in zip(actual, expected):
                    self.assertTrue(got.isfinite().all())
                    self.assertEqual(got.shape, ref.shape)
                    self.assertEqual(got.dtype, ref.dtype)
                    relative_rmse = ((got.float() - ref.float()).square().mean()
                                     / ref.float().square().mean().clamp_min(1e-12)).sqrt()
                    self.assertLess(float(relative_rmse), .001)
                for got, ref in zip((q, k), before):
                    torch.testing.assert_close(got, ref, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_unsupported_norm_and_autograd(self):
        q = torch.randn(1, 3, 2048, device='cuda', dtype=torch.bfloat16)
        freqs = (torch.ones_like(q, dtype=torch.float32), torch.zeros_like(q, dtype=torch.float32))
        for norm, reason in [(torch.nn.RMSNorm(2048, eps=1e-5).cuda().bfloat16(), 'norm_implementation'),
                             (RMSNorm(2048, eps=1e-5).cuda().bfloat16(), 'autograd')]:
            reference = Mock(return_value=(q, q))
            result = apply_fused_qk_rmsnorm_rope(q, q, norm, norm, freqs,
                                               decision=decision('auto'), reference=reference)
            self.assertIs(result[0], q)
            reference.assert_called_once()
            with self.assertRaisesRegex(RuntimeError, reason):
                apply_fused_qk_rmsnorm_rope(q, q, norm, norm, freqs,
                                          decision=decision(), reference=reference)


if __name__ == '__main__':
    unittest.main()
