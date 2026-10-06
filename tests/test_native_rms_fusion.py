"""Native-reduction RMS fusion: exact rounding, streams and safe dispatch."""
import unittest
from itertools import combinations
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from diffusers.models.normalization import RMSNorm

from layers.operator_fusion.config import normalize_operator_fusion_ops
from layers.operator_fusion.registry import resolve_operator_fusion_decision
from layers.operator_fusion.qk_rmsnorm_rope import apply_fused_qk_rmsnorm_rope
from layers.operator_fusion.rmsnorm_adaln import apply_fused_rmsnorm_adaln_native
from layers.operator_fusion.runtime import operator_fusion_request_scope
from models.dits.eraserdit_attention import apply_rotary_emb

OPS = 'qk_rmsnorm_rope_native,rmsnorm_adaln_native'


def decision(backend='triton', sp=1):
    return resolve_operator_fusion_decision(SimpleNamespace(
        operator_fusion_backend=backend, operator_fusion_ops=OPS, sp_degree=sp))


class NativeRMSFusionTests(unittest.TestCase):
    def test_explicit_selection_and_exclusion(self):
        self.assertFalse(set(OPS.split(',')) & set(normalize_operator_fusion_ops(None)))
        for base in ('qk_rmsnorm_rope', 'rmsnorm_adaln'):
            for pair in combinations((base, base + '_fast', base + '_native'), 2):
                with self.assertRaisesRegex(ValueError, 'only one'):
                    normalize_operator_fusion_ops(pair)
        for sp in (1, 2, 4):
            self.assertEqual(set(decision(sp=sp).effective_ops), set(OPS.split(',')))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_exact_outputs_and_stream_ordering(self):
        torch.manual_seed(197)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), torch.inference_mode():
            qn, kn = [RMSNorm(2048, eps=1e-5).cuda().bfloat16() for _ in range(2)]
            norm = RMSNorm(2048, eps=1e-6, elementwise_affine=False).cuda()
            qn.weight.uniform_(-2, 2)
            kn.weight.uniform_(.2, 1.8)
            for tokens, magnitude in [(0, 1.), (1, 0.), (257, .001), (1025, 100.),
                                      (2550, 1.), (5100, 1.), (8160, 1.), (16320, 1.)]:
                with self.subTest(tokens=tokens):
                    q, k = [torch.randn(1, tokens, 2048, device='cuda', dtype=torch.bfloat16)
                            * magnitude for _ in range(2)]
                    angles = torch.randn(q.shape, device='cuda')
                    freqs = (angles.cos(), angles.sin())
                    table = torch.randn(1, 1, 6, 2048, device='cuda', dtype=torch.bfloat16)
                    scale, shift = table[:, :, 1], table[:, :, 4]
                    before = [v.clone() for v in (q, k, scale, shift)]
                    expected = (apply_rotary_emb(qn(q), freqs), apply_rotary_emb(kn(k), freqs),
                                norm(q) * (1 + scale) + shift)
                    d = decision()
                    with operator_fusion_request_scope(d) as stats:
                        actual = apply_fused_qk_rmsnorm_rope(q, k, qn, kn, freqs, decision=d,
                            reference=Mock(side_effect=AssertionError('unexpected fallback')))
                        adaln = apply_fused_rmsnorm_adaln_native(q, scale, shift, norm, decision=d)
                    self.assertEqual(stats.fused_calls, dict.fromkeys(OPS.split(','), 1))
                    for got, ref in zip((*actual, adaln), expected):
                        torch.testing.assert_close(got, ref, rtol=0, atol=0)
                    for got, ref in zip((q, k, scale, shift), before):
                        torch.testing.assert_close(got, ref, rtol=0, atol=0)
        stream.synchronize()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_fallback_preserves_autograd(self):
        q = torch.randn(1, 3, 2048, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        norm = RMSNorm(2048, eps=1e-6, elementwise_affine=False).cuda()
        scale, shift = [torch.randn(1, 1, 2048, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
        got = apply_fused_rmsnorm_adaln_native(q, scale, shift, norm, decision=decision('auto'))
        torch.testing.assert_close(got, norm(q) * (1 + scale) + shift, rtol=0, atol=0)
        got.sum().backward()
        self.assertTrue(q.grad.isfinite().all())
        with self.assertRaisesRegex(RuntimeError, 'autograd'):
            apply_fused_rmsnorm_adaln_native(q, scale, shift, norm, decision=decision())
        qn = RMSNorm(2048, eps=1e-5).cuda().bfloat16()
        freqs = (torch.ones_like(q, dtype=torch.float32), torch.zeros_like(q, dtype=torch.float32))
        ref = Mock(return_value=(q, q))
        apply_fused_qk_rmsnorm_rope(q, q, qn, qn, freqs, decision=decision('auto'), reference=ref)
        ref.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, 'autograd'):
            apply_fused_qk_rmsnorm_rope(q, q, qn, qn, freqs, decision=decision(), reference=ref)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_custom_forward_falls_back(self):
        q = torch.randn(1, 3, 2048, device='cuda', dtype=torch.bfloat16)
        norm = RMSNorm(2048, eps=1e-6, elementwise_affine=False).cuda()
        norm.forward = lambda x: x + 2
        scale, shift = torch.zeros_like(q[:, :1]), torch.ones_like(q[:, :1])
        with torch.inference_mode():
            got = apply_fused_rmsnorm_adaln_native(q, scale, shift, norm, decision=decision('auto'))
            torch.testing.assert_close(got, norm(q) * (1 + scale) + shift, rtol=0, atol=0)
            with self.assertRaisesRegex(RuntimeError, 'norm_implementation'):
                apply_fused_rmsnorm_adaln_native(q, scale, shift, norm, decision=decision())


if __name__ == '__main__':
    unittest.main()
