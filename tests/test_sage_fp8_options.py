import os
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch

from config.server_args import ServerArgs, get_global_server_args, set_global_server_args
from layers.attention.backends.sage_fp8 import SageFP8AttentionImpl
from layers.attention.backends.attention_backend import AttentionBackendEnum, AttentionMetadata


class SageFP8OptionTests(unittest.TestCase):
    def test_configuration_rejects_ignored_or_invalid_options(self):
        self.assertEqual(ServerArgs().sage_fp8_accum_dtype, 'fp32+fp32')
        for kwargs in [dict(sage_fp8_accum_dtype='bad'),
                       dict(sage_fp8_qk_quant_gran='bad'),
                       dict(sage_fp8_accum_dtype='fp32+fp16')]:
            with self.assertRaises(ValueError):
                ServerArgs(**kwargs)

    def test_processor_forwards_and_reports_selected_options(self):
        from models.dits.eraserdit_attention import EraserDiTAttentionProcessor
        previous = get_global_server_args()
        self.addCleanup(set_global_server_args, previous)
        set_global_server_args(ServerArgs(attention_backend='sage_fp8',
            sage_fp8_accum_dtype='fp32+fp16', sage_fp8_qk_quant_gran='per_warp'))
        selection = SimpleNamespace(selected=AttentionBackendEnum.SAGE_FP8, fallback_reasons=())
        with patch('models.dits.eraserdit_attention.resolve_attention_backend', return_value=selection), \
             patch('layers.attention.backends.sage_fp8._load_sage_fp8_kernel',
                   return_value=(lambda *a, **k: None, None, None)):
            processor = EraserDiTAttentionProcessor()
            report = processor.preflight_self_attention_backend(device=torch.device('cpu'), dtype=torch.bfloat16)
            self.assertEqual(report['pv_accum_dtype'], 'fp32+fp16')
            self.assertEqual(report['qk_quant_granularity'], 'per_warp')
            _, impl = processor._resolve_self_attention(torch.zeros(1, 17, 2, 64), has_attn_mask=False)
            self.assertEqual(impl.pv_accum_dtype, 'fp32+fp16')
            self.assertEqual(impl.qk_quant_gran, 'per_warp')

    def test_ffn_up_selects_only_expansion(self):
        from models.dits.eraserdit_quantization import selected_names
        names = selected_names(SimpleNamespace(transformer_blocks=[None]*28), 'ffn_up')
        self.assertEqual(len(names), 28)
        self.assertTrue(all(n.endswith('ff.net.0.proj') for n in names))


@unittest.skipUnless(os.environ.get('ERASERDIT_TEST_SAGE_FP8') == '1', 'SM89 extension opt-in')
class SageFP8KernelTests(unittest.TestCase):
    def test_variants_on_ragged_and_aligned_sequences(self):
        torch.manual_seed(42)
        with torch.inference_mode():
            for n in (127, 256):
                q, k, v = [torch.randn(1, n, 2, 64, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
                ref = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).float()
                for accum in ('fp32+fp32', 'fp32+fp16'):
                    for gran in ('per_thread', 'per_warp'):
                        impl = SageFP8AttentionImpl(2, 64, 64**-.5,
                            pv_accum_dtype=accum, qk_quant_gran=gran)
                        out = impl.forward(q, k, v, AttentionMetadata())
                        self.assertTrue(out.isfinite().all())
                        self.assertEqual(out.dtype, torch.bfloat16)
                        self.assertEqual(out.shape, q.shape)
                        rmse = ((out.float()-ref).square().mean()/ref.square().mean()).sqrt().item()
                        self.assertLess(rmse, .07)
                        self.assertEqual(impl.report()['call_count'], 1)
                        with self.assertRaises(RuntimeError):
                            impl.forward(q, k, v, AttentionMetadata(attn_mask=torch.ones(1, n, n, device='cuda')))
