"""Policy and numerical regression checks for opt-in Q6 extensions."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from config.q6_extensions import q6_extensions, validate_q6_extensions


class PolicyTests(unittest.TestCase):
    def setUp(self):
        q6_extensions.cache_clear()

    def tearDown(self):
        q6_extensions.cache_clear()

    def test_defaults_preserve_legacy(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(q6_extensions().active)

    def test_incompatible_execution_rejected(self):
        args = SimpleNamespace(
            pipeline_config=SimpleNamespace(quantization_scope='ffn', dit_parallel_backend='peer'),
            attention_backend='sage_fp8', transformer_quantization='fp8_w8a8_static')
        with patch.dict(os.environ, {'MGERASE_Q6_SAGE_PERIPHERAL': '1'}, clear=True):
            validate_q6_extensions(args)
            args.enable_torch_compile = True
            with self.assertRaises(ValueError):
                validate_q6_extensions(args)

    def test_calibration_rejects_partial_or_nonfinite_scales(self):
        from layers.quantization.qkv_fp8 import read_calibration
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'scales.json'
            path.write_text(json.dumps(dict(completed=True, all_holdout_within_range=True,
                layers=[dict(block=i, scale=.02) for i in range(27)])))
            with self.assertRaises(ValueError):
                read_calibration(path)
            data = dict(completed=True, all_holdout_within_range=True,
                        layers=[dict(block=i, scale=.02) for i in range(28)])
            data['layers'][13]['scale'] = float('nan')
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                read_calibration(path)

    def test_fill_flag_restored_after_failure(self):
        from layers.audited_allocations import _allocation_scope
        for previous in (True, False):
            with patch.dict(os.environ, {'MGERASE_Q6_TARGETED_FILL': '1'}, clear=True):
                q6_extensions.cache_clear()
                old = torch.utils.deterministic.fill_uninitialized_memory
                torch.utils.deterministic.fill_uninitialized_memory = previous
                try:
                    with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, 'injected'):
                        with _allocation_scope('exception_test', True):
                            self.assertFalse(torch.utils.deterministic.fill_uninitialized_memory)
                            raise RuntimeError('injected')
                    self.assertEqual(torch.utils.deterministic.fill_uninitialized_memory, previous)
                finally:
                    torch.utils.deterministic.fill_uninitialized_memory = old


def same(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@unittest.skipUnless(os.environ.get('ERASERDIT_TEST_Q6') == '1', 'single SM89 GPU opt-in')
class KernelTests(unittest.TestCase):
    def setUp(self):
        q6_extensions.cache_clear()
        self.old_deterministic = torch.are_deterministic_algorithms_enabled()
        self.old_fill = torch.utils.deterministic.fill_uninitialized_memory
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True

    def tearDown(self):
        torch.utils.deterministic.fill_uninitialized_memory = self.old_fill
        torch.use_deterministic_algorithms(self.old_deterministic)
        q6_extensions.cache_clear()

    @torch.inference_mode()
    def test_cross_query_matches_native_and_fill_toggle(self):
        from diffusers.models.normalization import RMSNorm
        from layers.operator_fusion.triton.cross_q_rmsnorm import triton_cross_q_rmsnorm_native
        torch.manual_seed(11)
        norm = RMSNorm(2048, eps=1e-5, elementwise_affine=True).cuda().bfloat16().eval()
        for length in (1, 32, 32640):
            value = torch.randn(1, length, 2048, device='cuda', dtype=torch.bfloat16)
            reference = norm(value)
            for enabled in ('0', '1'):
                with patch.dict(os.environ, {'MGERASE_Q6_TARGETED_FILL': enabled}):
                    q6_extensions.cache_clear()
                    result = triton_cross_q_rmsnorm_native(value, norm.weight, norm.eps)
                    self.assertTrue(same(result, reference))
                    self.assertTrue(torch.utils.deterministic.fill_uninitialized_memory)

    @torch.inference_mode()
    def test_sage_preserves_native_output_with_poisoned_factories(self):
        import layers.audited_allocations as allocation
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
        from layers.attention.backends.sage_peripheral import SagePeripheral
        torch.manual_seed(31)
        q, k, v = [torch.randn(1, 32640, 32, 64, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
        reference = sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout='NHD',
            qk_quant_gran='per_thread', pv_accum_dtype='fp32+fp32', sm_scale=.125,
            smooth_k=True, smooth_v=False, is_causal=False, return_lse=False)
        operator = SagePeripheral()
        original_empty = allocation.empty
        for mode in ('keep', 'targeted', 'zero', 'nan'):
            def poisoned(*args, **kwargs):
                result = original_empty(*args, **kwargs)
                if mode == 'zero':
                    result.view(torch.uint8).zero_()
                if mode == 'nan':
                    if result.dtype == torch.float8_e4m3fn or not result.is_floating_point():
                        result.view(torch.uint8).fill_(127)
                    else:
                        result.fill_(float('nan'))
                return result
            with patch.dict(os.environ, {'MGERASE_Q6_TARGETED_FILL': '0' if mode == 'keep' else '1'}):
                q6_extensions.cache_clear()
                with patch.object(allocation, 'empty', poisoned):
                    actual = operator(q, k, v, .125)
                self.assertTrue(same(actual, reference), mode)
                self.assertTrue(torch.utils.deterministic.fill_uninitialized_memory)

    @torch.inference_mode()
    def test_qkv_matches_explicit_fp8_reference_and_offload_reload(self):
        from torch import nn
        from layers.quantization.qkv_fp8 import SharedQKV
        from layers.quantization.eraserdit_fp8 import _quant_fp8_tensor
        import triton
        torch.manual_seed(41)
        attn = nn.Module()
        for name in ('q', 'k', 'v'):
            setattr(attn, 'to_'+name, nn.Linear(2048, 2048, dtype=torch.bfloat16))
        group = SharedQKV(attn, .02, 'cuda').eval().cuda()
        value = torch.randn(1, 64, 2048, device='cuda', dtype=torch.bfloat16)
        quant = torch.empty_like(value, dtype=torch.float8_e4m3fn)
        _quant_fp8_tensor[(triton.cdiv(value.numel(),8192),)](
            value, quant, group.activation_scale, value.numel(), 8192, None, False, num_warps=8)
        expected = [torch._scaled_mm(quant.reshape(-1,2048), getattr(group, name+'_weight').t(),
            scale_a=group.activation_scale, scale_b=getattr(group, name+'_scale'),
            bias=getattr(group, name+'_bias'), out_dtype=torch.bfloat16, use_fast_accum=True).reshape_as(value)
            for name in ('q', 'k', 'v')]
        for enabled in ('0', '1'):
            with patch.dict(os.environ, {'MGERASE_Q6_TARGETED_FILL': enabled}):
                q6_extensions.cache_clear()
                actual = group(value)
                self.assertTrue(all(same(a,b) for a,b in zip(actual,expected)))
        held = list(group.buffers())
        group.cpu().cuda()
        self.assertTrue(all(same(a,b) for a,b in zip(group(value),expected)))
        del held

    @torch.inference_mode()
    def test_ffn_bf16_entry_matches_explicit_native_epilogue_and_reload(self):
        from torch import nn
        from layers.quantization.eraserdit_fp8 import NativeStaticFp8Linear, _quantize_activation
        from layers.quantization.gelu import ProjectionBeforeFusedGelu, make_gelu_lut
        from layers.quantization.ffn_epilogue import EpilogueFeedForward, Workspace, library
        torch.manual_seed(52)
        up = NativeStaticFp8Linear.from_linear(nn.Linear(2048,8192,device='cuda',dtype=torch.bfloat16))
        down = NativeStaticFp8Linear.from_linear(nn.Linear(8192,2048,device='cuda',dtype=torch.bfloat16))
        down.gelu_lut = make_gelu_lut('cuda')
        original = nn.Module()
        original.net = nn.ModuleList([ProjectionBeforeFusedGelu(up), nn.Identity(), down])
        module = EpilogueFeedForward(original, Workspace()).eval()
        value = torch.randn(1,32640,2048,device='cuda',dtype=torch.bfloat16)
        with patch.dict(os.environ, {'MGERASE_Q6_TARGETED_FILL': '1'}):
            q6_extensions.cache_clear()
            x = _quantize_activation(value.reshape(-1,2048), None, True, .125)[0]
            q = torch.empty((32640,8192),device='cuda',dtype=torch.float8_e4m3fn)
            handle = module._plan(x)
            workspace = module.workspace.get(x.device)
            self.assertEqual(library().lt_run(handle,0,up.weight_fp8.data_ptr(),x.data_ptr(),q.data_ptr(),
                workspace.data_ptr(),torch.cuda.current_stream().cuda_stream,1),0)
            expected = torch._scaled_mm(q,down.weight_fp8.t(),scale_a=down.activation_scale,
                scale_b=down.weight_scale,bias=down.bias,out_dtype=torch.bfloat16,use_fast_accum=True).reshape_as(value)
            self.assertTrue(same(module(value), expected))
            created = module.plans_created
            held = list(module.buffers())
            module.cpu().cuda()
            self.assertTrue(same(module(value),expected))
            self.assertEqual(module.plans_created,created+1)
            self.assertTrue(torch.utils.deterministic.fill_uninitialized_memory)
            del held


if __name__ == '__main__':
    unittest.main()
