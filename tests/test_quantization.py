import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from models.dits.eraserdit_quantization import selected_names, validate_quantization, runtime_report


class QuantizationPolicyTests(unittest.TestCase):
    def test_unquantized_report_does_not_require_int8_backend(self):
        with patch.dict(sys.modules, {
            'layers.quantization.eraserdit_int8': None,
            'triton': None,
        }):
            self.assertEqual(runtime_report(nn.Linear(2, 2)),
                             dict(mode='none', runtime_call_count=0))

    def test_scopes_exclude_io_and_conditioning(self):
        model = SimpleNamespace(transformer_blocks=[None]*28)
        names = selected_names(model,'blocks')
        self.assertEqual(len(names),224)
        self.assertEqual(len(selected_names(model,'ffn')),56)
        self.assertTrue(all(n.startswith('transformer_blocks.') for n in names))
        self.assertFalse(any('.attn2.to_k' in n or '.attn2.to_v' in n for n in names))

    def test_reject_incompatible_modes(self):
        c=SimpleNamespace(sp_degree=1,cfg_degree=1,vae_degree=1,cfg_parallel_device=None)
        args=SimpleNamespace(transformer_quantization='int8_w8a8_native',pipeline_config=c,
            enable_torch_compile=False,operator_fusion_backend='disabled')
        validate_quantization(args)
        c.sp_degree=2
        validate_quantization(args)
        args.enable_torch_compile = True
        validate_quantization(args,SimpleNamespace(transformer_cache_mode="teacache",cache_text_projections=False))
        args.operator_fusion_backend = "triton"
        with self.assertRaises(ValueError):
            validate_quantization(args)
        c.sp_degree = 1
        validate_quantization(args)


@unittest.skipUnless(os.environ.get('ERASERDIT_TEST_INT8')=='1','single GPU opt-in')
class Int8KernelTests(unittest.TestCase):
    def test_offloaded_weights_use_the_same_conversion_as_resident(self):
        from layers.quantization.eraserdit_int8 import NativeInt8Linear
        from layers.quantization.eraserdit_fp8 import NativeFp8Linear, NativeTensorwiseFp8Linear, NativeStaticFp8Linear
        torch.manual_seed(71)
        source = nn.Linear(256, 128, bias=True, dtype=torch.bfloat16)
        classes = [NativeInt8Linear]
        if torch.cuda.get_device_capability() >= (8, 9):
            classes += [NativeFp8Linear, NativeTensorwiseFp8Linear, NativeStaticFp8Linear]
        for cls in classes:
            source.cpu()
            offloaded = cls.from_linear(source, execution_device='cuda').cpu()
            self.assertEqual(source.weight.device.type, 'cpu')
            resident = cls.from_linear(source.cuda()).cpu()
            for name, actual in offloaded.named_buffers():
                expected = dict(resident.named_buffers())[name]
                # Compare FP8 storage bits as well as scales and bias.
                self.assertTrue(torch.equal(actual.contiguous().reshape(-1).view(torch.uint8),
                                            expected.contiguous().reshape(-1).view(torch.uint8)), name)

    def test_static_fp8_saturation_scale_and_numerical_reference(self):
        from layers.quantization.eraserdit_fp8 import NativeStaticFp8Linear, _quantize_activation
        from layers.quantization.gelu import make_gelu_lut
        if torch.cuda.get_device_capability() < (8, 9):
            self.skipTest('FP8 requires SM89 or newer')
        torch.manual_seed(67)
        lut = make_gelu_lut('cuda')
        for bias in (False, True):
            linear = NativeStaticFp8Linear.from_linear(nn.Linear(256, 128, bias=bias,
                device='cuda', dtype=torch.bfloat16))
            self.assertEqual(linear.activation_scale.item(), .125)
            self.assertTrue(linear.use_fast_accum)
            for rows in (0, 1, 17, 129):
                x = torch.randn(rows, 512, device='cuda', dtype=torch.bfloat16)[:, ::2] * 32
                if rows:
                    x[0, :7] = torch.tensor([-80., -56., -55., 0., 55., 56., 80.], device='cuda')
                for fused in (False, True):
                    linear.gelu_lut = lut if fused else None
                    actual = linear(x)
                    self.assertEqual(actual.shape, (rows, 128))
                    if not rows:
                        continue
                    y = torch.nn.functional.gelu(x, approximate='tanh') if fused else x
                    expected_q = (y.float() * 8).clamp(-448, 448).to(torch.float8_e4m3fn)
                    quant, scale = _quantize_activation(x.contiguous(), linear.gelu_lut, True, .125)
                    self.assertIsNone(scale)
                    torch.testing.assert_close(quant.float(), expected_q.float(), rtol=0, atol=0)
                    expected = torch._scaled_mm(expected_q, linear.weight_fp8.t(),
                        scale_a=linear.activation_scale, scale_b=linear.weight_scale, bias=linear.bias,
                        out_dtype=torch.bfloat16, use_fast_accum=True)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertTrue(actual.isfinite().all())

    def test_fp8_gelu_fusion_matches_separate_quantization(self):
        from layers.quantization.gelu import make_gelu_lut
        from layers.quantization.eraserdit_fp8 import (
            NativeFp8Linear, NativeTensorwiseFp8Linear, _quant_fp8_rows, quantize_fp8_tensor)
        import triton
        if torch.cuda.get_device_capability() < (8, 9):
            self.skipTest('FP8 requires SM89 or newer')
        torch.manual_seed(61)
        lut = make_gelu_lut('cuda')
        values = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        values[~values.isfinite()] = 0
        for x in (values.reshape(256, 256),
                  torch.randn(129, 259, device='cuda', dtype=torch.bfloat16),
                  torch.zeros(17, 256, device='cuda', dtype=torch.bfloat16)):
            y = torch.nn.functional.gelu(x, approximate='tanh')
            aq, sa = quantize_fp8_tensor(x, lut)
            eq, se = quantize_fp8_tensor(y)
            torch.testing.assert_close(aq.view(torch.uint8), eq.view(torch.uint8), rtol=0, atol=0)
            torch.testing.assert_close(sa, se, rtol=0, atol=0, equal_nan=True)
            rows, inner = x.shape
            sa, se = torch.empty(rows, device='cuda'), torch.empty(rows, device='cuda')
            _quant_fp8_rows[(rows,)](x, aq, sa, inner, triton.next_power_of_2(inner),
                                    lut, True, num_warps=8)
            _quant_fp8_rows[(rows,)](y, eq, se, inner, triton.next_power_of_2(inner), num_warps=8)
            torch.testing.assert_close(aq.view(torch.uint8), eq.view(torch.uint8), rtol=0, atol=0)
            torch.testing.assert_close(sa, se, rtol=0, atol=0, equal_nan=True)
        for linear_class in (NativeFp8Linear, NativeTensorwiseFp8Linear):
            for bias in (False, True):
                linear = linear_class.from_linear(nn.Linear(256, 128, bias=bias, device='cuda', dtype=torch.bfloat16))
                for rows in (0, 1, 17, 129):
                    x = torch.randn(2, rows, 512, device='cuda', dtype=torch.bfloat16)[..., ::2]
                    if rows:
                        x[:, 0] = 0
                    for multiplier in (.125, 1., 32.):
                        value = x * multiplier
                        linear.gelu_lut = None
                        expected = linear(torch.nn.functional.gelu(value, approximate='tanh'))
                        linear.gelu_lut = lut
                        torch.testing.assert_close(linear(value), expected, rtol=0, atol=0)

    def test_fused_gelu_quantization_matches_separate_path(self):
        from layers.quantization.gelu import make_gelu_lut, _gelu_quant_rows
        from layers.quantization.eraserdit_int8 import NativeInt8Linear, _quant_rows
        torch.manual_seed(53)
        lut = make_gelu_lut('cuda')
        # Every finite BF16 input, plus typical activation magnitudes, zeros,
        # non-contiguous inputs and the native GEMM's row padding boundary.
        values = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        values[~values.isfinite()] = 0
        x = values.reshape(256, 256)
        actual_q, expected_q = torch.empty_like(x, dtype=torch.int8), torch.empty_like(x, dtype=torch.int8)
        actual_s, expected_s = torch.empty(256, device='cuda'), torch.empty(256, device='cuda')
        _gelu_quant_rows[(256,)](x, lut, actual_q, actual_s, 256, 256, num_warps=8)
        y = torch.nn.functional.gelu(x, approximate='tanh')
        _quant_rows[(256,)](y, expected_q, expected_s, 256, 256, num_warps=8)
        torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(actual_s, expected_s, rtol=0, atol=0)
        for bias in (False, True):
            linear = NativeInt8Linear.from_linear(nn.Linear(256, 128, bias=bias, device='cuda', dtype=torch.bfloat16))
            for rows in (0, 1, 17, 129):
                x = torch.randn(2, rows, 512, device='cuda', dtype=torch.bfloat16)[..., ::2]
                if rows:
                    x[:, 0] = 0
                for multiplier in (.125, 1., 32.):
                    value = x * multiplier
                    linear.gelu_lut = None
                    expected = linear(torch.nn.functional.gelu(value, approximate='tanh'))
                    linear.gelu_lut = lut
                    actual = linear(value)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_gelu_fusion_cpu_conversion_offload_and_compile(self):
        self._check_gelu_fusion_cpu_conversion_offload_and_compile('int8_w8a8_native')

    def test_fp8_gelu_fusion_cpu_conversion_offload_and_compile(self):
        if torch.cuda.get_device_capability() < (8, 9):
            self.skipTest('FP8 requires SM89 or newer')
        for mode in ('fp8_w8a8_native', 'fp8_w8a8_tensorwise', 'fp8_w8a8_static'):
            with self.subTest(mode=mode):
                self._check_gelu_fusion_cpu_conversion_offload_and_compile(mode)

    def _check_gelu_fusion_cpu_conversion_offload_and_compile(self, mode):
        from layers.mlp import FeedForward
        from memory.backends.layerwise_offload import LayerwiseOffloadManager
        from models.dits.eraserdit_quantization import quantize_transformer

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.ff = FeedForward(32, activation_fn='gelu-approximate')

            def forward(self, value):
                return self.ff(value)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([Block(), Block()])

            def forward(self, value):
                for block in self.transformer_blocks:
                    value = block(value)
                return value

        torch.manual_seed(59)
        model = Model().bfloat16().eval()
        originals = [block.ff.net[0] for block in model.transformer_blocks]
        report = quantize_transformer(model, 'ffn', execution_device='cuda', mode=mode)
        self.assertEqual(report['fused_gelu_count'], 2)
        luts = [block.ff.net[2].gelu_lut for block in model.transformer_blocks]
        self.assertTrue(all(lut.device.type == 'cpu' for lut in luts))
        self.assertNotEqual(luts[0].data_ptr(), luts[1].data_ptr())
        model.cuda()
        value = torch.randn(1, 17, 32, device='cuda', dtype=torch.bfloat16)
        with torch.inference_mode():
            expected = model(value)
            # Independent reference uses the original PyTorch activation.
            for i, block in enumerate(model.transformer_blocks):
                net = block.ff.net
                originals[i].proj = net[0].proj
                net[0], originals[i] = originals[i], net[0]
                net[2].gelu_lut = None
            torch.testing.assert_close(model(value), expected, rtol=0, atol=0)
            for i, block in enumerate(model.transformer_blocks):
                block.ff.net[0] = originals[i]
                block.ff.net[2].gelu_lut = luts[i].cuda()
            # Compiled FFN remains compatible with changing buffer residency.
            for block in model.transformer_blocks:
                block.ff.forward = torch.compile(block.ff.forward, fullgraph=True)
            manager = LayerwiseOffloadManager(model, layers_attr_str='transformer_blocks',
                num_layers=2, enabled=True, prefetch_size=1)
            try:
                for _ in range(2):
                    manager.prepare_for_next_req(non_blocking=False)
                    torch.testing.assert_close(model(value), expected, rtol=0, atol=0)
                    manager.release_all()
            finally:
                manager.release_all()
                manager.remove_forward_hooks()

    def test_gelu_fusion_retains_other_activations_and_dropout(self):
        from layers.mlp import FeedForward
        from models.dits.eraserdit_quantization import quantize_transformer
        for activation, dropout, fused in [('gelu', 0., 0), ('gelu-approximate', .5, 0),
                                           ('gelu-approximate', 0., 1)]:
            model = nn.Module()
            block = nn.Module()
            block.ff = FeedForward(32, activation_fn=activation).bfloat16()
            block.ff.net[1] = nn.Dropout(dropout)
            original = block.ff.net[0]
            model.transformer_blocks = nn.ModuleList([block])
            report = quantize_transformer(model, 'ffn', execution_device='cuda')
            self.assertEqual(report['fused_gelu_count'], fused)
            if not fused:
                self.assertIs(block.ff.net[0], original)

    def test_fused_expansion_matches_native_epilogue(self):
        from layers.quantization.gemm import int8_scaled_gemm
        from layers.quantization.eraserdit_int8 import _epilogue
        import triton
        torch.manual_seed(7)
        # Ragged rows, zero activation scale, and nontrivial scales.
        a = torch.randint(-16, 16, (1031, 256), device='cuda').to(torch.int8)
        w = torch.randint(-16, 16, (1024, 256), device='cuda').to(torch.int8)
        sa = torch.rand(1031, device='cuda') / 127
        sa[0] = 0
        sw = torch.rand(1024, device='cuda') / 127
        bias = torch.randn(1024, device='cuda', dtype=torch.bfloat16)
        accum = torch._int_mm(torch.nn.functional.pad(a, (0, 0, 0, 1)), w.t())[:1031]
        expected = torch.empty((1031, 1024), device='cuda', dtype=torch.bfloat16)
        _epilogue[(triton.cdiv(expected.numel(), 1024),)](
            accum, sa, sw, bias, expected, 1024, expected.numel(), True, 1024)
        actual = int8_scaled_gemm(a, w, sa, sw, bias)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_tensorwise_fp8_dynamic_scale_and_reference(self):
        from layers.quantization.eraserdit_fp8 import NativeTensorwiseFp8Linear, quantize_fp8_tensor
        if torch.cuda.get_device_capability() < (8, 9):
            self.skipTest('FP8 requires SM89 or newer')
        original_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, 'allow_tf32', original_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(43)
        for bias in (False, True):
            linear = nn.Linear(256, 128, bias=bias, device='cuda', dtype=torch.bfloat16)
            quant = NativeTensorwiseFp8Linear.from_linear(linear)
            self.assertEqual(quant.weight_scale.ndim, 0)
            for rows in (0, 1, 17, 129):
                # Noncontiguous source, ragged reduction tail, and multiple reductions.
                x = torch.randn(rows, 512, device='cuda', dtype=torch.bfloat16)[:, ::2]
                for multiplier in (0., .125, 32.):
                    value = x * multiplier
                    actual = quant(value)
                    self.assertEqual(actual.shape, (rows, 128))
                    if not rows:
                        continue
                    scale = value.float().abs().amax() / 448
                    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
                    xq = (value.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
                    aq, sa = quantize_fp8_tensor(value.contiguous())
                    torch.testing.assert_close(aq.float(), xq.float(), rtol=0, atol=0)
                    torch.testing.assert_close(sa, scale, rtol=0, atol=0)
                    expected = (xq.float() @ quant.weight_fp8.float().t()) * scale * quant.weight_scale
                    if bias:
                        expected += quant.bias.float()
                    torch.testing.assert_close(actual, expected.to(torch.bfloat16),
                                               rtol=.008, atol=.0078125)
                    self.assertTrue(torch.isfinite(actual).all())
            self.assertEqual(quant.calls, 9)

    def test_tuned_int8_tile_matches_native_on_ragged_large_expansion(self):
        from layers.quantization.gemm import int8_scaled_gemm
        from layers.quantization.eraserdit_int8 import _epilogue
        import triton
        torch.manual_seed(17)
        for rows in (8193, 32640):
            inner, columns = 2048, 8192
            a = torch.randint(-127, 128, (rows, inner), device='cuda', dtype=torch.int8)
            w = torch.randint(-127, 128, (columns, inner), device='cuda', dtype=torch.int8)
            sa = torch.rand(rows, device='cuda') / 127
            sw = torch.rand(columns, device='cuda') / 127
            bias = torch.randn(columns, device='cuda', dtype=torch.bfloat16)
            accum = torch._int_mm(torch.nn.functional.pad(a, (0, 0, 0, (-rows) % 8)), w.t())[:rows]
            expected = torch.empty((rows, columns), device='cuda', dtype=torch.bfloat16)
            _epilogue[(triton.cdiv(expected.numel(), 1024),)](
                accum, sa, sw, bias, expected, columns, expected.numel(), True, 1024)
            actual = int8_scaled_gemm(a, w, sa, sw, bias)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_fp8_matches_dequantized_reference(self):
        from layers.quantization.eraserdit_fp8 import NativeFp8Linear, _quant_fp8_rows
        if torch.cuda.get_device_capability() < (8, 9):
            self.skipTest('FP8 requires SM89 or newer')
        original_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, 'allow_tf32', original_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(42)
        for bias in (False, True):
            linear = nn.Linear(256, 128, bias=bias, device='cuda', dtype=torch.bfloat16)
            quant = NativeFp8Linear.from_linear(linear)
            for rows in (0, 1, 17, 128):
                x = torch.randn(rows, 256, device='cuda', dtype=torch.bfloat16)
                if rows:
                    x[0] = 0
                actual = quant(x)
                scale = x.float().abs().amax(-1) / 448
                scale = torch.where(scale > 0, scale, torch.ones_like(scale))
                xq = (x.float() / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
                if rows:
                    # Regression for SM89 FP32 -> FP16 -> FP8 double rounding.
                    actual_q = torch.empty_like(xq)
                    actual_scale = torch.empty_like(scale)
                    _quant_fp8_rows[(rows,)](x, actual_q, actual_scale, 256, 256, num_warps=8)
                    torch.testing.assert_close(actual_q.float(), xq.float(), rtol=0, atol=0)
                    torch.testing.assert_close(actual_scale, scale, rtol=0, atol=0)
                expected = (xq.float() @ quant.weight_fp8.float().t()) * scale[:, None] * quant.weight_scale
                if bias:
                    expected += quant.bias.float()
                torch.testing.assert_close(actual, expected.to(torch.bfloat16), rtol=0, atol=.0078125)
                self.assertTrue(torch.isfinite(actual).all())
            self.assertEqual(quant.calls, 3)

    def test_real_int8_gemm_matches_explicit_reference(self):
        from layers.quantization.eraserdit_int8 import NativeInt8Linear
        torch.manual_seed(42)
        for bias in (False,True):
            linear=nn.Linear(256,128,bias=bias,device='cuda',dtype=torch.bfloat16)
            quant=NativeInt8Linear.from_linear(linear)
            self.assertEqual(quant.weight_int8.dtype,torch.int8)
            self.assertFalse(hasattr(quant,'weight'))
            for rows in (1,17,128):
                x=torch.randn(2,rows,256,device='cuda',dtype=torch.bfloat16)
                x[:,0]=0
                actual=quant(x)
                flat=x.reshape(-1,256).float()
                scale=flat.abs().amax(-1)/127
                scale=torch.where(scale>0,scale,torch.ones_like(scale))
                xq=(flat/scale[:,None]).round().clamp(-127,127).to(torch.int8)
                accum=(xq.cpu().int() @ quant.weight_int8.cpu().int().t()).cuda()
                expected=accum.float()*scale[:,None]*quant.weight_scale[None,:]
                if bias:expected+=quant.bias.float()
                torch.testing.assert_close(actual,expected.to(torch.bfloat16).reshape_as(actual),rtol=0,atol=.0078125)
                self.assertTrue(torch.isfinite(actual).all())
            self.assertEqual(quant.calls,3)

    def test_model_conversion_is_idempotent_and_executes_selected_layers(self):
        for mode, scope, expected_count in [
                (mode, scope, count)
                for mode in ('int8_w8a8_native', 'fp8_w8a8_native', 'fp8_w8a8_tensorwise', 'fp8_w8a8_static')
                for scope, count in [('blocks', 16), ('ffn', 4), ('ffn_up', 2)]]:
            if mode.startswith('fp8') and torch.cuda.get_device_capability() < (8, 9):
                continue
            from config.server_args import ServerArgs,set_global_server_args
            from models.dits.eraserdit_transformer import EraserDiTLTXVideoTransformer3DModel
            from models.dits.eraserdit_quantization import quantize_transformer,runtime_report
            set_global_server_args(ServerArgs(device='cuda:0',attention_backend='sdpa'))
            model=EraserDiTLTXVideoTransformer3DModel(in_channels=3,out_channels=1,
                num_attention_heads=2,attention_head_dim=16,cross_attention_dim=32,
                num_layers=2,caption_channels=16).to(device='cuda',dtype=torch.bfloat16).eval()
            model.layerwise_offload_managers = [object()]
            with self.assertRaisesRegex(ValueError, 'before registering offload'):
                quantize_transformer(model)
            model.layerwise_offload_managers = []
            from config.eraserdit import EraserDiTPipelineConfig
            from pipelines.eraserdit_erase_pipeline import EraserDiTErasePipeline
            from pipelines.base import ComposedPipelineBase
            args=ServerArgs(device='cuda:0', transformer_quantization=mode,
                            pipeline_config=EraserDiTPipelineConfig(quantization_scope=scope))
            pipeline=EraserDiTErasePipeline.__new__(EraserDiTErasePipeline)
            with patch.object(ComposedPipelineBase,'load_modules',return_value={'transformer':model}):
                loaded=pipeline.load_modules(args)
            self.assertIs(loaded['transformer'],model)
            report=model._eraserdit_quantization_report
            self.assertIs(quantize_transformer(model, scope=scope, mode=mode),report)
            self.assertEqual(report['quantized_count'],expected_count)
            self.assertGreater(report['source_linear_bytes'],report['quantized_linear_bytes'])
            with self.assertRaises(ValueError):quantize_transformer(model,'ffn_up' if scope == 'ffn' else 'ffn',mode=mode)
            x=torch.randn(1,1,1,2,2,device='cuda',dtype=torch.bfloat16)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(hidden_states=x,cond_latents=x,mask_values=torch.ones_like(x),
                    encoder_hidden_states=torch.randn(1,4,16,device='cuda',dtype=torch.bfloat16),
                    encoder_attention_mask=torch.ones(1,4,device='cuda'),timestep=torch.ones(1,device='cuda'),
                    num_frames=1,height=2,width=2,return_dict=False)[0]
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(runtime_report(model)['executed_module_count'],expected_count)
            fused_count = 2 if scope != 'ffn_up' else 0
            self.assertEqual(report['fused_gelu_count'], fused_count)
            self.assertEqual(runtime_report(model)['fused_gelu_call_count'], fused_count)
            if fused_count:
                luts = [block.ff.net[2].gelu_lut for block in model.transformer_blocks]
                self.assertEqual(len({lut.data_ptr() for lut in luts}), fused_count)
