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
            resource_policy='fullgpu',enable_torch_compile=False,operator_fusion_backend='disabled')
        validate_quantization(args)
        c.sp_degree=2
        validate_quantization(args)
        args.resource_policy = "dynamic_offload"
        args.enable_torch_compile = True
        validate_quantization(args,SimpleNamespace(transformer_cache_mode="teacache",cache_text_projections=False))
        args.operator_fusion_backend = "triton"
        with self.assertRaises(ValueError):
            validate_quantization(args)


@unittest.skipUnless(os.environ.get('ERASERDIT_TEST_INT8')=='1','single GPU opt-in')
class Int8KernelTests(unittest.TestCase):
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
        from config.server_args import ServerArgs,set_global_server_args
        from models.dits.eraserdit_transformer import EraserDiTLTXVideoTransformer3DModel
        from models.dits.eraserdit_quantization import quantize_transformer,runtime_report
        set_global_server_args(ServerArgs(device='cuda:0',attention_backend='sdpa'))
        model=EraserDiTLTXVideoTransformer3DModel(in_channels=3,out_channels=1,
            num_attention_heads=2,attention_head_dim=16,cross_attention_dim=32,
            num_layers=2,caption_channels=16).to(device='cuda',dtype=torch.bfloat16).eval()
        from config.eraserdit import EraserDiTPipelineConfig
        from pipelines.eraserdit_erase_pipeline import EraserDiTErasePipeline
        from pipelines.base import ComposedPipelineBase
        args=ServerArgs(device='cuda:0', transformer_quantization='int8_w8a8_native',
                        pipeline_config=EraserDiTPipelineConfig())
        pipeline=EraserDiTErasePipeline.__new__(EraserDiTErasePipeline)
        with patch.object(ComposedPipelineBase,'load_modules',return_value={'transformer':model}):
            loaded=pipeline.load_modules(args)
        self.assertIs(loaded['transformer'],model)
        report=model._eraserdit_int8_report
        self.assertIs(quantize_transformer(model),report)
        self.assertEqual(report['quantized_count'],16)
        self.assertGreater(report['source_linear_bytes'],report['quantized_linear_bytes'])
        with self.assertRaises(ValueError):quantize_transformer(model,'ffn')
        x=torch.randn(1,1,1,2,2,device='cuda',dtype=torch.bfloat16)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(hidden_states=x,cond_latents=x,mask_values=torch.ones_like(x),
                encoder_hidden_states=torch.randn(1,4,16,device='cuda',dtype=torch.bfloat16),
                encoder_attention_mask=torch.ones(1,4,device='cuda'),timestep=torch.ones(1,device='cuda'),
                num_frames=1,height=2,width=2,return_dict=False)[0]
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(runtime_report(model)['executed_module_count'],16)
