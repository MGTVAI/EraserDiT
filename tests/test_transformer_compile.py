"""Whole-DiT capture, shape reuse and unsupported execution boundaries."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from config.server_args import ServerArgs, get_global_server_args, set_global_server_args
from config.torch_compile import validate_transformer_compile
from layers.transformer_compile import configure_transformer_compile


class TransformerCompileTests(unittest.TestCase):
    def test_contracts(self):
        base = dict(enable_torch_compile=True, torch_compile_scope='transformer')
        ServerArgs(**base, text_encoder_cpu_offload=True, vae_cpu_offload=True)
        for changed in (dict(dit_layerwise_offload=True), dict(dit_cpu_offload=True),
                        dict(transformer_quantization='int8_w8a8_native'),
                        dict(operator_fusion_backend='auto'), dict(attention_backend='sage_attn'),
                        dict(pipeline_config=SimpleNamespace(sp_degree=2)),
                        dict(pipeline_config=SimpleNamespace(cfg_degree=4)),
                        dict(pipeline_config=SimpleNamespace(cfg_parallel_device='cuda:1'))):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                ServerArgs(**base, **changed)
        ServerArgs(**base, pipeline_config=SimpleNamespace(cfg_degree=2, sp_degree=1))
        args = ServerArgs(**base)
        for batch in ({'transformer_cache_mode': 'teacache'}, {'cache_text_projections': True}):
            with self.assertRaisesRegex(ValueError, 'caches off'):
                validate_transformer_compile(args, batch)
        validate_transformer_compile(args, {'transformer_cache_mode': 'off'})
        # Existing FFN/cache/offload configuration remains accepted.
        validate_transformer_compile(ServerArgs(enable_torch_compile=True, dit_layerwise_offload=True),
                                     {'transformer_cache_mode': 'teacache'})

    def test_stage_selects_whole_transformer_and_reports_execution(self):
        from config.eraserdit import EraserDiTPipelineConfig
        from pipelines.stages.eraserdit_erase.denoising import EraserDiTEraseDenoisingStage
        args = ServerArgs(enable_torch_compile=True, torch_compile_scope='transformer',
                          pipeline_config=EraserDiTPipelineConfig())
        model = self.model('cpu', torch.float32)
        with patch('layers.transformer_compile.torch.compile', side_effect=lambda fn, **kw: fn):
            stage = EraserDiTEraseDenoisingStage(model, object(), args)
        self.assertIsNot(stage.transformer_for_forward, model)
        self.assertEqual(stage.compile_status_snapshot()['scope'], 'transformer')
        self.assertEqual(stage.compile_status_snapshot()['successful_forwards'], 0)
        self.assertFalse(hasattr(model, '_block_compile_report'))
        with torch.no_grad():
            stage.transformer_for_forward(**self.inputs(model, 1))
        self.assertEqual(stage.compile_status_snapshot()['successful_forwards'], 1)
        stage.fallback_to_eager('explicit test reset')
        self.assertIs(stage.transformer_for_forward, model)
        self.assertFalse(stage.compile_status_snapshot()['applied'])

    def model(self, device, dtype):
        from models.dits.eraserdit_transformer import EraserDiTLTXVideoTransformer3DModel
        previous = get_global_server_args()
        self.addCleanup(set_global_server_args, previous)
        set_global_server_args(ServerArgs(device=device, attention_backend='sdpa'))
        torch.manual_seed(19)
        return EraserDiTLTXVideoTransformer3DModel(
            in_channels=3, out_channels=1, num_attention_heads=2,
            attention_head_dim=16, cross_attention_dim=32, num_layers=2,
            caption_channels=16).to(device=device, dtype=dtype).eval()

    def inputs(self, model, frames):
        parameter = next(model.parameters())
        hidden = torch.randn(1, 1, frames, 2, 3, device=parameter.device, dtype=parameter.dtype)
        values = dict(hidden_states=hidden, cond_latents=torch.randn_like(hidden), mask_values=torch.ones_like(hidden),
                      encoder_hidden_states=torch.randn(1, 4, 16, device=hidden.device, dtype=hidden.dtype),
                      encoder_attention_mask=torch.ones(1, 4, device=hidden.device),
                      timestep=torch.ones(1, device=hidden.device), num_frames=frames,
                      height=2, width=3, return_dict=False)
        values['image_rotary_emb'] = model.rope(hidden, frames, 2, 3, None)
        return values

    def test_full_graph_and_timestep_shape_reuse(self):
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        graphs = []
        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward
        actual_compile = torch.compile
        def compile_for_capture(fn, **options):
            return actual_compile(fn, backend=backend, fullgraph=options['fullgraph'], dynamic=False)
        model = self.model('cpu', torch.float32)
        with patch('layers.transformer_compile.torch.compile', side_effect=compile_for_capture):
            compiled = configure_transformer_compile(model, mode='default')
        with torch.no_grad():
            for frames in (1, 2, 1):
                values = self.inputs(model, frames)
                for step in (1., 2., 3.):
                    values['timestep'].fill_(step)
                    torch.testing.assert_close(compiled(**values)[0], model(**values)[0])
        # One complete graph per shape, no new graph for timestep values or reuse.
        self.assertEqual(len(graphs), 2)
        self.assertEqual(compiled.report['successful_forwards'], 9)
        targets = str(graphs[0].graph)
        self.assertIn('scaled_dot_product_attention', targets)
        self.assertIn('proj_in', targets)
        self.assertIn('proj_out', targets)
        with torch.no_grad(), self.assertRaisesRegex(ValueError, 'caches'):
            compiled(**values, text_cache=object())

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get('ERASERDIT_TEST_FULL_COMPILE') == '1',
                         'requires opt-in CUDA Inductor compilation')
    def test_cuda_inductor_multiple_shapes(self):
        model = self.model('cuda', torch.bfloat16)
        compiled = configure_transformer_compile(model, mode='default')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for frames in (1, 2, 1):
                values = self.inputs(model, frames)
                torch.testing.assert_close(compiled(**values)[0], model(**values)[0], atol=0.03, rtol=0.03)
        self.assertEqual(compiled.report['successful_forwards'], 3)

    def exercise_cfg_mesh(self, device, dtype):
        from config.eraserdit import EraserDiTPipelineConfig
        from models.adapters.eraserdit.mesh import EraserDiTMeshWindow
        from pipelines.stages.eraserdit_erase.denoising import EraserDiTEraseDenoisingStage
        args = ServerArgs(device='cuda:0', enable_torch_compile=True, torch_compile_scope='transformer',
                          pipeline_config=EraserDiTPipelineConfig(cfg_degree=2))
        model = self.model(device, dtype)
        devices = [torch.device('cuda', i) for i in range(2)] if device != 'cpu' else [torch.device('cpu')] * 2
        plan = dict(sp=1, cfg=2, devices=devices)
        with patch('pipelines.stages.eraserdit_erase.denoising.resolve_mesh', return_value=plan):
            stage = EraserDiTEraseDenoisingStage(model, object(), args)
        self.addCleanup(stage.close)
        pool = stage._replica_pool
        self.assertIs(pool.compiled_forwards[0], stage.transformer_for_forward)
        self.assertTrue(all(not hasattr(m, '_block_compile_report') for m in pool.models))
        wrappers = tuple(pool.compiled_forwards)
        for frames in (1, 2, 1):
            values = self.inputs(model, frames)
            negative = dict(values, encoder_hidden_states=values['encoder_hidden_states'] + .7)
            with EraserDiTMeshWindow(model, plan, pool=pool) as window:
                for step in (1., 2.):
                    values['timestep'].fill_(step)
                    neg, pos = window.predict(negative, values)
                    torch.testing.assert_close(pos, model(**values)[0].float(), atol=.03, rtol=.03)
                    torch.testing.assert_close(neg, model(**negative)[0].float(), atol=.03, rtol=.03)
            self.assertFalse(pool.active)
            self.assertEqual(tuple(pool.compiled_forwards), wrappers)
        report = stage.compile_status_snapshot()
        self.assertEqual(report['successful_forwards'], 12)
        self.assertEqual([r['successful_forwards'] for r in report['rank_compile']], [6, 6])
        self.assertEqual([len(r['first_call_history']) for r in report['rank_compile']], [2, 2])
        # A failed peer must leave the persistent pool usable for the next window.
        with patch.object(wrappers[1], 'forward', side_effect=RuntimeError('compiled peer failed')):
            with self.assertRaisesRegex(RuntimeError, 'compiled peer failed'):
                with EraserDiTMeshWindow(model, plan, pool=pool) as window:
                    window.predict(negative, values)
        self.assertFalse(pool.active)
        with EraserDiTMeshWindow(model, plan, pool=pool) as window:
            window.predict(negative, values)
        self.assertEqual(report['rank_compile'][0]['successful_forwards'], 6)
        stage.fallback_to_eager('test reset')
        self.assertFalse(pool.compiled_forwards)
        self.assertIs(stage.transformer_for_forward, model)

    def test_cfg_mesh_fullgraph_reuse_and_lifecycle(self):
        from contextlib import nullcontext
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        graphs = []
        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward
        actual_compile = torch.compile
        with patch('layers.transformer_compile.torch.compile',
                   side_effect=lambda fn, **kw: actual_compile(fn, backend=backend, fullgraph=True, dynamic=False)), \
                patch('torch.cuda.device', new=nullcontext), patch('torch.cuda.current_stream'), \
                patch('torch.cuda.synchronize'), patch('torch.autocast', side_effect=lambda *a, **kw: nullcontext()), \
                torch.no_grad():
            self.exercise_cfg_mesh('cpu', torch.float32)
        # Two rank-local models, two shapes; no retracing on timestep changes
        # or on the third window, which reuses the first shape.
        self.assertEqual(len(graphs), 4)

    @unittest.skipUnless(torch.cuda.device_count() >= 2 and os.environ.get('ERASERDIT_TEST_CFG_COMPILE') == '1',
                         'requires two GPUs and opt-in CFG Inductor compilation')
    def test_two_gpu_cfg_inductor(self):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            self.exercise_cfg_mesh('cuda:0', torch.bfloat16)


if __name__ == '__main__':
    unittest.main()
