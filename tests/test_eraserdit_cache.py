"""CFG isolation, guarded reuse, cleanup, request contracts and model equivalence."""
import unittest
from types import SimpleNamespace

import torch

from cache.eraserdit import EraserDiTCacheWindow
from config.eraserdit_cache import CACHE_DEFAULTS, resolve_eraserdit_cache_params
from config.service_contracts.eraserdit import EraserDiTVideoRequest, _validate_eraserdit_request


def batch(mode, **options):
    return SimpleNamespace(**(CACHE_DEFAULTS | {'transformer_cache_mode': mode} | options), extra={})


class CacheTests(unittest.TestCase):
    def run_step(self, window, branch, step, *, bias=1, layout=(1, 2, 2)):
        adapter = window.kwargs(branch, step)['cache_adapter']
        calls = []
        def blocks(x, start, end):
            calls.extend(range(start, end))
            return x + bias * (end - start)
        output = adapter.run(torch.ones(1, 4, 8), torch.ones(1, 1, 48), blocks,
                             num_blocks=4, layout=layout)
        return output, calls

    def test_cfg_isolation_guards_and_window_reset(self):
        for mode in ('teacache', 'cache_dit'):
            b = batch(mode, teacache_warmup_steps=1, cache_dit_warmup_steps=1)
            for repeat in range(2):
                with EraserDiTCacheWindow(b, total_steps=5, num_blocks=4) as w:
                    for step in range(5):
                        for branch, bias in [('negative', 2), ('positive', 7)]:
                            out, calls = self.run_step(w, branch, step, bias=bias)
                            torch.testing.assert_close(out, torch.full_like(out, 1+4*bias))
                            expected = (0 if mode == 'teacache' else 1) if step in (1, 3) else 4
                            self.assertEqual(len(calls), expected)
                self.assertTrue(b.extra['transformer_cache']['closed'])
                self.assertFalse(b.extra['transformer_cache']['aborted'])
                for state in w.controller._states.values():
                    self.assertFalse(any(isinstance(v, torch.Tensor) for v in vars(state).values()))

    def test_force_compute_and_geometry_invalidation(self):
        for mode in ('teacache', 'cache_dit'):
            for force in (False, True):
                b = batch(mode, transformer_cache_force_compute=force,
                          teacache_warmup_steps=0, cache_dit_warmup_steps=0)
                with EraserDiTCacheWindow(b, total_steps=5, num_blocks=4) as w:
                    self.run_step(w, 'positive', 0)
                    _, calls = self.run_step(w, 'positive', 1, layout=(1, 1, 4))
                    self.assertEqual(len(calls), 4)
                    _, calls = self.run_step(w, 'positive', 2, layout=(1, 1, 4))
                    self.assertEqual(len(calls), 4 if force else (0 if mode == 'teacache' else 1))

    def test_failure_clears_tensors(self):
        for mode in ('teacache', 'cache_dit'):
            b = batch(mode)
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with EraserDiTCacheWindow(b, total_steps=8, num_blocks=4) as w:
                    self.run_step(w, 'negative', 0)
                    raise RuntimeError('injected')
            self.assertTrue(b.extra['transformer_cache']['aborted'])
            for state in w.controller._states.values():
                self.assertFalse(any(isinstance(v, torch.Tensor) for v in vars(state).values()))

    def test_pending_step_cannot_be_overwritten_or_completed_out_of_order(self):
        from cache.base import CacheBranch
        b = batch('teacache', teacache_warmup_steps=0)
        with EraserDiTCacheWindow(b, total_steps=5, num_blocks=4) as w:
            c = w.controller
            values = dict(branch=CacheBranch.POSITIVE, modulated_input=torch.ones(1, 1, 48),
                          global_sequence_length=4, local_sequence_length=4,
                          valid_local_sequence_length=4, layout_signature='grid',
                          dtype='torch.float32', device='cpu', hidden_width=8)
            c.check(step=0, **values)
            with self.assertRaisesRegex(RuntimeError, 'previous step'):
                c.check(step=1, **values)
            with self.assertRaisesRegex(RuntimeError, 'pending'):
                c.record_compute(branch=CacheBranch.POSITIVE, step=1,
                                 modulated_input=values['modulated_input'],
                                 input_hidden_states=torch.ones(1, 4, 8),
                                 output_hidden_states=torch.ones(1, 4, 8))

    def test_back_blocks_execute_on_reuse(self):
        b = batch('cache_dit', cache_dit_back_blocks=1, cache_dit_warmup_steps=0)
        with EraserDiTCacheWindow(b, total_steps=4, num_blocks=4) as w:
            self.run_step(w, 'positive', 0)
            _, calls = self.run_step(w, 'positive', 1)
            self.assertEqual(calls, [0, 3])
        self.assertGreater(b.extra['transformer_cache']['peak_retained_tensor_bytes'], 0)

    def test_off_and_validation(self):
        b = batch('off')
        with EraserDiTCacheWindow(b, total_steps=5, num_blocks=4) as w:
            self.assertEqual(w.kwargs('positive', 0), {})
        for values in ({'transformer_cache_mode': 'unknown'}, {'teacache_threshold': float('nan')},
                       {'transformer_cache_mode': 'cache_dit', 'cache_dit_front_blocks': 28}, {'cache_end_guard_steps': 0},
                       {'max_teacache_consecutive_skip': True}):
            with self.assertRaises((TypeError, ValueError)):
                resolve_eraserdit_cache_params(values)
        resolve_eraserdit_cache_params({'transformer_cache_mode': 'off'}, num_blocks=1)
        resolve_eraserdit_cache_params({'transformer_cache_mode': 'teacache'}, num_blocks=1)
        with self.assertRaises(ValueError):
            resolve_eraserdit_cache_params({'transformer_cache_mode': 'cache_dit'}, num_blocks=1)
        for mode in ('teacache', 'cache_dit'):
            with self.assertRaisesRegex(ValueError, 'torch.compile'):
                _validate_eraserdit_request({'sampling': {'transformer_cache_mode': mode}},
                                           SimpleNamespace(enable_torch_compile=True))
            self.assertEqual(EraserDiTVideoRequest(transformer_cache_mode=mode).transformer_cache_mode, mode)

    def test_http_contract_round_trip_and_strict_validation(self):
        from config.service_contracts.eraserdit import (
            EraserDiTLocalVideoCreateRequest, EraserDiTMultipartVideoParameters,
            build_eraserdit_sampling_params,
        )
        for schema in (EraserDiTLocalVideoCreateRequest, EraserDiTMultipartVideoParameters):
            paths = {'video_path': '/tmp/v.mp4', 'mask_path': '/tmp/m.mp4'} if schema is EraserDiTLocalVideoCreateRequest else {}
            parsed = schema(**paths, transformer_cache_mode='cache_dit', cache_dit_back_blocks=2)
            sampling = parsed.model_dump(exclude={'video_path', 'mask_path', 'bbox_path'})
            params = build_eraserdit_sampling_params(
                {'sampling': sampling, 'video_input_path': '/tmp/v.mp4', 'mask_input_path': '/tmp/m.mp4',
                 'output_dir': '/tmp', 'output_file_name': 'o.mp4', 'runtime_workdir': '/tmp/cache-test'},
                runtime_mode='windowed_preload',
            )
            self.assertEqual(params.transformer_cache_mode, 'cache_dit')
            self.assertEqual(params.cache_dit_back_blocks, 2)
            with self.assertRaises(ValueError):
                schema(**paths, teacache_threshold=float('nan'))
            with self.assertRaises(ValueError):
                schema(**paths, cache_dit_front_blocks=True)
            with self.assertRaises(ValueError):
                schema(**paths, transformer_cache_mode='cache_dit', cache_dit_front_blocks=27, cache_dit_back_blocks=1)

    def test_warmup_does_not_change_original_request(self):
        from config.eraserdit import EraserDiTEraseSamplingParams
        from nodes.schedule_batch import Req
        req = Req(sampling_params=EraserDiTEraseSamplingParams(transformer_cache_mode='cache_dit'))
        warmup = req.copy_as_warmup(3)
        self.assertEqual(warmup.transformer_cache_mode, 'off')
        self.assertEqual(req.transformer_cache_mode, 'cache_dit')
        self.assertEqual(req.num_inference_steps, 50)

    def test_cli_task_overrides(self):
        from entrypoints.cli.erase_eraserdit import _build_parser, _task_to_sampling_params
        args = _build_parser().parse_args(['--video-input', 'v.mp4', '--mask-input', 'm.mp4',
                                         '--output-path', '/tmp/out.mp4', '--transformer-cache-mode', 'teacache'])
        self.assertEqual(_task_to_sampling_params({}, args).transformer_cache_mode, 'teacache')
        self.assertEqual(_task_to_sampling_params({'transformer_cache_mode': 'off'}, args).transformer_cache_mode, 'off')

    def test_real_transformer_force_compute_equivalence(self):
        from config.server_args import ServerArgs, set_global_server_args
        from models.dits.eraserdit_transformer import EraserDiTLTXVideoTransformer3DModel
        set_global_server_args(ServerArgs(device='cpu', attention_backend='sdpa'))
        torch.manual_seed(12)
        model = EraserDiTLTXVideoTransformer3DModel(
            in_channels=3, out_channels=1, num_attention_heads=2,
            attention_head_dim=16, cross_attention_dim=32, num_layers=3, caption_channels=16,
        ).eval()
        values = dict(hidden_states=torch.randn(1, 1, 1, 2, 2), cond_latents=torch.randn(1, 1, 1, 2, 2),
                      mask_values=torch.ones(1, 1, 1, 2, 2), encoder_hidden_states=torch.randn(1, 4, 16),
                      encoder_attention_mask=torch.ones(1, 4), timestep=torch.ones(1),
                      num_frames=1, height=2, width=2, return_dict=False)
        with torch.no_grad():
            reference = model(**values)[0]
            for mode in ('teacache', 'cache_dit'):
                b = batch(mode, transformer_cache_force_compute=True, cache_dit_back_blocks=1)
                with EraserDiTCacheWindow(b, total_steps=3, num_blocks=3) as w:
                    for step in range(3):
                        actual = model(**values, **w.kwargs('positive', step))[0]
                        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
