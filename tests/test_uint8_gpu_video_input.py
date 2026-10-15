"""Exact values and window contracts for optional uint8 device normalization."""
import gc
import os
import unittest
from unittest.mock import patch
import numpy as np
import torch
from utils.video_io import (
    ArrayFrameCache, frames_uint8_to_cpu_tensor, frames_uint8_to_tensor,
    video_input_to_float32,
)


class Uint8VideoInputTests(unittest.TestCase):
    def test_uint8_values_layout_and_lifetime(self):
        raw = np.tile(np.arange(256, dtype=np.uint8).reshape(1, 16, 16, 1), (2, 1, 1, 3))
        for view in (raw, raw[:, ::2], raw[:, ::-1]):
            for readonly in (False, True):
                value = view.view()
                value.flags.writeable = not readonly
                before = value.copy()
                result = frames_uint8_to_cpu_tensor(value)
                self.assertEqual(result.dtype, torch.uint8)
                torch.testing.assert_close(result, torch.from_numpy(before).permute(0, 3, 1, 2), rtol=0, atol=0)
                normalized = video_input_to_float32(result, 'cpu')
                torch.testing.assert_close(normalized, frames_uint8_to_tensor(before), rtol=0, atol=0)
                np.testing.assert_array_equal(value, before)
        owned = raw.copy()
        tensor = frames_uint8_to_cpu_tensor(owned)
        self.assertEqual(tensor.data_ptr(), owned.ctypes.data)
        expected = tensor.clone()
        del owned
        gc.collect()
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_all_256_values_exact_on_cuda(self):
        raw = np.tile(np.arange(256, dtype=np.uint8).reshape(1, 16, 16, 1), (1, 1, 1, 3))
        cpu_expected = frames_uint8_to_tensor(raw)
        actual = video_input_to_float32(frames_uint8_to_cpu_tensor(raw), 'cuda')
        torch.testing.assert_close(actual.cpu(), cpu_expected, rtol=0, atol=0)
        legacy = video_input_to_float32(cpu_expected, 'cuda')
        torch.testing.assert_close(actual, legacy, rtol=0, atol=0)

    def test_window_materialization_switch_and_tensor_cache(self):
        from tests.test_runtime_boundaries import WindowRuntimeTests
        from pipelines.runtime.objects import build_object_runtime_states
        from pipelines.runtime.windowing.materializer import materialize_window_batch
        for tensor_cache in (False, True):
            inputs, _ = WindowRuntimeTests().make_runtime(streaming=tensor_cache)
            context, params, batch = inputs['context'], inputs['params'], inputs['batch']
            context.window_runtime_mode = 'streaming'
            states, _ = build_object_runtime_states(context=context, params=params)
            state = states[0]
            spec = state.window_specs[0]
            common = dict(batch=batch, context=context, params=params, spec=spec,
                          object_index=0, bbox_frames=None, object_state=state,
                          video_cache=state.input_cache,
                          window_mask=torch.ones(1, 1, spec.input_len, 32, 32),
                          window_bbox=(0, 0, 32, 32),
                          aligned_crop_bbox=(0, 0, 32, 32) if tensor_cache else None)
            with patch.dict(os.environ, {'MGERASE_VIDEO_UINT8_GPU_NORMALIZE': '0'}):
                baseline = materialize_window_batch(**common)
            with patch.dict(os.environ, {'MGERASE_VIDEO_UINT8_GPU_NORMALIZE': '1'}):
                candidate = materialize_window_batch(**common)
            self.assertEqual(candidate.video.dtype, torch.bfloat16 if tensor_cache else torch.uint8)
            actual = video_input_to_float32(candidate.video, 'cpu')
            torch.testing.assert_close(actual, baseline.video.float(), rtol=0, atol=0)
            torch.testing.assert_close(candidate.mask, baseline.mask, rtol=0, atol=0)