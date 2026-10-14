from dataclasses import replace
from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch
import numpy as np
import torch
from utils.video_io import ChunkedFrameCache
from utils.windowing import WindowSpec
from pipelines.runtime.io.masks import materialize_window_mask

class CpuMaskBufferTests(unittest.TestCase):
    def make_context(self):
        source = (np.arange(10 * 4 * 7).reshape(10,4,7) % 256).astype(np.uint8)
        cache = ChunkedFrameCache(0, (4,7), np.uint8)
        cache.append_owned(source[:6].copy())
        cache.append_owned(source[6:].copy())
        return SimpleNamespace(runtime_mode='windowed_streaming', mask_frame_cache=cache), source
    def spec(self):
        return WindowSpec(window_index=1, scene_index=0, scene_start=0, scene_end=10,
            load_start=2, load_end=10, deal_start=4, deal_length=3, load_length=6,
            overlap_left=2, overlap_right=0, infer_len=9, overlap=1, time_sample=8, time_shift=1)
    def compare(self, context, spec, crop=None):
        with patch.dict(os.environ, {'MGERASE_MASK_WINDOW_BUFFER_OPT':'0'}):
            expected = materialize_window_mask(context=context, spec=spec, crop_bbox=crop)
        with patch.dict(os.environ, {'MGERASE_MASK_WINDOW_BUFFER_OPT':'1'}):
            actual = materialize_window_mask(context=context, spec=spec, crop_bbox=crop)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.device.type, 'cpu')
        self.assertTrue(actual.is_contiguous())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        np.testing.assert_array_equal(actual.numpy().view(np.uint32), expected.numpy().view(np.uint32))
        return actual
    def test_cross_chunk_overlap_and_future_padding_exact(self):
        context, source = self.make_context()
        actual = self.compare(context, self.spec())
        self.assertEqual(torch.count_nonzero(actual[:,:,:2]).item(), 0)
        self.assertEqual(torch.count_nonzero(actual[:,:,5:]).item(), 0)
        actual.zero_()
        np.testing.assert_array_equal(context.mask_frame_cache.slice(0,10), source)
    def test_first_window_full_active_and_all_uint8_values_exact(self):
        context, _ = self.make_context()
        spec = replace(self.spec(), window_index=0, load_start=0, load_end=10, deal_start=0, deal_length=10, overlap_left=0, load_length=10)
        self.compare(context, spec)
    def test_crop_and_python_slice_bounds_exact(self):
        context, _ = self.make_context()
        for crop in ((1,1,4,2), (4,2,8,8), (-3,0,2,4)):
            self.compare(context, self.spec(), crop)
    def test_safe_cast_only_and_default_dtype_contract_preserved(self):
        context, source = self.make_context()
        result = np.empty((6,2,3), dtype=np.float32)
        context.mask_frame_cache.slice_into(3,9,result,crop_bbox=(1,1,3,2),allow_cast=True)
        np.testing.assert_array_equal(result, source[3:9,1:3,1:4].astype(np.float32))
        with self.assertRaises(ValueError):
            context.mask_frame_cache.slice_into(3,9,result,crop_bbox=(1,1,3,2))
        with self.assertRaises(ValueError):
            context.mask_frame_cache.slice_into(3,9,np.empty((6,4,7),np.int8),allow_cast=True)
    def test_full_runtime_unchanged(self):
        context = SimpleNamespace(runtime_mode='full', mask_cache=torch.arange(280,dtype=torch.float32).reshape(1,1,10,4,7)/255)
        self.compare(context, self.spec())
