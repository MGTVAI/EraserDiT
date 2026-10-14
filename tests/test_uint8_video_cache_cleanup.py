import os
import unittest
from unittest.mock import patch
import numpy as np
import torch
from utils.video_io import frames_uint8_to_cpu_tensor, video_input_to_float32, maybe_clear_video_upload_cache

class Uint8CacheCleanupTests(unittest.TestCase):
    def test_cpu_or_disabled_does_not_clear_cache(self):
        raw = torch.ones(1, 3, 8, 8, dtype=torch.uint8)
        video = video_input_to_float32(raw, 'cpu')
        for flag in ('0', '1'):
            with patch.dict(os.environ, {'MGERASE_VIDEO_UINT8_GPU_CLEAR_CACHE': flag}), patch('torch.cuda.empty_cache') as clear:
                result = maybe_clear_video_upload_cache(raw, video)
            self.assertFalse(result['applied'])
            clear.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_cleanup_once_preserves_values(self):
        raw = np.tile(np.arange(256, dtype=np.uint8).reshape(1, 16, 16, 1), (1, 1, 1, 3))
        cpu = frames_uint8_to_cpu_tensor(raw)
        video = video_input_to_float32(cpu, 'cuda')
        expected = video.clone()
        original = torch.cuda.empty_cache
        with patch.dict(os.environ, {'MGERASE_VIDEO_UINT8_GPU_CLEAR_CACHE': '1'}), patch('torch.cuda.empty_cache', wraps=original) as clear:
            result = maybe_clear_video_upload_cache(cpu, video)
        clear.assert_called_once()
        self.assertTrue(result['applied'])
        self.assertEqual(result['allocated_before_bytes'], result['allocated_after_bytes'])
        self.assertLessEqual(result['reserved_after_bytes'], result['reserved_before_bytes'])
        torch.testing.assert_close(video, expected, rtol=0, atol=0)
        with patch.dict(os.environ, {'MGERASE_VIDEO_UINT8_GPU_CLEAR_CACHE': '0'}), patch('torch.cuda.empty_cache') as clear:
            self.assertFalse(maybe_clear_video_upload_cache(cpu, video)['applied'])
        clear.assert_not_called()