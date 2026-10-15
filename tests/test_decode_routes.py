import tempfile
import threading
import unittest
from pathlib import Path
import numpy as np
from utils.video_io import SequentialVideoReader, SequentialVideoWriter, read_paired_video_frames

class FakeReader:
    def __init__(self, value, barrier=None, short=False):
        self.value, self.barrier, self.short = value, barrier, short
        self.frames_read = 0
    def read_frames(self, count):
        if self.barrier:
            self.barrier.wait(timeout=3)
        count -= int(self.short)
        self.frames_read += count
        return np.full((count, 4, 4, 3), self.value, dtype=np.uint8)

class DecodeRouteTests(unittest.TestCase):
    def test_parallel_pair_keeps_roles_and_intervals(self):
        barrier = threading.Barrier(2)
        video, mask = FakeReader(17, barrier), FakeReader(231, barrier)
        for start, count in ((0, 3), (3, 2)):
            v, m, metrics = read_paired_video_frames(video, mask, count, parallel=True)
            self.assertTrue(np.all(v == 17))
            self.assertTrue(np.all(m == 231))
            self.assertEqual((metrics['start_frame'], metrics['end_frame']), (start, start + count))
    def test_short_or_misaligned_pair_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'count mismatch'):
            read_paired_video_frames(FakeReader(1, short=True), FakeReader(2), 3, parallel=True)
        video, mask = FakeReader(1), FakeReader(2)
        mask.frames_read = 1
        with self.assertRaisesRegex(RuntimeError, 'positions differ'):
            read_paired_video_frames(video, mask, 3, parallel=True)
    def test_red_is_exact_rgb_first_channel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'colored.mp4')
            colors = np.array([[32,200,128], [90,15,250], [5,50,130], [210,80,17]], dtype=np.uint8)
            source = np.broadcast_to(colors[:,None,None,:], (4,16,16,3)).copy()
            writer = SequentialVideoWriter(path, width=16, height=16, fps=25, thread_count=1)
            try:
                writer.write_frames(source)
            finally:
                writer.close()
            rgb = SequentialVideoReader(path, width=16, height=16, thread_count=1)
            red = SequentialVideoReader(path, width=16, height=16, thread_count=1, extract_red=True, squeeze_single_channel=True)
            try:
                for count in (1,3):
                    actual = red.read_frames(count)
                    expected = rgb.read_frames(count)[...,0]
                    np.testing.assert_array_equal(actual, expected)
                self.assertEqual(rgb.frames_read, red.frames_read)
                self.assertIn('format=rgb24,extractplanes=r', red.command)
            finally:
                rgb.close(); red.close()
    def test_invalid_red_format_fails_early(self):
        with self.assertRaises(ValueError):
            SequentialVideoReader('unused', width=16, height=16, pix_fmt='gray', extract_red=True)