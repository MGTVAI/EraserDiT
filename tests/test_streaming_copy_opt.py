import io
import tempfile
import types
import unittest
from pathlib import Path
import numpy as np
from utils.video_io import ChunkedFrameCache, SequentialVideoReader, SequentialVideoWriter

class ShortPipe(io.BytesIO):
    def readinto(self, target):
        return super().readinto(target[:7])

class StreamingCopyTests(unittest.TestCase):
    def test_readinto_short_reads_and_independent_buffers(self):
        data = np.arange(72, dtype=np.uint8).reshape(4, 2, 3, 3)
        reader = SequentialVideoReader.__new__(SequentialVideoReader)
        reader.width, reader.height, reader.channels = 3, 2, 3
        reader.read_into, reader.squeeze_single_channel = True, False
        reader.video_path, reader.frames_read, reader.array_copy_bytes = 'fake', 0, 0
        reader.process = types.SimpleNamespace(stdout=ShortPipe(data.tobytes()), stderr=io.BytesIO())
        first, second = reader.read_frames(2), reader.read_frames(2)
        np.testing.assert_array_equal(first, data[:2])
        np.testing.assert_array_equal(second, data[2:])
        self.assertFalse(np.shares_memory(first, second))
        self.assertTrue(first.flags.writeable)
        self.assertEqual(reader.frames_read, 4)
        self.assertEqual(reader.array_copy_bytes, 0)
        with self.assertRaisesRegex(RuntimeError, 'expected 18 bytes, got 0'):
            reader.read_frames(1)
        self.assertEqual(reader.frames_read, 4)

    def test_partial_eof_does_not_advance_position(self):
        reader = SequentialVideoReader.__new__(SequentialVideoReader)
        reader.width, reader.height, reader.channels = 3, 2, 3
        reader.read_into, reader.squeeze_single_channel = True, False
        reader.video_path, reader.frames_read, reader.array_copy_bytes = 'fake', 0, 0
        reader.process = types.SimpleNamespace(stdout=ShortPipe(b'x' * 17), stderr=io.BytesIO(b'decoder failed'))
        with self.assertRaisesRegex(RuntimeError, 'got 17.*decoder failed'):
            reader.read_frames(1)
        self.assertEqual(reader.frames_read, 0)

    def test_discard_matches_pop_and_releases_original_storage(self):
        source = np.arange(24, dtype=np.uint8).reshape(6, 2, 2)
        old, new = [ChunkedFrameCache(10, (2, 2), np.uint8) for _ in range(2)]
        old.append(source)
        self.assertEqual(new.append_owned(source), 0)
        old.pop_before(13)
        self.assertEqual(new.discard_before(13), 12)
        self.assertFalse(np.shares_memory(new._chunks[0], source))
        np.testing.assert_array_equal(new.slice(13, 16), old.slice(13, 16))
        source[:] = 0
        np.testing.assert_array_equal(new.slice(13, 16), old.slice(13, 16))
        self.assertEqual(new.discard_before(13), 0)
        self.assertEqual(new.discard_before(16), 0)
        self.assertEqual((new.start_index, new.end_index, new.num_frames), (16, 16, 0))
        with self.assertRaises(ValueError):
            new.discard_before(17)

    def test_slice_into_across_chunks_and_final_window_assembly(self):
        source = np.arange(40, dtype=np.uint8).reshape(10, 2, 2)
        cache = ChunkedFrameCache(0, (2, 2), np.uint8)
        cache.append_owned(source[:6].copy())
        cache.append_owned(source[6:].copy())
        destination = np.empty((5, 2, 2), dtype=np.uint8)
        cache.slice_into(4, 9, destination)
        np.testing.assert_array_equal(destination, source[4:9])
        overlap = ChunkedFrameCache(4, (2, 2), np.uint8)
        overlap.append_owned(np.full((2, 2, 2), 231, dtype=np.uint8))
        overlap.slice_into(4, 6, destination[:2])
        cache.slice_into(6, 9, destination[2:])
        np.testing.assert_array_equal(destination, np.concatenate([overlap.slice(4, 6), cache.slice(6, 9)]))
        destination[:] = 0
        np.testing.assert_array_equal(cache.slice(4, 9), source[4:9])
        with self.assertRaises(ValueError):
            cache.slice_into(4, 9, np.empty((4, 2, 2), np.uint8))

    def test_actual_rgb_and_red_decode_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'colored.mp4')
            colors = np.array([[32, 200, 128], [90, 15, 250], [5, 50, 130], [210, 80, 17]], np.uint8)
            frames = np.broadcast_to(colors[:, None, None, :], (4, 16, 16, 3)).copy()
            writer = SequentialVideoWriter(path, width=16, height=16, fps=25, thread_count=1)
            try:
                writer.write_frames(frames)
            finally:
                writer.close()
            for red in (False, True):
                old = SequentialVideoReader(path, width=16, height=16, thread_count=1, extract_red=red, squeeze_single_channel=red)
                new = SequentialVideoReader(path, width=16, height=16, thread_count=1, extract_red=red, squeeze_single_channel=red, read_into=True)
                try:
                    for count in (1, 3):
                        actual = new.read_frames(count)
                        np.testing.assert_array_equal(actual, old.read_frames(count))
                        self.assertTrue(actual.flags.writeable)
                    self.assertEqual(new.frames_read, old.frames_read)
                    self.assertEqual(new.array_copy_bytes, 0)
                    self.assertGreater(old.array_copy_bytes, 0)
                finally:
                    old.close()
                    new.close()
