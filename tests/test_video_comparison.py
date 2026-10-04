import unittest
from pathlib import Path
import shutil
import tempfile
import cv2
import numpy as np

from entrypoints.cli.compare_videos import aggregate, frame_metrics, compare


class VideoComparisonTests(unittest.TestCase):
    def test_identity_and_empty_mask(self):
        rgb = np.random.default_rng(4).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        row, error = frame_metrics(rgb, rgb, np.zeros((32, 32), dtype=bool))
        self.assertEqual(row['ssim'], 1.)
        self.assertEqual(row['mae'], 0.)
        self.assertIsNone(row['mask_ssim'])
        self.assertIsNone(row['temporal_delta_mae'])
        summary = aggregate([row])
        self.assertEqual(summary['mask_frames'], 0)
        self.assertIsNone(summary['min_mask_frame_ssim'])
        self.assertFalse(error.any())

    def test_local_damage_and_temporal_error(self):
        reference = np.full((64, 64, 3), 100, dtype=np.uint8)
        candidate = reference.copy()
        candidate[30:34, 30:34] = 0
        mask = np.zeros((64, 64), dtype=bool)
        mask[30:34, 30:34] = True
        row, error = frame_metrics(reference, candidate, mask, np.zeros_like(reference, dtype=np.float32))
        self.assertGreater(row['ssim'], row['mask_ssim'])
        self.assertEqual(row['mask_mae'], 100.)
        self.assertEqual(row['outside_mae'], 0.)
        self.assertEqual(row['temporal_delta_mae'], row['mae'])
        stable, _ = frame_metrics(reference, candidate, mask, error)
        self.assertEqual(stable['temporal_delta_mae'], 0.)

    def test_invalid_shapes_and_empty_video(self):
        with self.assertRaises(ValueError):
            aggregate([])
        with self.assertRaises(ValueError):
            frame_metrics(np.zeros((16, 16, 3)), np.zeros((15, 16, 3)), np.zeros((16, 16), dtype=bool))

    @unittest.skipUnless(shutil.which('ffprobe'), 'ffprobe required')
    def test_full_decode_identity_and_reject_short_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, count in (('reference', 3), ('candidate', 3), ('mask', 3), ('short', 2)):
                writer = cv2.VideoWriter(str(root/f'{name}.avi'), cv2.VideoWriter_fourcc(*'FFV1'), 10., (32, 24))
                self.assertTrue(writer.isOpened())
                for i in range(count):
                    writer.write(np.full((24, 32, 3), 0 if name == 'mask' else 30+i*20, dtype=np.uint8))
                writer.release()
            result = compare(root/'reference.avi', root/'candidate.avi', root/'mask.avi')
            self.assertTrue(result['exact_rgb_match'])
            self.assertEqual(result['frames'], 3)
            self.assertEqual(result['mean']['ssim'], 1.)
            self.assertIsNone(result['mean']['mask_ssim'])
            with self.assertRaisesRegex(ValueError, 'frame counts'):
                compare(root/'reference.avi', root/'short.avi', root/'mask.avi')
