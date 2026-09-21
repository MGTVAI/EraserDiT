import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.optimization_quality import compare, frame_metrics, passes


class QualityGateTests(unittest.TestCase):
    def test_identical_and_known_pixel_errors(self):
        a = np.full((32, 32, 3), 100, np.uint8)
        self.assertEqual(frame_metrics(a, a), dict(ssim=1., mse=0., mae=0.))
        metrics = frame_metrics(a, a + 6)
        self.assertEqual(metrics['mse'], 36.)
        self.assertEqual(metrics['mae'], 6.)
        self.assertTrue(passes(metrics))
        self.assertFalse(passes(frame_metrics(a, a + 7)))

    def test_all_three_inclusive_thresholds(self):
        self.assertTrue(passes(dict(ssim=.985, mse=36, mae=6)))
        for field, value in [('ssim', .9849), ('mse', 36.01), ('mae', 6.01)]:
            metrics = dict(ssim=1, mse=0, mae=0)
            metrics[field] = value
            self.assertFalse(passes(metrics))

    def test_video_decode_and_lossy_is_not_auto_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 24, (32, 32))
            self.assertTrue(writer.isOpened())
            for value in (30, 60):
                writer.write(np.full((32, 32, 3), value, np.uint8))
            writer.release()
            self.assertTrue(compare(path, path)['accepted'])
            result = compare(path, path, lossy=True)
            self.assertIsNone(result['accepted'])
            self.assertEqual(result['verdict'], 'visual_review_required')
            self.assertEqual(result['frame_count'], 2)
