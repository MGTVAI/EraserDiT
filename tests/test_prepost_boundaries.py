"""Compatibility and data contracts for model pre/postprocessing adapters."""
import importlib
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch

from models.adapters.ltx095.preprocess import preprocess_single_window
from models.adapters.ltx095.postprocess import postprocess_single_window
from utils.bbox import resolve_single_window_crop_bbox

ROOT = Path(__file__).resolve().parents[1]


class PrepostBoundaryTests(unittest.TestCase):
    def test_aliases_preserve_identity_in_both_import_orders(self):
        script = """
import importlib, sys
for old, new in (
 ('utils.erase_preprocess', 'models.adapters.ltx095.preprocess'),
 ('utils.erase_postprocess', 'models.adapters.ltx095.postprocess'),
 ('utils.eraserdit_preprocess', 'models.adapters.eraserdit.preprocess'),
 ('utils.eraserdit_postprocess', 'models.adapters.eraserdit.postprocess'),
):
 for name in ((old, new) if sys.argv[1] == 'legacy' else (new, old)):
  importlib.import_module(name)
 assert sys.modules[old] is sys.modules[new]
for name in ('pipelines', 'entrypoints', 'loader'):
 assert name not in sys.modules, name
"""
        for order in ('legacy', 'canonical'):
            with self.subTest(order=order):
                result = subprocess.run([sys.executable, '-c', script, order], cwd=ROOT,
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_legacy_patch_reaches_processing_globals(self):
        for old, new, fn, helper in (
            ('utils.erase_preprocess', 'models.adapters.ltx095.preprocess',
             'preprocess_single_window', 'resolve_single_window_crop_bbox'),
            ('utils.erase_postprocess', 'models.adapters.ltx095.postprocess',
             'postprocess_single_window', 'adaptive_instance_normalization_mask'),
            ('utils.eraserdit_preprocess', 'models.adapters.eraserdit.preprocess',
             'preprocess_eraserdit_window', 'binarize_and_dilate'),
            ('utils.eraserdit_postprocess', 'models.adapters.eraserdit.postprocess',
             'eraser_dit_window_output', 'adaptive_instance_normalization_mask'),
        ):
            legacy, canonical = importlib.import_module(old), importlib.import_module(new)
            marker = object()
            with patch.object(legacy, helper, marker):
                self.assertIs(getattr(canonical, fn).__globals__[helper], marker)

    def test_shared_crop_import_does_not_load_model_adapters(self):
        result = subprocess.run([sys.executable, '-c', """
import sys
from utils.bbox import resolve_single_window_crop_bbox
assert resolve_single_window_crop_bbox(bbox=None, video_width=16, video_height=16,
 align_h=8, align_w=8, scale_area_ratio=1, min_pixels=0, max_pixels=256,
 force_crop_align=True) == (0, 0, 16, 16)
assert 'models' not in sys.modules
"""], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_prealigned_crop_and_patch_only_output_contract(self):
        video = torch.full((1, 3, 9, 16, 16), .25)
        mask = torch.zeros((1, 1, 9, 16, 16))
        bbox = (8, 8, 16, 16)
        result = preprocess_single_window(video, mask, None, 9, 8, 8, 0, (3, 3),
                                         1., 0, 256, True, prealigned_crop_bbox=bbox)
        self.assertEqual(result.crop_bbox, bbox)
        torch.testing.assert_close(result.masked_video, video, rtol=0, atol=0)
        common = dict(original_video=video, masked_video=result.masked_video,
                      padded_mask=result.padded_mask, generated_video=torch.ones_like(video),
                      crop_bbox=(0, 0, 16, 16), original_num_frames=9,
                      crop_height=16, crop_width=16, enable_colorfix=False,
                      guss_dialate_iter=0)
        full = postprocess_single_window(**common)
        partial = postprocess_single_window(**common, materialize_full_output=False)
        self.assertIsNone(partial.output_video)
        torch.testing.assert_close(full.output_video, video, rtol=0, atol=0)
        torch.testing.assert_close(partial.crop_video_modified, full.crop_video_modified,
                                   rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'prealigned video shape'):
            preprocess_single_window(video, mask, None, 9, 8, 8, 0, (3, 3),
                                     1., 0, 256, True, prealigned_crop_bbox=(0, 0, 8, 8))


if __name__ == '__main__':
    unittest.main()
