"""Compatibility and data contracts for model pre/postprocessing adapters."""
import importlib
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch

from utils.bbox import resolve_single_window_crop_bbox

ROOT = Path(__file__).resolve().parents[1]


class PrepostBoundaryTests(unittest.TestCase):
    def test_aliases_preserve_identity_in_both_import_orders(self):
        script = """
import importlib, sys
for old, new in (
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



if __name__ == '__main__':
    unittest.main()
