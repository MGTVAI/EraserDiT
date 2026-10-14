import os
import unittest
from unittest.mock import patch
import torch

from models.adapters.eraserdit import postprocess as pp
from utils import eraserdit_mask as morphology
from pipelines.stages.eraserdit_erase.window_validation import _validate_window_tensor
from nodes.stages.validators import StageValidators

STYLE = 'MGERASE_STYLE_UINT8_GPU_UPLOAD'
DILATE = 'MGERASE_MASK_DILATE_CAST_OPT'
VALID = 'MGERASE_WINDOW_UINT8_VALIDATION_OPT'

class PrepostOpt(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_style_transfer_all_uint8_values_and_layouts(self):
        base = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16).expand(7, 3, -1, -1)
        for value in (base, base.contiguous(), base.contiguous(memory_format=torch.channels_last), base.transpose(2, 3)):
            with patch.dict(os.environ, {STYLE: '0'}):
                old = pp._style_to_device(value, torch.device('cuda')) / 255.0
            with patch.dict(os.environ, {STYLE: '1'}):
                new = pp._style_to_device(value, torch.device('cuda')) / 255.0
            self.assertTrue(torch.equal(old, new))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_color_output_modes_and_noncontiguous_destination(self):
        style_base = torch.randint(0, 256, (17, 3, 31, 32), dtype=torch.uint8)
        for dtype in (torch.float32, torch.bfloat16):
            generated = torch.rand(style_base.shape, device='cuda', dtype=dtype)
            for layout in ('contiguous', 'channels_last', 'transposed'):
                style = (style_base if layout == 'contiguous' else
                         style_base.contiguous(memory_format=torch.channels_last) if layout == 'channels_last' else
                         style_base.transpose(2, 3).contiguous().transpose(2, 3))
                for mask_kind in ('binary', 'soft', 'empty', 'singleton'):
                    mask = torch.zeros_like(style)
                    if mask_kind == 'binary':
                        mask[:, :, 2:7, :] = 255
                    elif mask_kind == 'soft':
                        mask = torch.randint(0, 256, style.shape, dtype=torch.uint8)
                        mask[:, :, :3] = 1
                    elif mask_kind in ('empty', 'singleton'):
                        mask.fill_(1)
                        if mask_kind == 'singleton':
                            mask[:, :, 0, 0] = 0
                    results = []
                    for enabled in ('0', '1'):
                        storage = torch.empty((*style.shape, 2), device='cuda')
                        with patch.dict(os.environ, {STYLE: enabled}):
                            out = pp.eraser_dit_window_output(generated, style, mask, out=storage[..., 0], chunked_fp32=True)
                        results.append(out.clone())
                    self.assertTrue(torch.equal(results[0], results[1]), (dtype, layout, mask_kind))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_style_fallback(self):
        for value in (torch.rand(3, 3, 8, 8), torch.randint(0, 256, (3, 3, 8, 8), device='cuda', dtype=torch.uint8)):
            with patch.dict(os.environ, {STYLE: '0'}): old = pp._style_to_device(value, torch.device('cuda'))
            with patch.dict(os.environ, {STYLE: '1'}): new = pp._style_to_device(value, torch.device('cuda'))
            self.assertTrue(torch.equal(old, new))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_dilation_exact_parameters_and_edge_thresholds(self):
        for channels in (1, 3):
            value = torch.randint(0, 256, (3, channels, 33, 35), device='cuda').float()
            value[0, :, 0, 0] = float('nan')
            value[0, :, 0, 1] = float('inf')
            for ksize in ((1, 1), (3, 5), (9, 9)):
                for approximate in (True, False):
                    for count in (0, 1, 3, 9):
                        for threshold in (0., .039, 1., float('nan'), float('inf')):
                            params = dict(ksize=ksize, dilate_iter=count, threshold=threshold, enable_approximate=approximate)
                            with patch.dict(os.environ, {DILATE: '0'}): old = morphology.binarize_and_dilate(value, **params)
                            with patch.dict(os.environ, {DILATE: '1'}): new = morphology.binarize_and_dilate(value, **params)
                            self.assertEqual(old.dtype, new.dtype)
                            self.assertTrue(torch.equal(old, new), params)

    def test_validator_semantics(self):
        values = [None, [], torch.zeros(1, 3, 2, 3, 4, dtype=torch.uint8),
                  torch.zeros(1, 3, 2, 3, dtype=torch.uint8), torch.empty(1, 3, 0, 3, 4, dtype=torch.uint8),
                  torch.zeros(1, 1, 2, 3, 4), torch.full((1, 1, 2, 3, 4), float('nan')),
                  torch.full((1, 1, 2, 3, 4), float('inf')),
                  torch.full((1, 1, 2, 3, 4), complex(0, float('nan')), dtype=torch.complex64)]
        for value in values:
            expected = StageValidators.tensor_with_dims(value, 5)
            for flag in ('0', '1'):
                with patch.dict(os.environ, {VALID: flag}):
                    self.assertEqual(expected, _validate_window_tensor(value))

    def test_validator_skips_only_uint8_nan_scan(self):
        with patch.dict(os.environ, {VALID: '1'}):
            with patch('torch.isnan', side_effect=AssertionError('uint8 scan')):
                self.assertTrue(_validate_window_tensor(torch.zeros(1, 1, 2, 2, 2, dtype=torch.uint8)))
            with patch('torch.isnan', wraps=torch.isnan) as scan:
                self.assertTrue(_validate_window_tensor(torch.zeros(1, 1, 2, 2, 2)))
                self.assertEqual(scan.call_count, 1)

if __name__ == '__main__':
    unittest.main()
