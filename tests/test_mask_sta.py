"""Mask geometry, request lifetime, and opt-in configuration regressions."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from config.q6_extensions import q6_extensions, validate_q6_extensions
from layers.attention.backends import mask_sta
from scripts.q6_presets import preset_environment


class MaskSTATests(unittest.TestCase):
    def setUp(self):
        q6_extensions.cache_clear()

    def tearDown(self):
        q6_extensions.cache_clear()

    def test_geometry_is_bijective_and_preserves_own_blocks(self):
        permutation, background = mask_sta.block_geometry('cpu')
        self.assertTrue(torch.equal(permutation.sort().values, torch.arange(mask_sta.LENGTH)))
        self.assertEqual(tuple(background.shape), (255, 510))
        self.assertEqual(int(background.sum()), 46096)
        query = torch.arange(255)
        self.assertTrue(background[query, query*2].all())
        self.assertTrue(background[query, query*2+1].all())
        lut, counts = mask_sta.block_table(background)
        positions = lut.cumsum(-1)
        for row in range(255):
            self.assertTrue(torch.equal(positions[row, :counts[row]],
                                        background[row].nonzero().flatten()))

    def test_low_nonzero_masks_are_protected_and_windows_are_independent(self):
        permutation, background = mask_sta.block_geometry('cpu')
        mask = torch.zeros((1, 1, *mask_sta.GRID))
        empty, _, _ = mask_sta.protected_blocks(mask, permutation)
        self.assertFalse(empty.any())
        mask[0, 0, 0, 10, 10] = .0039
        protected, binary, expanded = mask_sta.protected_blocks(mask, permutation)
        source_row = 10*mask_sta.GRID[2]+10
        reordered = int((permutation == source_row).nonzero()[0])
        self.assertTrue(protected[reordered//mask_sta.QUERY_BLOCK])
        self.assertEqual(int(binary.sum()), 1)
        self.assertGreater(int(expanded.sum()), 1)
        self.assertFalse(empty.any())
        mask.fill_(1)
        full, _, _ = mask_sta.protected_blocks(mask, permutation)
        table = background.clone()
        table[full] = True
        self.assertTrue(table.all())

    def test_context_restores_nested_windows_and_releases_on_error(self):
        first, second = Mock(), Mock()
        with patch.object(mask_sta, 'MaskSTA', side_effect=[first, second]):
            self.assertIsNone(mask_sta.current_mask_sta())
            with mask_sta.mask_sta_window(None, (), ''):
                self.assertIs(mask_sta.current_mask_sta(), first)
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    with mask_sta.mask_sta_window(None, (), ''):
                        self.assertIs(mask_sta.current_mask_sta(), second)
                        raise RuntimeError('injected')
                self.assertIs(mask_sta.current_mask_sta(), first)
            self.assertIsNone(mask_sta.current_mask_sta())
        first.close.assert_called_once()
        second.close.assert_called_once()

    def test_unsupported_layout_and_device_fail_before_loading_library(self):
        with self.assertRaisesRegex(ValueError, 'token grid'):
            mask_sta.MaskSTA(torch.zeros(1, 1, 8, 34, 60), (8, 34, 60), 'unused')
        with self.assertRaisesRegex(ValueError, 'SM89'):
            mask_sta.MaskSTA(torch.zeros(1, 1, *mask_sta.GRID), mask_sta.GRID, 'unused')

    def test_existing_q6c_preset_clears_sta_and_new_preset_requires_library(self):
        with tempfile.TemporaryDirectory() as folder:
            library = Path(folder)/'optional.so'
            library.touch()
            with patch.dict(os.environ, {'MGERASE_Q6_FFN_LIBRARY': str(library),
                                         'MGERASE_Q6_MASK_STA': '1'}, clear=True):
                env = preset_environment('q6c')
                self.assertEqual(env['MGERASE_Q6_MASK_STA'], '0')
                with self.assertRaisesRegex(ValueError, 'MASK_STA_LIBRARY'):
                    preset_environment('q6c_sta')
            with patch.dict(os.environ, {'MGERASE_Q6_FFN_LIBRARY': str(library),
                                         'MGERASE_Q6_MASK_STA_LIBRARY': str(library)}, clear=True):
                env = preset_environment('q6c_sta')
                self.assertEqual(env['MGERASE_Q6_MASK_STA'], '1')
                self.assertEqual(env['MGERASE_Q6_FFN_EPILOGUE'], '1')
                self.assertTrue(env['MGERASE_Q6_QKV_CALIBRATION'])

    def test_sta_requires_sage_peripheral_and_a_built_library(self):
        args = SimpleNamespace(
            pipeline_config=SimpleNamespace(quantization_scope='ffn', dit_parallel_backend='peer'),
            attention_backend='sage_fp8', transformer_quantization='fp8_w8a8_static')
        with patch.dict(os.environ, {'MGERASE_Q6_MASK_STA': '1'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'SAGE_PERIPHERAL'):
                validate_q6_extensions(args)
        with patch.dict(os.environ, {'MGERASE_Q6_MASK_STA': '1',
                                     'MGERASE_Q6_SAGE_PERIPHERAL': '1'}, clear=True):
            q6_extensions.cache_clear()
            with self.assertRaisesRegex(ValueError, 'MASK_STA_LIBRARY'):
                validate_q6_extensions(args)


@unittest.skipUnless(os.environ.get('ERASERDIT_TEST_MASK_STA') == '1', 'single SM89 GPU opt-in')
class MaskSTAStreamTests(unittest.TestCase):
    @torch.inference_mode()
    def test_nondefault_stream_matches_default_and_closes_window(self):
        from layers.attention.backends.sage_peripheral import SagePeripheral
        torch.manual_seed(67)
        inputs = [torch.randn((1, 32640, 32, 64), device='cuda', dtype=torch.bfloat16) for _ in range(3)]
        mask = torch.zeros((1, 1, *mask_sta.GRID), device='cuda')
        peripheral = SagePeripheral()
        library = os.environ['MGERASE_Q6_MASK_STA_LIBRARY']
        with mask_sta.mask_sta_window(mask, mask_sta.GRID, library) as operator:
            expected = operator(*inputs, .125, peripheral)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                actual = operator(*inputs, .125, peripheral)
            stream.synchronize()
            self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))
        self.assertIsNone(mask_sta.current_mask_sta())
        self.assertTrue(operator.report()['closed'])


if __name__ == '__main__':
    unittest.main()
