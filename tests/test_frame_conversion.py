"""Exact conversion and storage ownership across NumPy input layouts."""
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from utils.video_io import frames_uint8_to_tensor, mask_uint8_to_tensor


class FrameConversionTests(unittest.TestCase):
    def test_values_layout_and_independent_storage(self):
        for is_mask in (False, True):
            shape = (2, 8, 16) if is_mask else (2, 8, 16, 3)
            raw = np.arange(np.prod(shape), dtype=np.int64).astype(np.uint8).reshape(shape)
            for dtype in (np.uint8, np.float32):
                source = raw.astype(dtype)
                for view in (source, source[:, ::2], source[:, ::-1], source[:, :, ::-1]):
                    for readonly in (False, True):
                        value = view.view()
                        value.flags.writeable = not readonly
                        snapshot = value.copy()
                        expected = torch.from_numpy(value.copy())
                        expected = expected[:, None] if is_mask else expected.permute(0, 3, 1, 2)
                        expected = expected.float() / 255.0
                        convert = mask_uint8_to_tensor if is_mask else frames_uint8_to_tensor
                        actual = convert(value)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0, check_stride=True)
                        actual.add_(1)
                        np.testing.assert_array_equal(value, snapshot)
                        if not readonly:
                            value.fill(0)
                            torch.testing.assert_close(actual, expected + 1, rtol=0, atol=0)
                            value[...] = snapshot

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            frames_uint8_to_tensor(np.zeros((2, 3, 4), dtype=np.uint8))
        with self.assertRaises(ValueError):
            mask_uint8_to_tensor(np.zeros((2, 3, 4, 1), dtype=np.uint8))

    def test_warmup_output_released_before_real_request(self):
        from config.eraserdit import EraserDiTEraseSamplingParams
        from nodes.schedule_batch import Req
        from pipelines.session import EraseSession
        session = EraseSession.__new__(EraseSession)
        session.server_args = SimpleNamespace(enable_torch_compile=False)
        session._cpu_resources = None
        session.build_request = lambda params, **kw: Req(sampling_params=params)
        refs = []

        def forward(req):
            if req.is_warmup:
                req.output = torch.ones(16)
                refs.append(weakref.ref(req.output))
            else:
                self.assertEqual(len(refs), 1)
                self.assertIsNone(refs[0]())
            return req

        session._forward_with_operator_fusion = forward
        with patch('pipelines.session.barrier_if_distributed'), \
             patch('pipelines.session.diagnostic_timing_enabled', return_value=False):
            result = session.run(EraserDiTEraseSamplingParams(), warmup_steps=2)
        self.assertEqual(result.extra['warmup']['steps'], 2)
