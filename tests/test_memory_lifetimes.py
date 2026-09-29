"""Numerical and ownership checks for bounded window preprocessing."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import patch

import torch

from models.adapters.eraserdit.preprocess import (
    align_nchw, expand_mask_channels, pad_window_frames,
    preprocess_eraserdit_window, window_pad_plan,
    compact_tail_infer_len,
)
from utils.eraserdit_mask import binarize_and_dilate, compress_mask_temporal, gray_normalize_mask


def unchunked_reference(video, mask, *, head_batch, infer_len, approximate):
    video = align_nchw(video, 8, 8)
    mask = expand_mask_channels(align_nchw(mask, 8, 8))
    _, padding = window_pad_plan(len(video), head_batch=head_batch,
                                 infer_len=infer_len, shift_alpha=9)
    video, mask = pad_window_frames(video, mask, num_frames_padded=padding,
                                    head_batch=head_batch, infer_len=infer_len, shift_alpha=9)
    dilated = binarize_and_dilate(mask * 255, enable_approximate=approximate)
    return video * (1 - dilated.to(video.dtype)), gray_normalize_mask(
        compress_mask_temporal(dilated, head_batch=head_batch))


class WindowMemoryTests(unittest.TestCase):
    def test_compact_tail_shapes_preserve_real_frames_and_prefix(self):
        for frames, expected in ((1, 17), (24, 33), (25, 41), (112, 121)):
            length = compact_tail_infer_len(frames, head_batch=False, infer_len=121, overlap=9)
            self.assertEqual(length, expected)
            result = preprocess_eraserdit_window(
                torch.ones(frames, 3, 8, 8), torch.zeros(frames, 1, 8, 8),
                head_batch=False, infer_len=length, shift_alpha=9, align_h=8, align_w=8)
            self.assertEqual(result.masked_video.shape[0] + 9, length)
            self.assertEqual(result.mask_latents.shape[0] + 2, (length-1)//8 + 1)
        self.assertEqual(compact_tail_infer_len(33, head_batch=True, infer_len=121, overlap=9), 121)

    def test_compact_tail_request_contract_and_cli(self):
        from config.eraserdit import EraserDiTEraseSamplingParams
        from config.service_contracts.eraserdit import EraserDiTVideoRequest
        from entrypoints.cli.erase_eraserdit import _build_parser, _task_to_sampling_params
        args = _build_parser().parse_args(['--video-input', '/tmp/video.mp4', '--mask-input',
            '/tmp/mask.mp4', '--output-path', '/tmp/out.mp4', '--compact-tail-padding'])
        self.assertTrue(_task_to_sampling_params({}, args).compact_tail_padding)
        self.assertFalse(_task_to_sampling_params({'compact_tail_padding': False}, args).compact_tail_padding)
        self.assertTrue(EraserDiTVideoRequest(compact_tail_padding=True).compact_tail_padding)
        for cls in (EraserDiTEraseSamplingParams, EraserDiTVideoRequest):
            for kwargs in ({'overlap': 0}, {'infer_len': 120}):
                with self.assertRaises(ValueError):
                    cls(compact_tail_padding=True, **kwargs)

    def check_chunk_equivalence(self, device):
        generator = torch.Generator().manual_seed(41)
        for head, frames, channels, approximate in (
            (True, 20, 1, True), (False, 11, 3, True),
            (True, 3, 3, False), (False, 16, 1, False),
        ):
            video = torch.rand(frames, 3, 25, 33, generator=generator).to(device)
            mask = (torch.rand(frames, channels, 25, 33, generator=generator) > .99).float().to(device)
            reference = unchunked_reference(video, mask, head_batch=head,
                                            infer_len=25, approximate=approximate)
            for chunk_size in (1, 8, 100):
                with self.subTest(device=device, head=head, frames=frames, chunk=chunk_size):
                    actual = preprocess_eraserdit_window(
                        video, mask, head_batch=head, infer_len=25, shift_alpha=9,
                        align_h=8, align_w=8, enable_approximate=approximate,
                        mask_chunk_frames=chunk_size)
                    torch.testing.assert_close(actual.masked_video, reference[0], rtol=0, atol=0)
                    torch.testing.assert_close(actual.mask_latents, reference[1], rtol=0, atol=0)

    def test_chunked_preprocessing_cpu(self):
        self.check_chunk_equivalence('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_chunked_preprocessing_cuda(self):
        from utils.determinism import enable_deterministic_mode
        enable_deterministic_mode()
        self.check_chunk_equivalence('cuda')

    def test_video_released_before_vae_encode(self):
        from config.server_args import ServerArgs, set_global_server_args
        from pipelines.stages.eraserdit_erase._common import (
            EraserDiTTaskState, TASK_STATE_KEY, MODEL_FRAMES_KEY,
        )
        from pipelines.stages.eraserdit_erase.condition_encoding import EraserDiTEraseConditionEncodingStage
        args = ServerArgs(device='cpu')
        set_global_server_args(args)
        vae = torch.nn.Linear(1, 1)
        vae.latents_mean = torch.zeros(1)
        vae.latents_std = torch.ones(1)
        video = torch.rand(1, 3, 9, 8, 8)
        video_ref = weakref.ref(video)
        expected_input = video * 2 - 1
        batch = SimpleNamespace(
            padded_video=video, masked_video=video,
            padded_mask=torch.ones(1, 1, 2, 8, 8), modules={'vae': vae},
            extra={TASK_STATE_KEY: EraserDiTTaskState(), MODEL_FRAMES_KEY: 9},
        )
        del video
        def encode(_vae, value, *_args, **_kwargs):
            self.assertIsNone(video_ref())
            self.assertIsNone(batch.masked_video)
            torch.testing.assert_close(value, expected_input, rtol=0, atol=0)
            return SimpleNamespace(sample=lambda _: torch.ones(1, 1, 2, 2, 2))
        with patch('models.adapters.eraserdit.vae.tiled_vae', encode), patch('torch.autocast', return_value=nullcontext()):
            EraserDiTEraseConditionEncodingStage().forward(batch, args)
        self.assertIsNone(batch.padded_mask)
        self.assertEqual(batch.extra[MODEL_FRAMES_KEY], 9)
        self.assertEqual(batch.mask_values.shape, (1, 1, 2, 2, 2))


if __name__ == '__main__':
    unittest.main()
