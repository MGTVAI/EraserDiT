"""Reference color semantics, CUDA chunking, and final video layout."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from models.adapters.eraserdit.postprocess import (
    _channel_stats, eraser_dit_window_output, quantize_like_baseline,
)
from utils.colorfix_wmask import adaptive_instance_normalization_mask


def reference(generated, style, mask):
    if style.dtype == torch.uint8:
        style = style.float() / 255.
    value = adaptive_instance_normalization_mask(
        quantize_like_baseline(generated.contiguous().cpu()), style.cpu(),
        refer_mask=1 - mask.float().cpu(), valid_mask=None, type='RGB', per_channel=True)
    return quantize_like_baseline(value)


class ColorfixTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_POSTPROCESS') == '1', 'opt-in CUDA colorfix')
    def test_chunked_fp32_matches_whole_window_cast(self):
        from config.server_args import ServerArgs, set_global_server_args
        from pipelines.stages.eraserdit_erase._common import (
            EraserDiTTaskState, TASK_STATE_KEY, PREFIX_LEN_KEY, NEW_FRAMES_KEY,
            ORIG_SIZE_KEY, STYLE_VIDEO_KEY, STYLE_MASK_KEY, MODEL_FRAMES_KEY,
        )
        from pipelines.stages.eraserdit_erase.window_postprocess import EraserDiTEraseWindowPostprocessStage
        args = ServerArgs(device='cuda:0')
        set_global_server_args(args)
        torch.manual_seed(82)
        decoded = torch.rand(1, 3, 20, 32, 40, device='cuda:0', dtype=torch.bfloat16)
        source = torch.randint(0, 256, (17, 3, 29, 31), dtype=torch.uint8)
        for kind in ('binary', 'partial', 'black'):
            mask = torch.zeros_like(source)
            if kind == 'partial':
                mask[:, :, ::2] = 1
            if kind == 'black':
                decoded.zero_()
            results = []
            for enabled in ('0', '1'):
                state = EraserDiTTaskState()
                batch = SimpleNamespace(decoded_video=decoded, colorfix_type='RGB', colorfix_per_channel=True,
                    metrics=None, extra={TASK_STATE_KEY:state, PREFIX_LEN_KEY:2, NEW_FRAMES_KEY:17,
                        ORIG_SIZE_KEY:(29,31), STYLE_VIDEO_KEY:source, STYLE_MASK_KEY:mask,
                        MODEL_FRAMES_KEY:20, 'window_spec':{'overlap_right':3}})
                with patch.dict(os.environ, MGERASE_POSTPROCESS_CHUNKED_FP32=enabled):
                    EraserDiTEraseWindowPostprocessStage().forward(batch, args)
                results.append(batch.crop_video_modified)
                torch.testing.assert_close(state.prev_raw_tail,
                    decoded[0, :, -3:].permute(1, 0, 2, 3).float().cpu(), atol=0, rtol=0)
                self.assertEqual(state.prev_raw_tail.dtype, torch.float32)
                self.assertEqual(state.windows_seen, 1)
            torch.testing.assert_close(*results, atol=0, rtol=0)

    def test_masked_statistics_empty_singleton_and_soft_selection(self):
        torch.manual_seed(41)
        value = torch.rand(2, 3, 7, 9)
        mask = torch.randint(0, 2, value.shape).float()
        mask[0, 0] = 1  # empty
        mask[0, 1] = 1
        mask[0, 1, 0, 0] = .5  # one selected pixel; unbiased variance is NaN
        mask[1, 2] = 255  # raw 255 selects the whole plane
        mean, std = _channel_stats(value, mask != 1)
        for frame in range(2):
            for channel in range(3):
                selected = value[frame, channel][(1 - mask[frame, channel]).bool()]
                if selected.numel() == 0:
                    self.assertEqual(mean[frame, channel].item(), 0)
                    self.assertAlmostEqual(std[frame, channel].item(), 1e-8)
                else:
                    torch.testing.assert_close(mean[frame, channel, 0, 0], selected.mean())
                    if selected.numel() == 1:
                        self.assertTrue(std[frame, channel].isnan().all())
                    else:
                        torch.testing.assert_close(std[frame, channel, 0, 0], (selected.var() + 1e-8).sqrt())

    def test_non_rgb_retains_reference_helper_and_uint8_scale(self):
        source = torch.full((2, 3, 4, 5), 127, dtype=torch.uint8)
        generated = source.float() / 255
        output = torch.empty_like(generated)
        with patch('models.adapters.eraserdit.postprocess.adaptive_instance_normalization_mask',
                   side_effect=lambda **kw: kw['content_feat']) as helper:
            result = eraser_dit_window_output(generated, source, source,
                                             colorfix_type='YUV', out=output)
        self.assertIs(result, output)
        self.assertEqual(helper.call_args.kwargs['type'], 'YUV')
        torch.testing.assert_close(helper.call_args.kwargs['style_feat'], generated, atol=0, rtol=0)
        torch.testing.assert_close(helper.call_args.kwargs['refer_mask'], 1-source.float(), atol=0, rtol=0)

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_POSTPROCESS') == '1', 'opt-in CUDA colorfix')
    def test_cuda_reference_pixels_and_noncontiguous_output(self):
        torch.manual_seed(29)
        device = torch.device('cuda', torch.cuda.device_count()-1)
        # Leave the current device at zero: the fast path must follow its input.
        torch.cuda.set_device(0)
        for kind in ('binary', 'float_style', 'partial', 'soft', 'empty', 'singleton', 'black', 'constant'):
            with self.subTest(kind=kind):
                generated = torch.rand(17, 3, 29, 31, device=device)
                style = torch.randint(0, 256, generated.shape, dtype=torch.uint8)
                mask = torch.randint(0, 2, generated.shape).float() * 255
                if kind == 'float_style':
                    style = style.float() / 255.
                elif kind == 'partial':
                    mask[:, :, ::2] = 1
                elif kind == 'soft':
                    mask = torch.rand_like(mask)
                    mask[:, :, ::2] = 1
                elif kind in ('empty', 'singleton'):
                    mask.fill_(1)
                    if kind == 'singleton':
                        mask[:, :, 0, 0] = .5
                elif kind == 'black':
                    generated.zero_()
                elif kind == 'constant':
                    generated.fill_(.5)
                storage = torch.full((1, 3, 19, 29, 31), -1., device=device)
                out = storage[0, :, 1:-1].permute(1, 0, 2, 3)
                actual = eraser_dit_window_output(generated, style, mask, out=out)
                expected = reference(generated, style, mask)
                self.assertIs(actual, out)
                self.assertEqual(actual.device, device)
                self.assertTrue((storage[:, :, 0] == -1).all())
                self.assertTrue((storage[:, :, -1] == -1).all())
                delta = ((actual.cpu()*255).round() - (expected*255).round()).abs()
                self.assertLessEqual(delta.max().item(), 1)
                # Constant planes amplify tiny CPU/CUDA mean differences at
                # uint8 boundaries; still require every pixel to be within 1.
                if kind != 'constant':
                    self.assertLess((delta != 0).float().mean().item(), .01)
                self.assertTrue(actual.isfinite().all())

    @unittest.skipUnless(os.environ.get('ERASERDIT_TEST_POSTPROCESS') == '1', 'opt-in CUDA colorfix')
    def test_window_stage_preserves_raw_tail_prefix_and_layout(self):
        from config.server_args import ServerArgs, set_global_server_args
        from pipelines.stages.eraserdit_erase._common import (
            EraserDiTTaskState, TASK_STATE_KEY, PREFIX_LEN_KEY, NEW_FRAMES_KEY,
            ORIG_SIZE_KEY, STYLE_VIDEO_KEY, STYLE_MASK_KEY, MODEL_FRAMES_KEY,
        )
        from pipelines.stages.eraserdit_erase.window_postprocess import EraserDiTEraseWindowPostprocessStage
        args = ServerArgs(device='cuda:0')
        set_global_server_args(args)
        torch.manual_seed(82)
        decoded = torch.rand(1, 3, 20, 32, 40, device='cuda:0')
        source = torch.randint(0, 256, (17, 3, 29, 31), dtype=torch.uint8)
        mask = torch.zeros(17, 1, 29, 31, dtype=torch.uint8).expand(-1, 3, -1, -1)
        state = EraserDiTTaskState()
        batch = SimpleNamespace(decoded_video=decoded, colorfix_type='RGB', colorfix_per_channel=True,
                                metrics=None, extra={TASK_STATE_KEY:state, PREFIX_LEN_KEY:2,
                                NEW_FRAMES_KEY:17, ORIG_SIZE_KEY:(29,31), STYLE_VIDEO_KEY:source,
                                STYLE_MASK_KEY:mask, MODEL_FRAMES_KEY:20, 'window_spec':{'overlap_right':3}})
        expected = reference(decoded[0, :, 2:19, :29, :31].permute(1,0,2,3), source, mask)
        EraserDiTEraseWindowPostprocessStage().forward(batch, args)
        self.assertTrue(batch.crop_video_modified.is_contiguous())
        self.assertEqual(tuple(batch.crop_video_modified.shape), (1,3,20,29,31))
        self.assertEqual(batch.crop_video_modified[:, :, :2].count_nonzero(), 0)
        self.assertEqual(batch.crop_video_modified[:, :, 19:].count_nonzero(), 0)
        actual = batch.crop_video_modified[0, :, 2:19].permute(1,0,2,3).cpu()
        self.assertLessEqual(((actual*255).round()-(expected*255).round()).abs().max().item(), 1)
        torch.testing.assert_close(state.prev_raw_tail, decoded[0, :, -3:].permute(1,0,2,3).cpu(), atol=0, rtol=0)
        self.assertIsNone(batch.decoded_video)
        self.assertEqual(state.windows_seen, 1)


if __name__ == '__main__':
    unittest.main()
