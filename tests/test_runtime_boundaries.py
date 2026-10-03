"""Behavior across relocated resource, distributed and media boundaries."""

from dataclasses import replace
from fractions import Fraction
import importlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from config.server_args import ServerArgs
from config.resource_policy import resolve_runtime_resource_policy
from utils.encoding import VideoEncodingProfile
from utils.video_io import (
    SequentialVideoReader, SequentialVideoWriter, read_video_metadata,
    binarize_mask_array, ArrayFrameCache, TensorFrameCache, frames_uint8_to_tensor,
)
from memory.tensor_ops import maybe_pin_tensor, module_device, move_module_to_device
from parallel import runtime
from utils import logging_utils


ROOT = Path(__file__).resolve().parents[1]


class WindowRuntimeTests(unittest.TestCase):
    def make_runtime(self, *, streaming=False, objects=1, skip_tail=False):
        from config.eraserdit import EraserDiTEraseSamplingParams
        from nodes.schedule_batch import Req
        from pipelines.runtime.contracts import EraseRuntimeContext

        frames = np.full((17, 32, 32, 3), 64, dtype=np.uint8)
        mode = 'windowed_streaming' if streaming else 'windowed_preload'
        params = EraserDiTEraseSamplingParams(
            num_frames=17, height=32, width=32, infer_len=9, overlap=1,
            prompt='background', save_output=False, suppress_logs=True, runtime_mode=mode,
        )
        batch = Req(sampling_params=params, generator=torch.Generator().manual_seed(42))
        cache = (
            TensorFrameCache(start_index=0, shape_tail=(3, 32, 32), dtype=torch.bfloat16)
            if streaming else ArrayFrameCache(start_index=0, shape_tail=(32, 32, 3), dtype=np.uint8)
        )
        cache.append(frames_uint8_to_tensor(frames).bfloat16() if streaming else frames)
        mask = ArrayFrameCache(start_index=0, shape_tail=(32, 32), dtype=np.uint8)
        mask.append(np.full((17, 32, 32), 255, dtype=np.uint8))
        boxes = torch.tensor([[0, 0, 32, 32]] * 17)
        if skip_tail:
            boxes[9:] = 0
        context = EraseRuntimeContext(
            original_video=None, working_video=None, final_video=None, mask_cache=None,
            fps=25, codec_name=None, runtime_mode=mode, requested_runtime_mode=mode,
            effective_runtime_mode=mode, window_runtime_mode='streaming' if streaming else 'preload',
            video_frame_cache=cache, mask_frame_cache=mask, object_count=objects,
            bbox_tracks=[boxes.clone() for _ in range(objects)], request_batch=batch,
        )
        calls = []

        def execute(stages, window, args):
            calls.append((window.extra['object_index'], window.extra['window_index']))
            window.crop_bbox = (0, 0, 32, 32)
            window.crop_video_modified = (window.video + 16 / 255).clamp(0, 1)
            window.output = window.crop_video_modified
            return window

        inputs = dict(executor=SimpleNamespace(execute_with_profiling=execute), stages=[],
                      batch=batch, context=context, params=params, server_args=ServerArgs(),
                      logger=logging.getLogger(__name__))
        return inputs, calls

    def test_window_order_overlap_skip_and_object_forwarding(self):
        from pipelines.runtime.drivers.windowed import run_windowed_runtime

        for streaming in (False, True):
            for objects in (1, 2):
                for skip_tail in (False, True):
                    with self.subTest(streaming=streaming, objects=objects, skip_tail=skip_tail):
                        inputs, calls = self.make_runtime(
                            streaming=streaming, objects=objects, skip_tail=skip_tail)
                        run_windowed_runtime(**inputs)
                        context = inputs['context']
                        self.assertEqual(calls, [(obj, win) for obj in range(objects)
                                                 for win in range(1 if skip_tail else 2)])
                        expected = torch.full((1, 3, 17, 32, 32), 64 / 255)
                        if streaming:
                            expected = expected.bfloat16()
                        for _ in range(objects):
                            # A skipped tail passes through from its load start,
                            # including the preceding window's uncommitted overlap.
                            expected[:, :, :8 if skip_tail else 17] += 16 / 255
                        torch.testing.assert_close(context.final_video, expected, rtol=0, atol=0)
                        self.assertTrue(all(state.finished for state in context.object_states))
                        self.assertEqual([state.skip_count for state in context.object_states],
                                         [int(skip_tail)] * objects)
                        self.assertIsNone(context.pending_window_reclaim)

    def test_failed_commit_releases_payload_and_records_timing(self):
        from pipelines.runtime.drivers.windowed import run_windowed_runtime

        inputs, _ = self.make_runtime()
        windows = []

        def fail_commit(**kwargs):
            windows.append(kwargs['window_batch'])
            raise RuntimeError('commit failed')

        with patch('pipelines.runtime.drivers.windowed.commit_window_to_object_output',
                   side_effect=fail_commit), \
             patch('pipelines.runtime.windowing.commit_sync.resolve_active_window_commit_context',
                   return_value=object()):
            with self.assertRaisesRegex(RuntimeError, 'commit failed'):
                run_windowed_runtime(**inputs)
        self.assertIsNone(windows[0].crop_video_modified)
        self.assertIsNone(windows[0].output)
        self.assertEqual(inputs['context'].runtime_timing_counts['cache_commit'], 1)

    def test_file_context_and_output_for_both_runtime_modes(self):
        from pipelines.eraserdit_erase_pipeline import EraserDiTErasePipeline
        from pipelines.runtime.drivers.windowed import run_windowed_runtime
        from pipelines.runtime.io.output import close_runtime_resources

        with tempfile.TemporaryDirectory() as directory:
            for name, value in (('video', 64), ('mask', 255)):
                writer = SequentialVideoWriter(str(Path(directory) / f'{name}.mp4'),
                                               width=32, height=32, fps=25, thread_count=1)
                try:
                    pixels = np.full((17, 32, 32, 3), value, dtype=np.uint8)
                    if name == 'mask':
                        pixels[:, :16, :, 0] = 30
                        pixels[:, :16, :, 1:] = 200
                    writer.write_frames(pixels)
                finally:
                    writer.close()
            file_bytes = {}
            for streaming, return_tensor, cache_dtype in ((False, True, "bf16"), (False, False, "bf16"),
                    (True, True, "bf16"), (True, False, "bf16"),
                    (True, True, "uint8"), (True, False, "uint8")):
                inputs, _ = self.make_runtime(streaming=streaming)
                batch, params = inputs['batch'], inputs['params']
                params.streaming_cache_dtype = cache_dtype
                params.video_input_path = str(Path(directory) / 'video.mp4')
                params.mask_input_path = str(Path(directory) / 'mask.mp4')
                params.output_path = directory
                params.output_file_name = f'result-{streaming}-{return_tensor}.mp4'
                params.save_output = True
                batch.extra['return_output_tensor'] = return_tensor
                pipeline = EraserDiTErasePipeline.__new__(EraserDiTErasePipeline)
                pipeline._memory_adapter = None
                context = pipeline._prepare_global_context(batch, inputs['server_args'])
                try:
                    inputs['context'] = context
                    if streaming and cache_dtype == 'uint8':
                        from pipelines.runtime.io.streaming import ensure_window_cache_loaded
                        from utils.video_io import read_mask_rgb_array
                        ensure_window_cache_loaded(context, SimpleNamespace(load_start=0, load_end=9))
                        expected_mask, _ = read_mask_rgb_array(params.mask_input_path, threshold_ratio=params.mask_threshold / 2)
                        np.testing.assert_array_equal(context.mask_frame_cache.slice(0, 9), expected_mask[:9])
                    run_windowed_runtime(**inputs)
                    pipeline._maybe_save_output(batch, context)
                    output = read_video_metadata(batch.extra['output_file_path'])
                    self.assertEqual((output['num_frames'], output['width'], output['height']),
                                     (17, 32, 32))
                    self.assertEqual(Fraction(output['fps_fraction']), 25)
                    if streaming or not return_tensor:
                        self.assertIsNone(batch.output)
                    else:
                        self.assertIsNotNone(batch.output)
                    payload = Path(batch.extra['output_file_path']).read_bytes()
                    if return_tensor:
                        file_bytes[streaming, cache_dtype] = payload
                    else:
                        self.assertEqual(payload, file_bytes[streaming, cache_dtype])
                    if streaming and cache_dtype == "uint8":
                        self.assertEqual(payload, file_bytes[False, "bf16"])
                finally:
                    close_runtime_resources(context)

    def test_long_uint8_stream_has_bounded_caches_and_preserves_frame_order(self):
        from pipelines.eraserdit_erase_pipeline import EraserDiTErasePipeline
        from pipelines.runtime.drivers.windowed import run_windowed_runtime
        from pipelines.runtime.io.output import close_runtime_resources
        from pipelines.runtime.io.streaming import ensure_window_cache_loaded
        from utils.video_io import read_mask_rgb_array
        with tempfile.TemporaryDirectory() as directory:
            for name in ('video', 'mask'):
                writer = SequentialVideoWriter(str(Path(directory) / f'{name}.mp4'),
                    width=32, height=32, fps=25, thread_count=1)
                try:
                    for start in range(0, 257, 16):
                        n = min(16, 257-start)
                        if name == 'video':
                            pixels = np.broadcast_to(np.arange(start, start+n, dtype=np.int64)[:, None, None, None] % 256,
                                                     (n, 32, 32, 3)).astype(np.uint8).copy()
                        else:
                            pixels = np.full((n, 32, 32, 3), 1 if start < 128 else 128, dtype=np.uint8)
                        writer.write_frames(pixels)
                finally:
                    writer.close()
            outputs = []
            for streaming in (False, True):
                inputs, _ = self.make_runtime(streaming=streaming)
                batch, params = inputs['batch'], inputs['params']
                params.streaming_cache_dtype = 'uint8'
                params.video_input_path = str(Path(directory) / 'video.mp4')
                params.mask_input_path = str(Path(directory) / 'mask.mp4')
                params.output_path = directory
                params.output_file_name = f'long-{streaming}.mp4'
                params.save_output = True
                batch.extra['return_output_tensor'] = False
                pipeline = EraserDiTErasePipeline.__new__(EraserDiTErasePipeline)
                pipeline._memory_adapter = None
                context = pipeline._prepare_global_context(batch, inputs['server_args'])
                inputs['context'] = context
                observed = []
                expected_mask, _ = read_mask_rgb_array(params.mask_input_path, threshold_ratio=params.mask_threshold / 2)
                def load(ctx, spec):
                    ensure_window_cache_loaded(ctx, spec)
                    observed.append((ctx.video_frame_cache.num_frames, ctx.mask_frame_cache.num_frames))
                    if streaming:
                        np.testing.assert_array_equal(ctx.mask_frame_cache.slice(spec.deal_start, spec.load_end),
                                                      expected_mask[spec.deal_start:spec.load_end])
                try:
                    with patch('pipelines.runtime.windowing.materializer.ensure_window_cache_loaded', side_effect=load):
                        run_windowed_runtime(**inputs)
                    pipeline._maybe_save_output(batch, context)
                    output = Path(batch.extra['output_file_path'])
                    self.assertEqual(read_video_metadata(str(output))['num_frames'], 257)
                    outputs.append(output.read_bytes())
                    if streaming:
                        self.assertGreater(len(observed), 20)
                        self.assertLessEqual(max(max(pair) for pair in observed), 2 * params.infer_len)
                finally:
                    close_runtime_resources(context)
            self.assertEqual(*outputs)


class RuntimeBoundaryTests(unittest.TestCase):
    def test_entrypoint_numpy_hugepages_default_and_override(self):
        for override, expected in ((None, "0"), ("1", "1")):
            env = dict(os.environ)
            env.pop("NUMPY_MADVISE_HUGEPAGE", None)
            if override is not None:
                env["NUMPY_MADVISE_HUGEPAGE"] = override
            result = subprocess.run(
                [sys.executable, "-c",
                 "import entrypoints; import numpy as np; "
                 "print(int(np._core.multiarray._get_madvise_hugepage()))"],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), expected)

    def test_mask_max_scan_cancel_closes_reader(self):
        from unittest.mock import MagicMock
        from utils.video_io import read_mask_rgb_max
        reader = MagicMock()
        def cancel():
            raise RuntimeError('request cancelled')
        with patch('utils.video_io.SequentialVideoReader', return_value=reader):
            with self.assertRaisesRegex(RuntimeError, 'request cancelled'):
                read_mask_rgb_max('unused', width=32, height=32, num_frames=100, checkpoint=cancel)
        reader.close.assert_called_once()
        reader.read_frames.assert_not_called()

    def test_mask_binarization_preserves_threshold_and_dtype(self):
        for dtype in (np.uint8, np.int8, np.uint16, np.float32, np.float64):
            for rgb in (False, True):
                mask = np.array([0, 29, 30, 31, 60, 100], dtype=dtype).reshape(1, 2, 3)
                if rgb:
                    mask = np.stack((mask, np.zeros_like(mask)), axis=-1)
                source = mask.max(axis=-1) if rgb else mask
                expected = np.where(source <= 30.0, 0, 255).astype(dtype)
                actual = binarize_mask_array(mask)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, dtype)
        for mask in (np.zeros((2, 3, 4), dtype=np.uint8),
                     np.empty((0, 3, 4), dtype=np.uint8)):
            np.testing.assert_array_equal(binarize_mask_array(mask), mask)

    def test_logging_and_media_import_without_inference_runtime(self):
        result = subprocess.run([sys.executable, "-c", """
import sys
from utils.logging_utils import get_is_main_process
assert get_is_main_process()
import utils.video_io
for name in ('config', 'distributed', 'parallel', 'models', 'memory', 'nodes', 'pipelines'):
    assert name not in sys.modules, name
"""], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_compatibility_aliases_share_runtime_state(self):
        for old, new in (
            ("utils.distributed_runtime", "parallel.runtime"),
        ):
            self.assertIs(importlib.import_module(old), importlib.import_module(new))
        legacy = importlib.import_module("utils.distributed_runtime")
        context = replace(runtime.get_runtime_distributed_context(), rank=1)
        with patch.object(runtime, "_RUNTIME_DISTRIBUTED_CONTEXT", context):
            self.assertIs(legacy.get_runtime_distributed_context(), context)
            self.assertFalse(logging_utils.get_is_main_process())
            legacy.set_runtime_distributed_context(runtime._disabled_runtime_distributed_context())
            self.assertTrue(logging_utils.get_is_main_process())

    def test_logging_provider_failure_keeps_previous_fallback(self):
        with patch.object(logging_utils, "_main_process_check", side_effect=RuntimeError("probe")):
            self.assertTrue(logging_utils.get_is_main_process())

    def test_split_resource_exports_and_cpu_policy(self):
        legacy = importlib.import_module("utils.resource_policy")
        self.assertIs(legacy.resolve_runtime_resource_policy, resolve_runtime_resource_policy)
        self.assertIs(legacy.module_device, module_device)
        from memory.validation import validate_memory_config
        args = ServerArgs(dit_layerwise_offload=True, device="cuda:0")
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "available CUDA"):
                validate_memory_config(args)
        self.assertTrue(args.resolve_resource_policy().dit_layerwise_offload)

    def test_async_video_writer_flushes_in_order_and_keeps_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "output.mp4")
            profile = VideoEncodingProfile.from_metadata({
                "width": 16, "height": 16, "codec_name": "h264",
                "fps_fraction": "24000/1001", "pix_fmt": "yuv420p",
            })
            legacy = importlib.import_module("utils.window_contract")
            self.assertIs(legacy.VideoEncodingProfile, VideoEncodingProfile)
            writer = SequentialVideoWriter(path, encoding_profile=profile,
                                           thread_count=1, async_queue_depth=1)
            try:
                for value in (32, 96, 160, 224):
                    writer.write_frames_owned(np.full((1, 16, 16, 3), value, dtype=np.uint8))
            finally:
                writer.close()
            writer.close()
            metadata = read_video_metadata(path)
            self.assertEqual(metadata["num_frames"], 4)
            self.assertEqual(metadata["fps_fraction"], "24000/1001")
            reader = SequentialVideoReader(path, width=16, height=16, thread_count=1)
            try:
                frames = np.concatenate([reader.read_frames(1), reader.read_frames(3)])
            finally:
                reader.close()
            np.testing.assert_allclose(frames.mean(axis=(1, 2, 3)), (32, 96, 160, 224), atol=3)


if __name__ == "__main__":
    unittest.main()
