"""Behavior across relocated resource, distributed and media boundaries."""

from dataclasses import replace
import importlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from config.server_args import ServerArgs
from config.resource_policy import resolve_runtime_resource_policy
from media.encoding import VideoEncodingProfile
from media.video_io import SequentialVideoReader, SequentialVideoWriter, read_video_metadata
from memory.tensor_ops import maybe_pin_tensor, module_device, move_module_to_device
from parallel import runtime
from utils import logging_utils


ROOT = Path(__file__).resolve().parents[1]


class RuntimeBoundaryTests(unittest.TestCase):
    def test_logging_and_media_import_without_inference_runtime(self):
        result = subprocess.run([sys.executable, "-c", """
import sys
from utils.logging_utils import get_is_main_process
assert get_is_main_process()
import media.video_io
for name in ('config', 'distributed', 'parallel', 'models', 'memory', 'nodes', 'pipelines'):
    assert name not in sys.modules, name
"""], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_compatibility_aliases_share_runtime_state(self):
        for old, new in (
            ("utils.distributed_runtime", "parallel.runtime"),
            ("utils.video_io", "media.video_io"),
            ("utils.ltx095_text", "models.text_encoders.ltx095_text"),
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
        args = ServerArgs(resource_policy="dynamic_offload", pin_memory=True)
        with patch("torch.cuda.is_available", return_value=False):
            policy = args.resolve_resource_policy()
        self.assertTrue(policy.requested_dynamic_offload)
        self.assertTrue(policy.requested_pin_memory)
        self.assertFalse(policy.dynamic_offload)
        self.assertFalse(policy.pin_memory)
        self.assertEqual(policy.fallback_reasons, (
            "pin_memory_disabled_without_cuda", "dynamic_offload_disabled_without_cuda",
        ))
        module = torch.nn.Linear(2, 2)
        weight = module.weight
        self.assertFalse(move_module_to_device(module, torch.device("cpu")))
        self.assertIs(module.weight, weight)
        self.assertIs(maybe_pin_tensor(weight, enable=False), weight)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg unavailable")
    def test_async_video_writer_flushes_in_order_and_keeps_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "output.mp4")
            profile = VideoEncodingProfile.from_metadata({
                "width": 16, "height": 16, "codec_name": "h264",
                "fps_fraction": "24000/1001", "pix_fmt": "yuv420p",
            })
            legacy = importlib.import_module("utils.ltx095_origin_contract")
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
