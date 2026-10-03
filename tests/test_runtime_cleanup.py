"""Request-local contexts must release frame storage without cyclic GC."""
import gc
from types import SimpleNamespace
import unittest
import weakref

import torch

from config.eraserdit import EraserDiTEraseSamplingParams
from nodes.schedule_batch import Req
from pipelines.runtime.contracts import EraseRuntimeContext
from pipelines.runtime.hooks import RuntimeHookRegistry
from pipelines.runtime.io.output import close_runtime_resources


class RuntimeCleanupTests(unittest.TestCase):
    def check_cleanup(self, *, chain=False, fail=False):
        enabled = gc.isenabled()
        gc.disable()
        try:
            batch = Req(sampling_params=EraserDiTEraseSamplingParams())
            frame = torch.ones(2, 3)
            frame_ref = weakref.ref(frame)
            context = EraseRuntimeContext(
                original_video=None, working_video=None, final_video=frame,
                mask_cache=None, fps=24, codec_name=None, request_batch=batch)
            batch.extra.update(runtime_context=context, runtime_video_metadata={'num_frames': 2})
            # An independently returned output must remain usable after cleanup.
            batch.output = torch.zeros(1)
            if chain:
                registry = RuntimeHookRegistry()
                registry.register_pop_raw_input_hook('next', lambda a, b, ctx=context: ctx)
                context.object_states.append(SimpleNamespace(hook_registry=registry))
                del registry
            if fail:
                def broken_close():
                    raise RuntimeError('reader close failed')
                context.sequential_video_reader = SimpleNamespace(close=broken_close)
            context_ref = weakref.ref(context)
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'reader close failed'):
                    close_runtime_resources(context)
            else:
                close_runtime_resources(context)
                close_runtime_resources(context)  # Idempotent detached ownership.
            self.assertNotIn('runtime_context', batch.extra)
            self.assertIsNone(context.request_batch)
            del frame, context
            self.assertIsNone(context_ref())
            self.assertIsNone(frame_ref())
            self.assertEqual(batch.extra['runtime_video_metadata']['num_frames'], 2)
            self.assertEqual(batch.output.item(), 0)
        finally:
            if enabled:
                gc.enable()
            gc.collect()

    def test_request_context_cycle(self):
        self.check_cleanup()

    def test_object_chain_callbacks(self):
        self.check_cleanup(chain=True)

    def test_cleanup_error_detaches_request(self):
        self.check_cleanup(fail=True)
