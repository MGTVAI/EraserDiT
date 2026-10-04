"""GPU integration test; run from the repository root with local data/model weights."""
import unittest
from pathlib import Path

import torch

from utils.inference_utils import init


@unittest.skipUnless(torch.cuda.is_available() and Path("data/model").is_dir(),
                     "Requires CUDA and local model weights")
class StageCPUOffloadTest(unittest.TestCase):
    def test_residency_parity_repeated_calls_and_failure_cleanup(self):
        pipe = init(torch.device("cuda"), torch.bfloat16, "data/model")
        models = {name: getattr(pipe, name) for name in ("text_encoder", "vae", "transformer")}

        def generate(**extra):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return pipe(video=torch.zeros(9, 3, 64, 64), masks=torch.ones(2, 1, 64, 64),
                            prompt="A lake.", negative_prompt="", num_frames=9, height=64, width=64,
                            num_inference_steps=2, strength=1.0, output_type="pt",
                            generator=torch.Generator("cuda").manual_seed(42), **extra).frames

        expected = generate().cpu()
        pipe.enable_stage_cpu_offload()
        seen = []
        original_activate = pipe._activate_stage

        def activate(name=None):
            original_activate(name)
            seen.append(name)
            for key, model in models.items():
                self.assertEqual(next(model.parameters()).device.type,
                                 "cuda" if key == name else "cpu")

        pipe._activate_stage = activate
        for _ in range(2):
            actual = generate().cpu()
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(seen[-5:], ["text_encoder", "vae", "transformer", "vae", None])

        def fail(*args, **kwargs):
            raise RuntimeError("injected callback failure")

        with self.assertRaisesRegex(RuntimeError, "injected callback failure"):
            generate(callback_on_step_end=fail)
        self.assertEqual(seen[-1], None)
        for model in models.values():
            self.assertEqual(next(model.parameters()).device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
