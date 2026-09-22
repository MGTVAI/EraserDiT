"""Dual GPU numerical equivalence and worker cleanup; GPU checks are opt-in."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.adapters.eraserdit.cfg import EraserDiTCFGWindow, validate_cfg_parallel


class ConfigTests(unittest.TestCase):
    def test_incompatible_configs_fail_before_execution(self):
        args = SimpleNamespace(device="cuda:0", resource_policy="fullgpu",
                               enable_torch_compile=False,
                               pipeline_config=SimpleNamespace(cfg_parallel_device="cuda:1"))
        with patch("torch.cuda.device_count", return_value=2):
            self.assertEqual(validate_cfg_parallel(args), torch.device("cuda:1"))
            for mode in ("teacache", "cache_dit"):
                with self.assertRaises(ValueError):
                    validate_cfg_parallel(args, SimpleNamespace(transformer_cache_mode=mode))
            args.pipeline_config.cfg_parallel_device = "cuda:0"
            with self.assertRaises(ValueError):
                validate_cfg_parallel(args)
            args.pipeline_config.cfg_parallel_device = "cuda:1"
            args.enable_torch_compile = True
            with self.assertRaises(ValueError):
                validate_cfg_parallel(args)
            args.enable_torch_compile = False
            args.resource_policy = "dynamic_offload"
            with self.assertRaises(ValueError):
                validate_cfg_parallel(args)


@unittest.skipUnless(os.environ.get("ERASERDIT_TEST_TWO_GPU") == "1", "explicit two-GPU test")
class DualGPUModelTests(unittest.TestCase):
    def test_predictions_and_exception_cleanup(self):
        from config.server_args import ServerArgs, set_global_server_args
        from models.dits.eraserdit_transformer import EraserDiTLTXVideoTransformer3DModel
        from utils.determinism import enable_deterministic_mode
        enable_deterministic_mode()
        set_global_server_args(ServerArgs(device="cuda:0", attention_backend="sdpa"))
        torch.manual_seed(42)
        model = EraserDiTLTXVideoTransformer3DModel(
            in_channels=3, out_channels=1, num_attention_heads=2,
            attention_head_dim=16, cross_attention_dim=32, num_layers=2, caption_channels=16,
        ).to(device="cuda:0", dtype=torch.bfloat16).eval()
        values = dict(hidden_states=torch.randn(1, 1, 1, 2, 2, device="cuda:0", dtype=torch.bfloat16),
                      cond_latents=torch.randn(1, 1, 1, 2, 2, device="cuda:0", dtype=torch.bfloat16),
                      mask_values=torch.ones(1, 1, 1, 2, 2, device="cuda:0", dtype=torch.bfloat16),
                      encoder_hidden_states=torch.randn(1, 4, 16, device="cuda:0", dtype=torch.bfloat16),
                      encoder_attention_mask=torch.ones(1, 4, device="cuda:0"),
                      timestep=torch.ones(1, device="cuda:0"), num_frames=1, height=2, width=2,
                      return_dict=False)
        for window_index in range(2):
            values["cond_latents"] = values["cond_latents"] + window_index
            with EraserDiTCFGWindow(model, torch.device("cuda:1")) as window:
                for _ in range(2):
                    future = window.submit(**values)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        expected = model(**values)[0].float()
                    torch.testing.assert_close(future.result().to("cuda:0"), expected, rtol=0, atol=0)
                    values["hidden_states"] = values["hidden_states"] + 0.1
            self.assertIsNone(window.replica)
            self.assertIsNone(window.executor)
            self.assertIsNone(window.static)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with EraserDiTCFGWindow(model, torch.device("cuda:1")) as window:
                with patch.object(window.replica, "forward", side_effect=RuntimeError("injected")):
                    window.submit(**values).result()
        self.assertIsNone(window.replica)
        self.assertIsNone(window.executor)


if __name__ == "__main__":
    unittest.main()
