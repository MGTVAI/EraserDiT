"""Explicit two/four GPU model, peer failure and VAE tiling verification."""
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from config.eraserdit import EraserDiTPipelineConfig
from config.server_args import ServerArgs, set_global_server_args
from models.adapters.eraserdit.mesh import EraserDiTMeshWindow, PeerExchange, SequenceRank
from models.adapters.eraserdit.vae import tiled_vae
from utils.determinism import enable_deterministic_mode


@unittest.skipUnless(os.environ.get("ERASERDIT_TEST_TWO_GPU") == "1", "explicit GPU opt-in")
class GPUParallelTests(unittest.TestCase):
    def setUp(self):
        from models.dits.eraserdit_transformer import EraserDiTLTXVideoTransformer3DModel
        enable_deterministic_mode()
        set_global_server_args(ServerArgs(device="cuda:0", attention_backend="sdpa"))
        torch.manual_seed(42)
        self.model = EraserDiTLTXVideoTransformer3DModel(
            in_channels=3, out_channels=1, num_attention_heads=4, attention_head_dim=16,
            cross_attention_dim=64, num_layers=2, caption_channels=16,
        ).to(device="cuda:0", dtype=torch.bfloat16).eval()
        self.values = dict(hidden_states=torch.randn(1, 1, 1, 3, 3, device="cuda:0", dtype=torch.bfloat16),
            cond_latents=torch.randn(1, 1, 1, 3, 3, device="cuda:0", dtype=torch.bfloat16),
            mask_values=torch.ones(1, 1, 1, 3, 3, device="cuda:0", dtype=torch.bfloat16),
            encoder_hidden_states=torch.randn(1, 4, 16, device="cuda:0", dtype=torch.bfloat16),
            encoder_attention_mask=torch.ones(1, 4, device="cuda:0"), timestep=torch.ones(1, device="cuda:0"),
            num_frames=1, height=3, width=3, return_dict=False)

    def reference(self, values):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(**values)[0].float()

    def test_cfg_sp_and_available_hybrid(self):
        negative = dict(self.values, encoder_hidden_states=self.values["encoder_hidden_states"] + 0.2)
        expected = self.reference(self.values)
        expected_neg = self.reference(negative)
        cases = [(2, 1), (1, 2)]
        if torch.cuda.device_count() >= 4:
            cases += [(2, 2), (4, 1)]
        for sp, cfg in cases:
            plan = dict(sp=sp, cfg=cfg, devices=[torch.device("cuda", i) for i in range(sp * cfg)])
            with EraserDiTMeshWindow(self.model, plan) as window:
                for _ in range(2):
                    neg, pos = window.predict(negative, self.values)
                    torch.testing.assert_close(pos, expected, atol=0.01, rtol=0.01)
                    torch.testing.assert_close(neg, expected_neg, atol=0.01, rtol=0.01)
            self.assertFalse(window.models)
            self.assertIsNone(window.executor)

    def test_peer_failure_cleanup_and_next_window(self):
        plan = dict(sp=2, cfg=1, devices=[torch.device("cuda", i) for i in range(2)])
        with self.assertRaises(Exception):
            with EraserDiTMeshWindow(self.model, plan) as window:
                with patch.object(window.models[1], "forward", side_effect=RuntimeError("injected peer failure")):
                    window.predict(self.values, self.values)
        self.assertFalse(window.models)
        self.assertIsNone(window.executor)
        with EraserDiTMeshWindow(self.model, plan) as window:
            neg, pos = window.predict(self.values, self.values)
            torch.testing.assert_close(neg, pos, rtol=0, atol=0)

    def test_sage_attention_preserves_global_key_centering(self):
        from layers.attention.backends.sage_attn import SageAttentionImpl
        from layers.attention.backends.attention_backend import AttentionMetadata
        values = [torch.randn(1, 2049, 32, 64, device="cuda:0", dtype=torch.bfloat16) for _ in range(3)]
        impl = SageAttentionImpl(num_heads=32, head_size=64, softmax_scale=64 ** -0.5)
        metadata = AttentionMetadata(attn_mask=None)
        expected = impl.forward(values[0], values[1].clone(), values[2], metadata)
        torch.cuda.synchronize(0)
        exchange = PeerExchange(2)
        def work(index):
            with torch.cuda.device(index):
                rank = SequenceRank(exchange, index)
                shard = rank.partition(values[0].shape[1])
                output = rank.attention(*[x[:, shard].to(f"cuda:{index}") for x in values], impl, metadata)
                torch.cuda.synchronize(index)
                return output
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(work, index) for index in range(2)]
            actual = torch.cat([f.result().to("cuda:0") for f in futures], dim=1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(os.environ.get("ERASERDIT_TEST_MODEL"), "real VAE checkpoint opt-in")
    def test_native_vae_tiling_matches_two_devices(self):
        from models.vaes.eraserdit_vae import EraserDiTAutoencoderKLLTXVideo
        # Actual checkpoint, including its causal encoder and conditioned decoder.
        vae = EraserDiTAutoencoderKLLTXVideo.from_pretrained(
            os.environ["ERASERDIT_TEST_MODEL"] + "/vae", torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to("cuda:0").eval()
        vae.enable_tiling(tile_sample_min_height=128, tile_sample_min_width=128,
                          tile_sample_stride_height=96, tile_sample_stride_width=96)
        value = torch.randn(1, 3, 9, 160, 192, device="cuda:0", dtype=torch.bfloat16)
        args = ServerArgs(device="cuda:0", pipeline_config=EraserDiTPipelineConfig(
            vae_degree=2, vae_tiling=True, vae_tile_size=128, vae_tile_stride=96))
        batch = SimpleNamespace(extra={}, transformer_cache_mode="off", cache_text_projections=False)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = vae.tiled_encode(value)
            actual = tiled_vae(vae, value, args, batch, operation="encode").parameters
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            latents = actual[:, :128]
            temb = torch.zeros(1, device="cuda:0", dtype=torch.bfloat16)
            expected = vae.tiled_decode(latents, temb, return_dict=False)[0]
            actual = tiled_vae(vae, latents, args, batch, operation="decode", temb=temb)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(batch.extra["vae_parallel_encode"]["effective_degree"], 2)
        # Uneven spatial shards (5 latent rows -> 2 + 3) preserve untiled
        # convolution boundary context, posterior moments and decoded frames.
        args.pipeline_config.vae_tiling = False
        # Both uneven shards and larger reduction layouts must stay exact.
        for frames, height in ((9, 160), (17, 320)):
            value = torch.randn(1, 3, frames, height, 192, device="cuda:0", dtype=torch.bfloat16)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                expected = vae.encoder(value)
                actual = tiled_vae(vae, value, args, batch, operation="encode").parameters
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                latents = actual[:, :128]
                expected = vae.decoder(latents, temb)
                actual = tiled_vae(vae, latents, args, batch, operation="decode", temb=temb)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(batch.extra["vae_parallel_encode"]["algorithm"], "spatial_halo_reference")
        with patch.object(vae.encoder.conv_in.conv, "forward", side_effect=RuntimeError("injected VAE")):
            with self.assertRaises(Exception):
                tiled_vae(vae, value, args, batch, operation="encode")
        self.assertFalse(any(hasattr(m, "_parallel_spatial_layout") for m in vae.encoder.modules()))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = vae.encoder(value)
            actual = tiled_vae(vae, value, args, batch, operation="encode").parameters
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
