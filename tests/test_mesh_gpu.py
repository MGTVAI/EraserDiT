"""Explicit two/four GPU model, peer failure and VAE tiling verification."""
import os
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
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


def assert_vae_close(actual, expected):
    """Local BF16 convolutions need not be bitwise identical to full-height ones."""
    actual, expected = actual.float(), expected.float()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.15)
    relative_rmse = (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8)
    if relative_rmse.item() > 0.01:
        raise AssertionError(f"VAE relative RMSE exceeds 1%: {relative_rmse.item()}")


@unittest.skipUnless(os.environ.get("ERASERDIT_TEST_TWO_GPU") == "1", "explicit GPU opt-in")
class VAEHaloTests(unittest.TestCase):
    def test_local_convolution_boundaries_and_reused_storage(self):
        from models.adapters.eraserdit.vae_spatial import SpatialExchange
        from models.vaes.eraserdit_vae import LTXVideoCausalConv3d

        enable_deterministic_mode()
        torch.manual_seed(42)
        # FP32 isolates halo correctness from accumulated BF16 rounding.
        # Non-default streams, uneven shards, one-row shards, global edges,
        # causal/noncausal time padding, and pointwise (no exchange) kernels.
        # Four logical ranks on two GPUs also exercise both-neighbor exchange.
        with torch.backends.cudnn.flags(enabled=True, allow_tf32=False, deterministic=True):
            for causal in (True, False):
                for kernel in (1, 3):
                    model = LTXVideoCausalConv3d(4, 4, kernel, is_causal=causal).cuda(0).eval()
                    peer = LTXVideoCausalConv3d(4, 4, kernel, is_causal=causal).cuda(1).eval()
                    peer.load_state_dict(model.state_dict())
                    for boundaries in ((0, 1, 11), (0, 5, 11), (0, 1, 3, 6, 11)):
                        degree = len(boundaries) - 1
                        value = torch.randn(1, 4, 5, 11, 7, device="cuda:0")
                        for device in (0, 1):
                            torch.cuda.synchronize(device)
                        exchange = SpatialExchange(degree)
                        outputs = [[] for _ in range(degree)]

                        def work(rank):
                            device = torch.device("cuda", rank % 2)
                            stream = torch.cuda.Stream(device=device)
                            with torch.cuda.device(device), torch.cuda.stream(stream), torch.no_grad():
                                local = value[..., boundaries[rank]:boundaries[rank + 1], :]
                                local = local.to(device).clone()
                                component = model if rank % 2 == 0 else peer
                                for step in range(3):
                                    # Reuse source storage after the prior exchange.
                                    local.add_(0.125)
                                    seen_heights = []
                                    def original(x):
                                        seen_heights.append(x.shape[-2])
                                        return component(x)
                                    output = exchange.convolution(rank, local, original, kernel // 2)
                                    self.assertEqual(seen_heights, [local.shape[-2] + 2 * (kernel // 2)])
                                    outputs[rank].append(output)
                                stream.synchronize()

                        model.conv.padding = peer.conv.padding = (0, 0, kernel // 2)
                        with ThreadPoolExecutor(max_workers=degree) as pool:
                            list(pool.map(work, range(degree)))
                        model.conv.padding = peer.conv.padding = (0, kernel // 2, kernel // 2)
                        with torch.no_grad():
                            for step in range(3):
                                value.add_(0.125)
                                expected = model(value)
                                actual = torch.cat([outputs[r][step].to("cuda:0") for r in range(degree)], dim=-2)
                                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                        self.assertEqual(exchange.calls, [3] * degree)


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
        # Local convolution shapes can select different BF16 cuDNN kernels.
        for frames, height in ((9, 160), (17, 320)):
            value = torch.randn(1, 3, frames, height, 192, device="cuda:0", dtype=torch.bfloat16)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                expected = vae.encoder(value)
                actual = tiled_vae(vae, value, args, batch, operation="encode").parameters
                assert_vae_close(actual, expected)
                latents = actual[:, :128]
                expected = vae.decoder(latents, temb)
                actual = tiled_vae(vae, latents, args, batch, operation="decode", temb=temb)
            # Random-image encoder means are a numerically sensitive decoder
            # stress input. Compare both BF16 paths to FP32, rather than treating
            # one BF16 kernel's rounding as ground truth or widening allclose.
            with torch.no_grad(), torch.backends.cudnn.flags(enabled=True, allow_tf32=False, deterministic=True):
                try:
                    vae.decoder.float()
                    precise = vae.decoder(latents.float(), temb.float())
                finally:
                    vae.decoder.bfloat16()
            baseline_error = (expected.float() - precise).square()
            parallel_error = (actual.float() - precise).square()
            self.assertTrue(torch.isfinite(parallel_error).all().item())
            self.assertLessEqual(parallel_error.mean().sqrt().item(),
                                 baseline_error.mean().sqrt().item() * 1.1 + 1e-6)
            self.assertLessEqual(parallel_error.max().sqrt().item(),
                                 baseline_error.max().sqrt().item() * 1.25 + 1e-6)
            boundary = height // 32 // 2 * 32
            self.assertLessEqual(parallel_error[..., boundary-1:boundary+1, :].mean().sqrt().item(),
                                 baseline_error[..., boundary-1:boundary+1, :].mean().sqrt().item() * 1.25 + 1e-6)
        self.assertEqual(batch.extra["vae_parallel_encode"]["algorithm"], "spatial_halo")
        with patch.object(vae.encoder.conv_in.conv, "forward", side_effect=RuntimeError("injected VAE")):
            with self.assertRaises(Exception):
                tiled_vae(vae, value, args, batch, operation="encode")
        from models.vaes.eraserdit_vae import LTXVideoCausalConv3d
        for module in vae.encoder.modules():
            if isinstance(module, LTXVideoCausalConv3d):
                self.assertNotIn("forward", module.__dict__)
                self.assertEqual(module.conv.padding[1], module.kernel_size[1] // 2)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = vae.encoder(value)
            actual = tiled_vae(vae, value, args, batch, operation="encode").parameters
            assert_vae_close(actual, expected)


@unittest.skipUnless(os.environ.get('ERASERDIT_TEST_MESH') == '1' and torch.cuda.device_count() >= 4,
                     'requires opt-in and four visible CUDA GPUs')
class PeerExchangeGpuTests(unittest.TestCase):
    def test_strided_uneven_shards_and_reused_source_storage(self):
        # Two independent CFG groups and one SP4 group reuse source storage.
        # Large QKV -> sequence gather and strided head gather exercise direct
        # writes; the next small shape must return to the cat path.
        for degree, length in ((2, 32769), (2, 101), (4, 32769), (4, 101)):
            groups = [PeerExchange(degree) for _ in range(4 // degree)]
            def worker(index):
                with torch.cuda.device(index), torch.no_grad():
                    rank, group = index % degree, groups[index // degree]
                    heads = 32 // degree
                    start, end = length * rank // degree, length * (rank + 1) // degree
                    values = tuple(torch.empty((1, end-start, 32, 64), dtype=torch.bfloat16,
                                               device=f'cuda:{index}') for _ in range(3))
                    for generation in range(3):
                        for k, value in enumerate(values):
                            value.fill_(index + generation * 8 + k / 4)
                        parts = group.exchange(rank, values, lambda x: x[:, :, rank*heads:(rank+1)*heads], 1)
                        for k, value in enumerate(parts):
                            for peer in range(degree):
                                a, b = length*peer//degree, length*(peer+1)//degree
                                self.assertTrue(torch.all(value[:, a:b] == index-rank + peer + generation*8 + k/4).item())
                        result = group.exchange(rank, (parts[0],), lambda x: x[:, start:end], 2)[0]
                        torch.testing.assert_close(result, values[0], rtol=0, atol=0)
                    self.assertEqual(group.calls[rank], 6)
                    self.assertEqual(group.direct_copies[rank], 12 if length > 1000 else 0)
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(worker, range(4)))


@unittest.skipUnless(os.environ.get("ERASERDIT_TEST_VAE_BENCHMARK") == "1",
                     "explicit real-weight VAE benchmark opt-in")
class VAESpatialBenchmarkTests(unittest.TestCase):
    """Paired untiled VAE measurements; performance is reported, not asserted.

    Optional CPU fixtures contain encoder input, or decoder (latent, timestep).
    Loading, fixture transfer, output comparison and allocator cleanup are outside
    timing. The production adapter's replica setup and communication are included.
    """

    def test_single_vs_two_gpu(self):
        from models.vaes.eraserdit_vae import EraserDiTAutoencoderKLLTXVideo
        from loader.meta_load import load_safetensors_model

        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        model_path = Path(os.environ.get("ERASERDIT_TEST_MODEL", "data/model")) / "vae"
        report_path = Path(os.environ.get("ERASERDIT_VAE_REPORT", "/tmp/eraserdit_vae_benchmark.json"))
        frames, height, width = map(int, os.environ.get("ERASERDIT_VAE_SHAPE", "17,320,192").split(","))
        repeats = int(os.environ.get("ERASERDIT_VAE_REPEATS", "3"))
        self.assertGreaterEqual(repeats, 1)
        self.assertEqual((frames - 1) % 8, 0)
        self.assertEqual(height % 32, 0)
        self.assertEqual(width % 32, 0)
        profile = enable_deterministic_mode()
        config = json.loads((model_path / "config.json").read_text())
        vae, _ = load_safetensors_model(
            lambda: EraserDiTAutoencoderKLLTXVideo.from_config(config), model_path,
            dtype=torch.bfloat16, device="cuda:0",
        )
        args = ServerArgs(device="cuda:0", pipeline_config=EraserDiTPipelineConfig())
        set_global_server_args(args)
        devices = (0, 1)
        report = dict(
            torch=torch.__version__, profile=profile,
            visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            devices=[torch.cuda.get_device_name(d) for d in devices],
            model_path=str(model_path.resolve()), shape=[frames, height, width],
            weight_bytes=sum(t.numel() * t.element_size() for t in vae.parameters()),
            source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
                "models/vaes/eraserdit_vae.py", "models/adapters/eraserdit/vae.py",
                "models/adapters/eraserdit/vae_spatial.py", "tests/test_mesh_gpu.py")},
            scope="isolated VAE, BF16, no tiling/compile/offload; includes replica setup",
            tolerances=dict(rtol=0.02, atol=0.15, relative_rmse=0.01, seam_relative_rmse=0.02, ssim_min=0.99),
            stages={},
        )

        def save():
            report_path.write_text(json.dumps(report, indent=2))

        def gpu_usage():
            return subprocess.check_output([
                "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader"], text=True).strip()

        def cleanup():
            gc.collect()
            for device in devices:
                torch.cuda.synchronize(device)
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

        def measure(operation, degree, value, temb):
            args.pipeline_config.vae_degree = degree
            batch = SimpleNamespace(extra={}, transformer_cache_mode="off", cache_text_projections=False)
            cleanup()
            before = [torch.cuda.memory_allocated(d) for d in devices]
            for device in devices:
                torch.cuda.reset_peak_memory_stats(device)
            usage_before = gpu_usage()
            started = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = tiled_vae(vae, value, args, batch, operation=operation, temb=temb)
            for device in devices:
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            tensor = output.parameters if operation == "encode" else output
            peaks = [torch.cuda.max_memory_allocated(d) for d in devices]
            if degree == 2:
                metadata = batch.extra[f"vae_parallel_{operation}"]
                self.assertEqual(metadata["effective_degree"], 2)
                self.assertEqual(metadata["algorithm"], "spatial_halo")
                self.assertGreater(metadata["convolutions_per_rank"][0], 0)
                self.assertEqual(*metadata["convolutions_per_rank"])
            item = dict(
                degree=degree, seconds=seconds, baseline_allocated_bytes=before,
                peak_allocated_bytes=peaks,
                peak_increment_bytes=[p - b for p, b in zip(peaks, before)],
                peak_reserved_bytes=[torch.cuda.max_memory_reserved(d) for d in devices],
                output_shape=list(tensor.shape),
                adapter=batch.extra.get(f"vae_parallel_{operation}"),
                gpu_processes_before=usage_before, gpu_processes_after=gpu_usage(),
            )
            # Baselines live on CPU so they cannot inflate the following GPU peak.
            result = tensor.detach().cpu()
            del output, tensor
            return result, item

        def difference(actual, expected):
            self.assertEqual(actual.shape, expected.shape)
            # Include the partition boundary explicitly; an average over the
            # entire video can otherwise conceal a narrow seam.
            boundary = (height // 32 // 2) * (32 if actual.shape[-2] == height else 1)
            seam_a = actual[..., max(0, boundary - 1):boundary + 1, :].float()
            seam_b = expected[..., max(0, boundary - 1):boundary + 1, :].float()
            seam_rmse = (seam_a - seam_b).square().mean().sqrt().item()
            actual, expected = actual.reshape(-1), expected.reshape(-1)
            maximum, absolute_sum, different, finite = 0.0, 0.0, 0, True
            squared_sum, reference_squared_sum, outside_tolerance = 0.0, 0.0, 0
            for start in range(0, actual.numel(), 1_048_576):
                a, b = actual[start:start + 1_048_576].float(), expected[start:start + 1_048_576].float()
                delta = (a - b).abs()
                finite = finite and bool(torch.isfinite(delta).all())
                maximum = max(maximum, delta.max().item())
                absolute_sum += delta.double().sum().item()
                squared_sum += delta.double().square().sum().item()
                reference_squared_sum += b.double().square().sum().item()
                outside_tolerance += torch.count_nonzero(delta > 0.15 + 0.02 * b.abs()).item()
                different += torch.count_nonzero(a != b).item()
            reference_rms = max((reference_squared_sum / actual.numel()) ** 0.5, 1e-8)
            return dict(max_abs=maximum, mean_abs=absolute_sum / actual.numel(),
                        relative_rmse=(squared_sum / actual.numel()) ** 0.5 / reference_rms,
                        seam_relative_rmse=seam_rmse / reference_rms,
                        outside_tolerance=outside_tolerance,
                        different_elements=different, finite=finite)

        def acceptable(item, degree):
            if not item["finite"]:
                return False
            if degree == 1:
                return item["different_elements"] == 0
            return (item["outside_tolerance"] == 0 and item["relative_rmse"] <= 0.01
                    and item["seam_relative_rmse"] <= 0.02)

        def decoded_quality(actual, expected):
            import cv2
            import numpy as np
            cv2.setNumThreads(1)
            scores, seams = [], []
            boundary = height // 32 // 2 * 32
            for frame in range(actual.shape[2]):
                a, b = [(v[0, :, frame].float().clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5
                        for v in (actual, expected)]
                def blur(x):
                    return cv2.GaussianBlur(x, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)
                ma, mb = blur(a), blur(b)
                va, vb, cov = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
                ssim = ((2 * ma * mb + 2.55**2) * (2 * cov + 7.65**2)
                        / ((ma * ma + mb * mb + 2.55**2) * (va + vb + 7.65**2)))
                scores.append(float(ssim.mean(dtype=np.float64)))
                seams.append(float(ssim[max(0, boundary - 16):boundary + 16].mean(dtype=np.float64)))
            return dict(mean_ssim=statistics.mean(scores), min_frame_ssim=min(scores),
                        seam_ssim=statistics.mean(seams))

        failures = []
        try:
            for operation in ("encode", "decode"):
                fixture = os.environ.get(f"ERASERDIT_VAE_{operation.upper()}_INPUT")
                if fixture:
                    data = torch.load(fixture, map_location="cpu", weights_only=True)
                    if operation == "encode":
                        value, temb = data, None
                    else:
                        value, temb = data
                    value = value.to(device="cuda:0", dtype=torch.bfloat16)
                    temb = temb.to(device="cuda:0", dtype=torch.bfloat16) if temb is not None else None
                    del data
                else:
                    generator = torch.Generator(device="cuda:0").manual_seed(42)
                    shape = ((1, 3, frames, height, width) if operation == "encode" else
                             (1, 128, (frames - 1) // 8 + 1, height // 32, width // 32))
                    value = torch.randn(shape, generator=generator, device="cuda:0", dtype=torch.bfloat16)
                    temb = torch.zeros(1, device="cuda:0", dtype=torch.bfloat16) if operation == "decode" else None
                stage = dict(input_shape=list(value.shape), fixture=fixture, warmup=[], samples=[])
                report["stages"][operation] = stage
                reference = None
                for degree in (1, 2):
                    actual, item = measure(operation, degree, value, temb)
                    if reference is None:
                        reference = actual
                    else:
                        item["difference"] = difference(actual, reference)
                        if not acceptable(item["difference"], degree):
                            failures.append((operation, degree, "warmup", item["difference"]))
                        if operation == "decode":
                            stage["quality"] = decoded_quality(actual, reference)
                            if min(stage["quality"].values()) < 0.99:
                                failures.append((operation, degree, "quality", stage["quality"]))
                    stage["warmup"].append(item)
                    del actual
                    save()
                    print("VAE_WARMUP", operation, degree, round(item["seconds"], 3), flush=True)
                for repeat in range(repeats):
                    for degree in ((1, 2) if repeat % 2 == 0 else (2, 1)):
                        actual, item = measure(operation, degree, value, temb)
                        item["repeat"] = repeat
                        item["difference"] = difference(actual, reference)
                        stage["samples"].append(item)
                        if not acceptable(item["difference"], degree):
                            failures.append((operation, degree, repeat, item["difference"]))
                        del actual
                        save()
                        print("VAE_SAMPLE", operation, degree, round(item["seconds"], 3),
                              [round(p / 1024**3, 3) for p in item["peak_allocated_bytes"]],
                              item["difference"], flush=True)
                summary = {}
                for degree in (1, 2):
                    samples = [s for s in stage["samples"] if s["degree"] == degree]
                    times = [s["seconds"] for s in samples]
                    summary[str(degree)] = dict(
                        median_seconds=statistics.median(times), range_seconds=[min(times), max(times)],
                        peak_allocated_bytes=[max(s["peak_allocated_bytes"][d] for s in samples) for d in devices],
                    )
                stage["summary"] = summary
                save()
                del reference, value, temb
                cleanup()
            self.assertFalse(failures, f"spatial VAE exceeded numerical tolerances: {failures}")
        finally:
            report["output_mismatches"] = failures
            save()
            del vae
            cleanup()


if __name__ == "__main__":
    unittest.main()
