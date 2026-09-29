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
            actual, expected = actual.reshape(-1), expected.reshape(-1)
            maximum, absolute_sum, different, finite = 0.0, 0.0, 0, True
            for start in range(0, actual.numel(), 1_048_576):
                a, b = actual[start:start + 1_048_576].float(), expected[start:start + 1_048_576].float()
                delta = (a - b).abs()
                finite = finite and bool(torch.isfinite(delta).all())
                maximum = max(maximum, delta.max().item())
                absolute_sum += delta.double().sum().item()
                different += torch.count_nonzero(a != b).item()
            return dict(max_abs=maximum, mean_abs=absolute_sum / actual.numel(),
                        different_elements=different, finite=finite)

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
                        if not item["difference"]["finite"] or item["difference"]["different_elements"]:
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
            self.assertFalse(failures, f"spatial reference path changed output: {failures}")
        finally:
            report["output_mismatches"] = failures
            save()
            del vae
            cleanup()
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(worker, range(4)))


if __name__ == "__main__":
    unittest.main()
