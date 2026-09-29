"""Collective ordering, uneven shards, failure propagation and task isolation."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import json
from pathlib import Path
import tempfile

import torch
from config.server_args import ServerArgs

from models.adapters.eraserdit.mesh import PeerExchange, SequenceRank, resolve_mesh, EraserDiTMeshWindow
from entrypoints.cli.erase_parallel import partition_tasks


class Attention:
    def forward(self, q, k, v, metadata):
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)


class MeshTests(unittest.TestCase):
    def test_cfg_first_step_and_window_rope_lifecycle(self):
        from contextlib import nullcontext, ExitStack
        from config.server_args import ServerArgs, set_global_server_args
        from models.dits.eraserdit_transformer import EraserDiTLTXVideoTransformer3DModel
        set_global_server_args(ServerArgs(device='cpu', attention_backend='sdpa'))
        model = EraserDiTLTXVideoTransformer3DModel(
            in_channels=3, out_channels=1, num_attention_heads=2, attention_head_dim=16,
            cross_attention_dim=32, num_layers=1, caption_channels=16).eval()
        with torch.no_grad(), patch('torch.cuda.device', return_value=nullcontext()), \
                patch('torch.cuda.current_stream'), patch('torch.cuda.synchronize'), \
                patch('torch.autocast', side_effect=lambda *a, **k: nullcontext()):
            for frames in (2, 1):
                hidden = torch.randn(1, 1, frames, 2, 3)
                values = dict(hidden_states=hidden, cond_latents=torch.randn_like(hidden),
                              mask_values=torch.ones_like(hidden), encoder_hidden_states=torch.randn(1, 4, 16),
                              encoder_attention_mask=torch.ones(1, 4), timestep=torch.ones(1),
                              num_frames=frames, height=2, width=3, return_dict=False)
                negative = dict(values, encoder_hidden_states=values['encoder_hidden_states'] + .7)
                expected = model(**values)[0], model(**negative)[0]
                with EraserDiTMeshWindow(model, dict(sp=1, cfg=2, devices=[torch.device('cpu')]*2)) as mesh:
                    neg, pos = mesh.predict(negative, values)
                    torch.testing.assert_close(pos, expected[0], rtol=0, atol=0)
                    torch.testing.assert_close(neg, expected[1], rtol=0, atol=0)
                    with ExitStack() as patches:
                        for rank_model in mesh.models:
                            patches.enter_context(patch.object(rank_model.rope, 'forward',
                                side_effect=AssertionError('RoPE recomputed')))
                        neg, pos = mesh.predict(negative, values)
                        torch.testing.assert_close(pos, expected[0], rtol=0, atol=0)
                        torch.testing.assert_close(neg, expected[1], rtol=0, atol=0)
                self.assertFalse(mesh.rotary)

    def test_four_logical_ranks_cfg_sp_with_real_cpu_transformer(self):
        from contextlib import nullcontext
        from config.server_args import ServerArgs, set_global_server_args
        from models.dits.eraserdit_transformer import EraserDiTLTXVideoTransformer3DModel
        set_global_server_args(ServerArgs(device="cpu", attention_backend="sdpa"))
        torch.manual_seed(42)
        model = EraserDiTLTXVideoTransformer3DModel(in_channels=3, out_channels=1,
            num_attention_heads=4, attention_head_dim=16, cross_attention_dim=64,
            num_layers=2, caption_channels=16).eval()
        values = dict(hidden_states=torch.randn(1, 1, 1, 3, 3), cond_latents=torch.randn(1, 1, 1, 3, 3),
                      mask_values=torch.ones(1, 1, 1, 3, 3), encoder_hidden_states=torch.randn(1, 4, 16),
                      encoder_attention_mask=torch.ones(1, 4), timestep=torch.ones(1), num_frames=1,
                      height=3, width=3, return_dict=False)
        negative = dict(values, encoder_hidden_states=values["encoder_hidden_states"] + 0.7)
        with torch.no_grad():
            expected, expected_neg = model(**values)[0], model(**negative)[0]
        # Exercise the actual two independent SP groups and CFG branch mapping,
        # emulating transport on CPU; this is not a physical four-GPU benchmark.
        with patch("torch.cuda.device", return_value=nullcontext()), patch("torch.cuda.current_stream"), \
             patch("torch.cuda.synchronize"), patch("torch.autocast", side_effect=lambda *a, **k: nullcontext()):
            with EraserDiTMeshWindow(model, dict(sp=2, cfg=2, devices=[torch.device("cpu")] * 4)) as mesh:
                actual_neg, actual = mesh.predict(negative, values)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(actual_neg, expected_neg, rtol=1e-5, atol=1e-6)
                self.assertEqual(len(mesh.groups), 2)
                from contextlib import ExitStack
                with ExitStack() as patches:
                    for rank_model in mesh.models:
                        patches.enter_context(patch.object(rank_model.rope, 'forward',
                            side_effect=AssertionError('mesh recomputed static RoPE')))
                    again_neg, again = mesh.predict(negative, values)
                    torch.testing.assert_close(again, actual, rtol=0, atol=0)
                    torch.testing.assert_close(again_neg, actual_neg, rtol=0, atol=0)
                self.assertEqual(mesh.report()['rotary_cache_entries'], 4)
            self.assertEqual(mesh.rotary, [])

    def test_ulysses_two_and_four_ranks_uneven_sequence(self):
        torch.manual_seed(42)
        q, k, v = [torch.randn(1, 13, 4, 8) for _ in range(3)]
        expected = Attention().forward(q, k, v, None)
        for degree in (2, 4):
            exchange = PeerExchange(degree, timeout=5)
            def work(index):
                rank = SequenceRank(exchange, index)
                shard = rank.partition(q.shape[1])
                # Repeated exchange generations must not read overwritten slots.
                for _ in range(3):
                    output = rank.attention(q[:, shard], k[:, shard], v[:, shard], Attention(), None)
                return output
            with patch("torch.cuda.current_stream"), ThreadPoolExecutor(max_workers=degree) as pool:
                futures = [pool.submit(work, index) for index in range(degree)]
                output = torch.cat([future.result() for future in futures], dim=1)
            torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
            self.assertEqual(exchange.calls, [6] * degree)

    def test_peer_failure_unblocks_waiter(self):
        exchange = PeerExchange(2, timeout=2)
        with patch("torch.cuda.current_stream"), ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(exchange.exchange, 0, (torch.ones(1),), lambda x: x, 0)
            exchange.abort()
            with self.assertRaises(Exception):
                future.result(timeout=3)

    def test_sp_cache_metric_uses_global_sums_and_rejects_divergent_decisions(self):
        from cache.consensus import CacheConsensusStats, CacheDecisionConsensus
        from models.adapters.eraserdit.mesh import PeerCacheCoordinator
        exchange = PeerExchange(2, timeout=5)
        def work(rank):
            consensus = CacheDecisionConsensus(PeerCacheCoordinator(exchange, rank), stats=CacheConsensusStats())
            previous = torch.ones(1 if rank == 0 else 3)
            current = previous * (3 if rank == 0 else 1)
            ratio = consensus.relative_l1(current, previous, context=None)
            with self.assertRaisesRegex(RuntimeError, 'decision mismatch'):
                consensus.assert_decision(rank, device=torch.device('cpu'))
            return ratio
        with patch('torch.cuda.current_stream'), ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(work, rank) for rank in range(2)]
            self.assertEqual([f.result(timeout=10) for f in futures], [.5, .5])

    def test_composable_mesh_and_ring_validation(self):
        from config.eraserdit import EraserDiTPipelineConfig
        args = ServerArgs(device='cuda:0', enable_torch_compile=True,
                          pipeline_config=EraserDiTPipelineConfig(sp_degree=2, sp_attention_mode='ring'))
        with patch('torch.cuda.device_count', return_value=2):
            self.assertEqual(resolve_mesh(args)['attention_mode'], 'ring')
            args.attention_backend = 'sage_attn'
            with self.assertRaisesRegex(ValueError, 'explicit sdpa'):
                resolve_mesh(args)
            args.attention_backend = 'sdpa'
            args.pipeline_config.sp_degree = 1
            with self.assertRaisesRegex(ValueError, 'SP2'):
                resolve_mesh(args)
            args.pipeline_config.sp_attention_mode = 'ulysses'
            args.pipeline_config.vae_degree = 2
            args.dit_layerwise_offload = True
            with self.assertRaisesRegex(ValueError, 'single-GPU'):
                resolve_mesh(args)

    def test_configuration_and_mesh_cardinality(self):
        config = SimpleNamespace(sp_degree=2, cfg_degree=2, vae_degree=4, vae_tiling=True,
                                 parallel_devices=(0, 1, 2, 3), cfg_parallel_device=None)
        args = ServerArgs(pipeline_config=config, device="cuda:0",
                               enable_torch_compile=False)
        with patch("torch.cuda.device_count", return_value=4):
            self.assertEqual(len(resolve_mesh(args)["devices"]), 4)
            config.parallel_devices = (0, 1)
            with self.assertRaises(ValueError):
                resolve_mesh(args)
            config.parallel_devices = (0, 1, 2, 3)
            config.vae_tiling = False
            self.assertEqual(resolve_mesh(args)["vae"], 4)
            config.vae_tiling = True
            self.assertEqual(resolve_mesh(args, SimpleNamespace(transformer_cache_mode="teacache"))["sp"], 2)

    def test_task_partition_has_no_duplicate_or_missing_tasks(self):
        tasks = list(range(7))
        groups = partition_tasks(tasks, 2)
        self.assertEqual(groups, [[0, 2, 4, 6], [1, 3, 5]])
        with self.assertRaises(ValueError):
            partition_tasks(tasks, 8)

    def test_vae_one_tile_fallback_does_not_crop_to_stride(self):
        from config.eraserdit import EraserDiTPipelineConfig
        from models.adapters.eraserdit.vae import tiled_vae
        value = torch.zeros(1, 3, 1, 480, 32)
        vae = SimpleNamespace(spatial_compression_ratio=32, config=SimpleNamespace(decoder_inject_noise=()),
                              encoder=lambda x: torch.cat([x, x], dim=1))
        args = ServerArgs(device="cuda:0", enable_torch_compile=False,
                               pipeline_config=EraserDiTPipelineConfig(vae_degree=2, vae_tiling=True))
        batch = SimpleNamespace(extra={})
        with patch("torch.cuda.device_count", return_value=2):
            posterior = tiled_vae(vae, value, args, batch, operation="encode")
        self.assertEqual(posterior.parameters.shape[-2:], (480, 32))
        self.assertEqual(batch.extra["vae_parallel_encode"]["effective_degree"], 1)

    def test_dispatcher_cancels_owned_peers_after_failure(self):
        from entrypoints.cli.erase_parallel import main
        import signal
        class Worker:
            def __init__(self, code, pid):
                self.returncode, self.pid, self.terminated = code, pid, False
            def poll(self):
                return self.returncode
            def terminate(self):
                self.terminated, self.returncode = True, -15
            def wait(self, timeout=None):
                return self.returncode
        failed, peer = Worker(1, 100), Worker(None, 101)
        def kill_group(pid, sig):
            self.assertEqual(sig, signal.SIGTERM)
            if pid == peer.pid:
                peer.terminate()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            tasks = root / "tasks.json"
            tasks.write_text(json.dumps([{"video": "v.mp4", "mask": "m.mp4", "output": str(root / f"{i}.mp4")}
                                         for i in range(2)]))
            argv = ["erase_parallel", "--dp-degree", "2", "--parallel-run-dir", str(root / "run"),
                    "--task-file", str(tasks), "--model-path", str(root / "model")]
            with patch("sys.argv", argv), patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2,3"}), \
                 patch("subprocess.Popen", side_effect=[failed, peer]) as start, \
                 patch("os.killpg", side_effect=kill_group) as terminate_group:
                with self.assertRaisesRegex(RuntimeError, "worker failed"):
                    main()
                self.assertEqual([c.args[0] for c in terminate_group.call_args_list], [100, 101])
                self.assertTrue(all(c.kwargs['start_new_session'] for c in start.call_args_list))
            self.assertTrue(peer.terminated)
            report = json.loads((root / "run/report.json").read_text())
            self.assertFalse(report["passed"])
            self.assertEqual([w["exit_code"] for w in report["workers"]], [1, -15])


if __name__ == "__main__":
    unittest.main()
