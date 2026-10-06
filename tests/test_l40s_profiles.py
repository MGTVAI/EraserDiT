"""Validate the generated benchmark CLI, including resident VAE comparisons."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from entrypoints.cli.benchmark_l40s import main
from entrypoints.cli.erase_eraserdit import _build_parser, _build_server_args
from memory.validation import validate_memory_config


class L40SProfileTests(unittest.TestCase):
    def prepare(self, root, profile, degree, extra=()):
        directory = Path(root) / profile
        argv = ['benchmark_l40s', '--run-dir', str(directory), '--profile', profile,
                '--devices', ','.join(map(str, range(degree))), '--repeats', '1',
                '--prepare-only', *extra]
        with patch('sys.argv', argv):
            main()
        manifest = json.loads((directory / 'manifest.json').read_text())
        with patch('torch.cuda.device_count', return_value=degree):
            args = _build_server_args(_build_parser().parse_args(manifest['command'][4:]))
            validate_memory_config(args)
        return args, manifest

    def test_all_vae_profiles_generate_supported_paired_configurations(self):
        with tempfile.TemporaryDirectory() as root:
            for profile, cfg, sp, vae in (
                ('v0', 2, 1, 1), ('v1', 2, 1, 2),
                ('v2', 1, 2, 1), ('v3', 1, 2, 2),
                ('v4', 2, 2, 1), ('v5', 2, 2, 4),
            ):
                with self.subTest(profile=profile):
                    args, manifest = self.prepare(root, profile, cfg * sp)
                    config = args.pipeline_config
                    self.assertEqual((config.cfg_degree, config.sp_degree, config.vae_degree),
                                     (cfg, sp, vae))
                    self.assertFalse(args.resolve_resource_policy().enabled)
                    self.assertFalse(config.vae_tiling)
                    self.assertTrue(config.vae_low_memory)
                    self.assertEqual(args.cuda_memory_limit_gib, 44.)
                    self.assertEqual(manifest['memory_target_gib'], 46.)

    def test_single_gpu_budget_and_explicit_overrides_are_preserved(self):
        with tempfile.TemporaryDirectory() as root, patch('torch.cuda.is_available', return_value=True):
            args, manifest = self.prepare(root, 's3', 1)
            self.assertEqual(args.cuda_memory_limit_gib, 22.)
            self.assertEqual(manifest['memory_target_gib'], 24.)
            args, manifest = self.prepare(root, 'v1', 2,
                ['--allocator-gib', '40', '--memory-target-gib', '43'])
            self.assertEqual(args.cuda_memory_limit_gib, 40.)
            self.assertEqual(manifest['memory_target_gib'], 43.)

    def test_original_parallel_vae_offload_conflict_still_fails_early(self):
        options = ['--model-path', 'data/model', '--dit-parallel-backend', 'nccl',
                   '--cfg-degree', '2', '--vae-degree', '2',
                   '--no-dit-layerwise-offload', '--no-dit-cpu-offload',
                   '--text-encoder-cpu-offload', '--vae-cpu-offload']
        with patch('torch.cuda.device_count', return_value=2), \
             patch('torch.cuda.is_available', return_value=True):
            args = _build_server_args(_build_parser().parse_args(options))
            with self.assertRaisesRegex(ValueError, 'offload currently requires single-GPU'):
                validate_memory_config(args)

    def test_aligned_profiles_keep_pure_sp_topology(self):
        with tempfile.TemporaryDirectory() as root:
            for degree in (2, 4):
                args, _ = self.prepare(root, f'sp{degree}_aligned', degree)
                config = args.pipeline_config
                self.assertEqual((config.cfg_degree, config.sp_degree, config.sp_linear_mode),
                                 (1, degree, 'aligned'))

    def test_native_rms_profiles_generate_supported_configurations(self):
        from config.dit_parallel import validate_nccl_fusion
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as root:
            for degree in (2, 4):
                args, _ = self.prepare(root, f'sp{degree}_native_rms', degree)
                config = args.pipeline_config
                self.assertEqual((config.cfg_degree, config.sp_degree, config.sp_linear_mode),
                                 (1, degree, 'aligned'))
                self.assertEqual(args.operator_fusion_ops,
                                 'qk_rmsnorm_rope_native,rmsnorm_adaln_native')
                topology = SimpleNamespace(tp=1, ring=1, replicas=1, sp=degree)
                validate_nccl_fusion(args, topology)
                args.operator_fusion_ops = 'qk_rmsnorm_rope_fast'
                with self.assertRaisesRegex(ValueError, 'native reductions only'):
                    validate_nccl_fusion(args, topology)


if __name__ == '__main__':
    unittest.main()
