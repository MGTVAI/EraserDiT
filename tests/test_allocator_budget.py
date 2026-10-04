import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from config.server_args import ServerArgs
from memory.allocator import configure_cuda_allocator, release_idle_cuda_cache


class AllocatorBudgetTests(unittest.TestCase):
    def test_idle_cache_release_is_opt_in(self):
        with patch('torch.cuda.device') as device, patch('torch.cuda.empty_cache') as empty:
            release_idle_cuda_cache(None, 'cuda:0')
            release_idle_cuda_cache(22, 'cpu')
            empty.assert_not_called()
            release_idle_cuda_cache(22, 'cuda:1')
            empty.assert_called_once_with()
            device.assert_called_once_with('cuda:1')

    def test_validation_and_cpu_noop(self):
        for value in (0, -1, math.nan, math.inf, True, '22'):
            with self.assertRaises(ValueError):
                ServerArgs(cuda_memory_limit_gib=value)
        configure_cuda_allocator(None, 'cpu')
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            configure_cuda_allocator(22, 'cpu')

    def test_cap_is_per_device_and_checked_before_applying(self):
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_properties', return_value=SimpleNamespace(total_memory=48*1024**3)), \
             patch('torch.cuda.set_per_process_memory_fraction') as setter:
            configure_cuda_allocator(22, 'cuda:1')
            self.assertEqual(setter.call_args.args[0], 22/48)
            self.assertEqual(str(setter.call_args.args[1]), 'cuda:1')
            with self.assertRaisesRegex(ValueError, 'exceed'):
                configure_cuda_allocator(49, 'cuda:1')
            self.assertEqual(setter.call_count, 1)

    def test_unspecified_cuda_index_uses_current_device(self):
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.current_device', return_value=2), \
             patch('torch.cuda.get_device_properties', return_value=SimpleNamespace(total_memory=48*1024**3)), \
             patch('torch.cuda.set_per_process_memory_fraction') as setter:
            configure_cuda_allocator(22, 'cuda')
            self.assertEqual(str(setter.call_args.args[1]), 'cuda:2')

    def test_cli_and_server_propagate_budget(self):
        from entrypoints.cli.erase_eraserdit import _build_parser, _build_server_args
        from entrypoints.server import serve
        from pipelines.eraserdit_erase_pipeline import EraserDiTErasePipeline
        options = ['--model-path', 'data/model', '--cuda-memory-limit-gib', '22']
        cli = _build_server_args(_build_parser().parse_args(options))
        server = serve._build_server_args(serve._build_parser().parse_args(
            [*options, '--task-root', '/tmp/eraserdit-test-tasks']), EraserDiTErasePipeline)
        self.assertEqual(cli.cuda_memory_limit_gib, 22)
        self.assertEqual(server.cuda_memory_limit_gib, 22)
