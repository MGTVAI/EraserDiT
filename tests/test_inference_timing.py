"""Diagnostic timing must never synchronize normal inference."""
import os
import unittest
from unittest.mock import patch

from utils.inference_timing import diagnostic_stage_timer
from utils.perf_logger import RequestMetrics


class DiagnosticTimingTests(unittest.TestCase):
    def test_disabled_or_missing_metrics_does_not_sync(self):
        metrics = RequestMetrics('timing')
        for enabled, target in (('0', metrics), ('1', None)):
            with self.subTest(enabled=enabled), patch.dict(os.environ, MGERASE_DIAGNOSTIC_TIMING=enabled):
                with patch('torch.cuda.synchronize') as sync:
                    with diagnostic_stage_timer(target, 'diagnostic.test', device='cuda:1'):
                        pass
                    sync.assert_not_called()
        self.assertEqual(metrics.stages, {})

    def test_enabled_syncs_selected_device_and_records_failure(self):
        metrics = RequestMetrics('timing')
        with patch.dict(os.environ, MGERASE_DIAGNOSTIC_TIMING='1'):
            with patch('torch.cuda.synchronize') as sync:
                with self.assertRaisesRegex(ValueError, 'test failure'):
                    with diagnostic_stage_timer(metrics, 'diagnostic.test', device='cuda:1'):
                        raise ValueError('test failure')
                self.assertEqual(sync.call_count, 2)
                self.assertEqual(str(sync.call_args.args[0]), 'cuda:1')
        self.assertEqual(metrics.stage_counts, {'diagnostic.test': 1})
        self.assertGreaterEqual(metrics.stages['diagnostic.test'], 0)


if __name__ == '__main__':
    unittest.main()
