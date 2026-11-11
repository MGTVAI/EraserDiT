"""The public benchmark wrapper supplies its metadata observer output path."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from scripts import cpu_pipeline_benchmark as benchmark


class AuditPathTests(unittest.TestCase):
    def test_required_observer_path_is_recorded_for_both_cli_forms(self):
        expected = str(Path('test_run').resolve()/'cpu_pipeline_audit.json')
        for options in (['--run-dir', 'test_run'], ['--run-dir=test_run']):
            with self.subTest(options=options), patch.dict(os.environ, {}, clear=True):
                with patch.object(sys, 'argv', ['benchmark', *options]), \
                        patch.object(benchmark, 'benchmark_environment', return_value={}), \
                        patch.object(benchmark.benchmark_l40s, 'main') as main:
                    original = subprocess.Popen
                    benchmark.main()
                    self.assertEqual(main.call_args.kwargs['extra_env']['ERASERDIT_CPU_PIPELINE_AUDIT'], expected)
                    self.assertIs(subprocess.Popen, original)

    def test_explicit_audit_destination_takes_precedence_and_failure_restores_popen(self):
        with patch.dict(os.environ, {'ERASERDIT_CPU_PIPELINE_AUDIT': 'environment.json'}, clear=True):
            with patch.object(sys, 'argv', ['benchmark', '--run-dir', 'test_run']), \
                    patch.object(benchmark, 'benchmark_environment', return_value={}), \
                    patch.object(benchmark.benchmark_l40s, 'main', side_effect=RuntimeError('injected')) as main:
                original = subprocess.Popen
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    benchmark.main(extra_env={'ERASERDIT_CPU_PIPELINE_AUDIT': 'explicit.json'})
                self.assertEqual(main.call_args.kwargs['extra_env']['ERASERDIT_CPU_PIPELINE_AUDIT'], 'explicit.json')
                self.assertIs(subprocess.Popen, original)


if __name__ == '__main__':
    unittest.main()
