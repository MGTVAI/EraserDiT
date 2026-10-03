"""Ensure performance reports cannot silently include multiple windows."""
import json
from pathlib import Path
import tempfile
import unittest

from entrypoints.cli.benchmark_window import summarize


class WindowBenchmarkTests(unittest.TestCase):
    def test_paired_modes_and_end_to_end_are_separate(self):
        tasks = []
        for mode, seconds in (('off', 10), ('on', 8), ('on', 9), ('off', 11)):
            tasks.append({'id': f'cfg2_sp2_{len(tasks)}_text_{mode}',
                'e2e_seconds_excluding_warmup': seconds + 5,
                'runtime_video_metadata': {'num_frames': 121},
                'timing': {'pure_inference_seconds': seconds,
                    'pure_inference_stage_counts': {'DenoisingStage': 1},
                    'pure_inference_stage_breakdown_ms': {'DenoisingStage': (seconds - 1) * 1000}}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.log'
            path.write_text(json.dumps({'tasks': tasks}, indent=2))
            result = summarize(path)
        self.assertEqual(result['text_on']['pure_seconds']['median'], 8.5)
        self.assertEqual(result['text_off']['pure_seconds']['median'], 10.5)
        self.assertEqual(result['text_on']['e2e_seconds']['median'], 13.5)
        self.assertFalse(result['diagnostic_only'])
        tasks[0]['parallel_history'] = [{'dit_parallel': {'rank_reports': [
            {'diagnostic_profiles': [{'trace': 'diagnostic.trace.json'}]}]}}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'diagnostic.log'
            path.write_text(json.dumps({'tasks': tasks}, indent=2))
            self.assertTrue(summarize(path)['diagnostic_only'])

    def test_stage_timing_excludes_warmup_and_io(self):
        report = {'tasks': []}
        for value in (2.0, 4.0, 3.0):
            report['tasks'].append({
                'elapsed_seconds': 1000, 'warmup': {'duration_seconds': 100},
                'runtime_video_metadata': {'num_frames': 121},
                'timing': {'pure_inference_seconds': value + 1,
                           'pure_inference_stage_counts': {'EraserDiTEraseDenoisingStage': 1},
                           'pure_inference_stage_breakdown_ms': {'EraserDiTEraseDenoisingStage': value * 1000}}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.log'
            path.write_text('warmup and progress logs\n' + json.dumps(report, indent=2))
            result = summarize(path)
            self.assertEqual(result['pure_seconds'], {'median': 4, 'min': 3, 'max': 5})
            self.assertEqual(result['denoise_seconds'], {'median': 3, 'min': 2, 'max': 4})
            report['tasks'][0]['timing']['pure_inference_stage_counts']['EraserDiTEraseDenoisingStage'] = 2
            path.write_text(json.dumps(report, indent=2))
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                summarize(path)
            report['tasks'][0]['timing']['pure_inference_stage_counts']['EraserDiTEraseDenoisingStage'] = 1
            report['tasks'][0]['runtime_video_metadata']['num_frames'] = 145
            path.write_text(json.dumps(report, indent=2))
            with self.assertRaisesRegex(ValueError, '121 frames'):
                summarize(path)
