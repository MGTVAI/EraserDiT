import unittest
from unittest.mock import Mock

from entrypoints.server.control import ServiceProgressState
from entrypoints.server.task import TaskPhase


class ServiceProgressTests(unittest.TestCase):
    def test_steps_advance_within_each_window_without_claiming_completion(self):
        store=Mock()
        progress=ServiceProgressState(store,'task')
        progress.add_pipeline_task(2)
        for window in range(2):
            progress.reset_denoise_task(object_index=0,object_count=1,
                window_index=window,window_count=2,total_steps=10)
            for step in range(10):
                progress.update_denoise(step,10,timestep_value=None)
            progress.update_pipeline(window+1,2,object_index=0,object_count=1,
                window_index=window,window_count=2)
        calls=store.update_progress.call_args_list
        values=[c.kwargs['progress'] for c in calls]
        self.assertEqual(values,sorted(values))
        self.assertGreater(len(set(values)),10)
        self.assertGreater(values[0],1)
        self.assertEqual(values[-1],94)
        self.assertTrue(all(c.kwargs['phase'] is TaskPhase.PROCESSING for c in calls))
        self.assertEqual(calls[-2].kwargs['window_index'],1)
        self.assertLess(values[-2],values[-1])
