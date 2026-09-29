"""Stage ownership contracts for migrated component placement."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch
from config.server_args import ServerArgs
from memory.adapters.sglang_memory_adapter import SGLangMemoryAdapter
from memory.policies.component_offload import offload_component
from memory.policies.memory_phase_controller import MemoryPhaseController

class ComponentOffloadTests(unittest.TestCase):
    def test_stage_releases_on_success_and_exception(self):
        class Stage:
            @offload_component('transformer')
            def forward(self, batch, args):
                if batch.fail:
                    raise RuntimeError('stage failed')
                return batch
        for fail in (False, True):
            module = Mock()
            args = ServerArgs(device='cpu', dit_cpu_offload=True)
            adapter = SGLangMemoryAdapter({'transformer': module}, args)
            controller = MemoryPhaseController(adapter, rank=0, device='cpu')
            batch = SimpleNamespace(fail=fail, extra={'memory_phase_controller': controller})
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'stage failed'):
                    Stage().forward(batch, args)
            else:
                self.assertIs(Stage().forward(batch, args), batch)
            self.assertIsNone(adapter.active_component_name)
            self.assertEqual(len(module.to.call_args_list), 2)
            self.assertEqual(module.to.call_args_list[-1].args, ('cpu',))
            adapter.shutdown()

    def test_partial_acquire_rolls_back(self):
        module = Mock()
        module.to.side_effect = [RuntimeError('transfer failed'), module]
        adapter = SGLangMemoryAdapter({'vae': module}, ServerArgs(vae_cpu_offload=True))
        with self.assertRaisesRegex(RuntimeError, 'transfer failed'):
            adapter.acquire_component_residency('vae', reason='test')
        self.assertIsNone(adapter.active_component_name)
        self.assertEqual(module.to.call_args_list[-1].args, ('cpu',))
        adapter.shutdown()

    def test_pipeline_initialization_failure_closes_memory(self):
        from pipelines.base import ComposedPipelineBase
        pipeline = Mock(server_args=ServerArgs())
        pipeline.create_pipeline_stages.side_effect = RuntimeError('stage setup failed')
        with self.assertRaisesRegex(RuntimeError, 'stage setup failed'):
            ComposedPipelineBase.__post_init__(pipeline)
        pipeline.close.assert_called_once_with(terminal=True)
