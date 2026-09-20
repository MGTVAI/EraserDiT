"""CPU regressions for phase boundaries; optional CUDA numerical checks."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from config.server_args import ServerArgs
from memory.policies.component_offload import offload_component


class Stage:
    @offload_component('transformer')
    def forward(self, batch, server_args):
        batch.calls.append('forward')
        if batch.fail:
            raise RuntimeError('stage failed')
        return batch


class RecordingModule:
    def __init__(self, calls, fail_onload=False):
        self.calls = calls
        self.fail_onload = fail_onload

    def to(self, *, device):
        self.calls.append(str(device))
        if self.fail_onload and str(device) == 'cuda:0':
            raise RuntimeError('transfer failed')
        return self


class ComponentOffloadTests(unittest.TestCase):
    def run_stage(self, policy, *, fail=False, fail_onload=False):
        calls = []
        batch = SimpleNamespace(
            modules={'transformer': RecordingModule(calls, fail_onload)},
            calls=calls, fail=fail,
        )
        args = ServerArgs(resource_policy=policy, device='cuda:0')
        if fail or fail_onload:
            with self.assertRaises(RuntimeError):
                Stage().forward(batch, args)
        else:
            self.assertIs(Stage().forward(batch, args), batch)
        return calls

    def test_resident_path_does_not_transfer(self):
        self.assertEqual(self.run_stage('fullgpu'), ['forward'])

    def test_offload_at_stage_boundary(self):
        self.assertEqual(self.run_stage('component_offload'), ['cuda:0', 'forward', 'cpu'])

    def test_exception_returns_weights_to_cpu(self):
        self.assertEqual(self.run_stage('component_offload', fail=True), ['cuda:0', 'forward', 'cpu'])

    def test_partial_onload_returns_weights_to_cpu(self):
        self.assertEqual(self.run_stage('component_offload', fail_onload=True), ['cuda:0', 'cpu'])

    def test_policy_loads_all_components_on_cpu(self):
        for available in (False, True):
            with patch('torch.cuda.is_available', return_value=available):
                policy = ServerArgs(resource_policy='component_offload').resolve_resource_policy()
            self.assertTrue(policy.text_encoder_cpu_offload)
            self.assertTrue(policy.vae_cpu_offload)
            self.assertTrue(policy.dit_cpu_offload)
            self.assertFalse(policy.dynamic_offload)

    def test_pipeline_rejects_unsupported_modes_before_loading(self):
        from pipelines.eraserdit_erase_pipeline import EraserDiTErasePipeline
        from nodes.composed_pipeline_base import ComposedPipelineBase

        pipeline = object.__new__(EraserDiTErasePipeline)
        with patch.object(ComposedPipelineBase, 'load_modules') as loader:
            with self.assertRaises(ValueError):
                pipeline.load_modules(ServerArgs(resource_policy='dynamic_offload'))
            with self.assertRaises(ValueError):
                pipeline.load_modules(ServerArgs(resource_policy='component_offload', enable_torch_compile=True))
            loader.assert_not_called()

    def test_preloaded_components_are_offloaded(self):
        from pipelines.eraserdit_erase_pipeline import EraserDiTErasePipeline
        from nodes.composed_pipeline_base import ComposedPipelineBase

        pipeline = object.__new__(EraserDiTErasePipeline)
        calls = []
        modules = {name: RecordingModule(calls) for name in ('text_encoder', 'vae', 'transformer')}
        with patch.object(ComposedPipelineBase, 'load_modules', return_value=modules):
            result = pipeline.load_modules(ServerArgs(resource_policy='component_offload'), modules)
        self.assertIs(result, modules)
        self.assertEqual(calls, ['cpu', 'cpu', 'cpu'])

    def test_unintegrated_pipeline_rejects_component_policy(self):
        with self.assertRaises(ValueError):
            ServerArgs(
                resource_policy='component_offload',
                pipeline_class_name='LTX095ErasePipeline',
            ).resolve_resource_policy()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_repeated_gpu_transfer_preserves_outputs_and_tied_weights(self):
        class Tied(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Linear(8, 8)
                self.b = torch.nn.Linear(8, 8)
                self.b.weight = self.a.weight
                self.register_buffer('offset', torch.ones(8))

            def forward(self, value):
                return self.b(self.a(value)) + self.offset

        class Compute:
            @offload_component('transformer')
            def forward(self, batch, args):
                return batch.modules['transformer'](batch.value)

        model = Tied().eval().cuda()
        batch = SimpleNamespace(modules={'transformer': model}, value=torch.ones(2, 8, device='cuda'))
        args = ServerArgs(resource_policy='component_offload', device='cuda')
        with torch.no_grad():
            reference = model(batch.value)
            model.cpu()
            for _ in range(3):
                result = Compute().forward(batch, args)
                torch.testing.assert_close(reference, result, rtol=0, atol=0)
                self.assertIs(model.a.weight, model.b.weight)
                self.assertEqual(model.offset.device.type, 'cpu')
                self.assertTrue(all(p.device.type == 'cpu' for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
