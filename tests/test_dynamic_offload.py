"""Budget, asynchronous lifetime, failure recovery and storage regression tests."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from config.server_args import ServerArgs
from memory.adapters.eraserdit_memory_adapter import EraserDiTMemoryAdapter
from memory.backends.event_type import EventType
from memory.backends.federated_storage import FederatedStorage
from memory.backends.flexible_memory_device_state import FlexibleMemoryDeviceState
from memory.backends.flexible_memory_states import FlexibleMemoryState
from memory.backends.op_event import OPEvent
from memory.policies.component_offload import offload_component
from memory.policies.memory_phase_controller import MemoryPhaseController


class EventQueueTests(unittest.TestCase):
    def test_retiring_old_event_preserves_new_transfers(self):
        state = FlexibleMemoryDeviceState(torch.device('cuda:0'), initialize_cuda=False)
        self.addCleanup(state.release)
        events = [OPEvent(kind, 8, start_cuda_event=object(), end_cuda_event=object())
                  for kind in (EventType.HTOD, EventType.DTOH, EventType.HTOD)]
        for event in events:
            state.add_op_event(event)
        state.remove_op_event_to_timestemp(events[1].timestamp)
        self.assertEqual(state.flexible_usage_bytes, 0)
        state.remove_op_event_to_timestemp(events[1].timestamp)
        self.assertEqual(list(state.event_queue), [events[2]])
        self.assertEqual(state.flexible_wait_onload, 8)

    def test_direct_parameter_aliases_survive_storage_roundtrip(self):
        module = nn.Module()
        module.weight = nn.Parameter(torch.arange(8.0))
        module.alias = module.weight
        storage = FederatedStorage.register_federate_module(module, use_pin_memory=False)
        self.assertIs(module.weight, module.alias)
        storage.release()
        self.assertIs(module.weight, module.alias)
        torch.testing.assert_close(module.weight, torch.arange(8.0))


class Chain(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2048, 2048, bias=False) for _ in range(3)])
        self.register_buffer('offset', torch.ones(2048))
        self.fail = False

    def forward(self, value):
        for layer in self.layers:
            value = layer(value)
        if self.fail:
            raise RuntimeError('injected compute failure')
        return value + self.offset


class Compute:
    @offload_component('transformer')
    def forward(self, batch, args):
        return batch.modules['transformer'](batch.value)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
class DynamicOffloadTests(unittest.TestCase):
    def prepare(self, budget):
        model = Chain().eval().cuda()
        value = torch.ones(2, 2048, device='cuda')
        with torch.no_grad():
            reference = model(value)
        model.cpu()
        adapter = EraserDiTMemoryAdapter()
        self.addCleanup(adapter.close)
        adapter.register(modules={'transformer': model}, device=torch.device('cuda'),
                         dynamic_offload=True, pin_memory=True,
                         max_weight_usage=budget, rank=0)
        args = ServerArgs(resource_policy='dynamic_offload', device='cuda', max_weight_usage=budget)
        return model, value, reference, adapter, args

    def run_request(self, model, value, adapter, args):
        controller = MemoryPhaseController(adapter, rank=0, device='cuda')
        batch = SimpleNamespace(modules={'transformer': model}, value=value,
                                extra={'memory_phase_controller': controller})
        with torch.no_grad():
            try:
                return Compute().forward(batch, args)
            finally:
                controller.close()

    def assert_idle(self, model, adapter, budget):
        self.assertTrue(all(p.device.type == 'cpu' for p in model.parameters()))
        self.assertTrue(all(b.device.type == 'cpu' for b in model.buffers()))
        state = adapter.snapshot()['flexible_state']
        self.assertEqual(state['event_queue_size'], 0)
        self.assertEqual(state['flexible_usage_bytes'], 0)
        self.assertEqual(state['resident_bytes'], 0)
        self.assertLessEqual(state['peak_flexible_usage_bytes'], budget)
        self.assertIsNone(adapter.active_component_name)

    def test_layerwise_budget_and_repeated_requests(self):
        budget = 20 * 1024**2
        model, value, reference, adapter, args = self.prepare(budget)
        for _ in range(3):
            torch.testing.assert_close(self.run_request(model, value, adapter, args), reference, rtol=0, atol=0)
            self.assert_idle(model, adapter, budget)
        report = adapter.snapshot()['component_residency']['transformer']
        self.assertEqual(report['fallback_reasons']['budget_insufficient'], 3)
        self.assertEqual(report['total_onload_count'], 9)
        adapter.close()
        self.assertFalse(any(hasattr(m, 'flexible_extent') for m in model.modules()))

    def test_component_residency_when_budget_fits(self):
        budget = 64 * 1024**2
        model, value, reference, adapter, args = self.prepare(budget)
        torch.testing.assert_close(self.run_request(model, value, adapter, args), reference, rtol=0, atol=0)
        self.assert_idle(model, adapter, budget)
        report = adapter.snapshot()['component_residency']['transformer']
        self.assertEqual(report['component_acquire_count'], 1)
        self.assertGreater(report['resident_fast_path_forward_count'], 0)

    def test_layer_exception_recovers_for_next_request(self):
        budget = 20 * 1024**2
        model, value, reference, adapter, args = self.prepare(budget)
        # Fail *inside* an extent after it has submitted a CUDA operation.
        extent = model.layers[0].flexible_extent
        original = extent._original_forward
        def fail(value):
            original(value)
            raise RuntimeError('injected layer failure')
        with patch.object(extent, '_original_forward', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'injected layer failure'):
                self.run_request(model, value, adapter, args)
        self.assert_idle(model, adapter, budget)
        torch.testing.assert_close(self.run_request(model, value, adapter, args), reference, rtol=0, atol=0)

    def test_partial_transfer_recovers_for_next_request(self):
        budget = 20 * 1024**2
        model, value, reference, adapter, args = self.prepare(budget)
        storage = model.layers[0].flexible_extent._federated_storage
        original = storage.htod
        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError('injected transfer failure')
        with patch.object(storage, 'htod', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'injected transfer failure'):
                self.run_request(model, value, adapter, args)
        self.assert_idle(model, adapter, budget)
        torch.testing.assert_close(self.run_request(model, value, adapter, args), reference, rtol=0, atol=0)

    def test_nondefault_caller_stream(self):
        budget = 20 * 1024**2
        model, value, reference, adapter, args = self.prepare(budget)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            value = torch.ones_like(value)
            actual = self.run_request(model, value, adapter, args)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        self.assert_idle(model, adapter, budget)

    def test_registration_failure_releases_wrappers_and_device_state(self):
        from memory.backends.flexible_module_extent_cuda_async import FlexibleModuleExtentCudaAsync
        model = Chain().eval()
        adapter = EraserDiTMemoryAdapter()
        register = FlexibleModuleExtentCudaAsync.register_module_flexible_extent
        count = 0
        def fail_second(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError('injected registration failure')
            return register(*args, **kwargs)
        with patch.object(FlexibleModuleExtentCudaAsync, 'register_module_flexible_extent', side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, 'injected registration failure'):
                adapter.register(modules={'transformer': model}, device=torch.device('cuda'),
                                 dynamic_offload=True, pin_memory=True,
                                 max_weight_usage=20 * 1024**2, rank=0)
        self.assertFalse(any(hasattr(m, 'flexible_extent') for m in model.modules()))
        self.assertNotIn(torch.cuda.current_device(), FlexibleMemoryState.device_states)

    def test_small_unwrapped_model_releases_device_state(self):
        model = nn.Linear(4, 4).eval()
        adapter = EraserDiTMemoryAdapter()
        adapter.register(modules={'transformer': model}, device=torch.device('cuda'),
                         dynamic_offload=True, pin_memory=False,
                         max_weight_usage=20 * 1024**2, rank=0)
        self.assertEqual(adapter.snapshot()['registered_extent_count'], 0)
        adapter.close()
        self.assertNotIn(torch.cuda.current_device(), FlexibleMemoryState.device_states)

    def test_cross_extent_alias_rejected_before_registration(self):
        model = Chain().eval()
        model.layers[1].weight = model.layers[0].weight
        adapter = EraserDiTMemoryAdapter()
        with self.assertRaisesRegex(ValueError, 'shared tensor crosses'):
            adapter.register(modules={'transformer': model}, device=torch.device('cuda'),
                             dynamic_offload=True, pin_memory=False,
                             max_weight_usage=20 * 1024**2, rank=0)
        self.assertFalse(any(hasattr(m, 'flexible_extent') for m in model.modules()))

    def test_too_small_budget_rejected_before_registration(self):
        model = Chain().eval()
        adapter = EraserDiTMemoryAdapter()
        with self.assertRaisesRegex(ValueError, 'exceeding'):
            adapter.register(modules={'transformer': model}, device=torch.device('cuda'),
                             dynamic_offload=True, pin_memory=False,
                             max_weight_usage=1024, rank=0)
        self.assertFalse(any(hasattr(m, 'flexible_extent') for m in model.modules()))

    def test_t5_distinct_embedding_objects_keep_shared_weights(self):
        from transformers import T5Config, T5EncoderModel
        text = T5EncoderModel(T5Config(
            vocab_size=65536, d_model=64, d_ff=64, d_kv=16, num_heads=4,
            num_layers=1, dropout_rate=0,
        )).eval().cuda()
        tokens = torch.ones(1, 4, dtype=torch.long, device='cuda')
        with torch.no_grad():
            reference = text(tokens).last_hidden_state
        text.cpu()
        original = text.encoder.embed_tokens
        adapter = EraserDiTMemoryAdapter()
        self.addCleanup(adapter.close)
        adapter.register(modules={'text_encoder': text}, device=torch.device('cuda'),
                         dynamic_offload=True, pin_memory=True,
                         max_weight_usage=20 * 1024**2, rank=0)
        class TextStage:
            @offload_component('text_encoder')
            def forward(self, batch, args):
                return text(tokens).last_hidden_state
        controller = MemoryPhaseController(adapter, rank=0, device='cuda')
        batch = SimpleNamespace(modules={'text_encoder': text}, extra={'memory_phase_controller': controller})
        with torch.no_grad():
            actual = TextStage().forward(batch, ServerArgs(resource_policy='dynamic_offload', device='cuda'))
        controller.close()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        self.assertIs(text.shared.weight, text.encoder.embed_tokens.weight)
        adapter.close()
        self.assertIs(text.encoder.embed_tokens, original)
        self.assertIs(text.shared.weight, original.weight)

    def test_eraserdit_fused_block_path_uses_extent_wrapper(self):
        from config.server_args import set_global_server_args
        from models.dits.eraserdit_transformer import EraserDiTLTXVideoTransformer3DModel
        budget = 80 * 1024**2
        args = ServerArgs(resource_policy='dynamic_offload', device='cuda', max_weight_usage=budget)
        set_global_server_args(args)
        model = EraserDiTLTXVideoTransformer3DModel(
            in_channels=3, out_channels=1, num_attention_heads=16,
            attention_head_dim=64, cross_attention_dim=1024, num_layers=2,
            caption_channels=64,
        ).eval().cuda()
        values = dict(hidden_states=torch.ones(1, 1, 1, 1, 1, device='cuda'),
                      cond_latents=torch.zeros(1, 1, 1, 1, 1, device='cuda'),
                      mask_values=torch.ones(1, 1, 1, 1, 1, device='cuda'),
                      encoder_hidden_states=torch.ones(1, 4, 64, device='cuda'),
                      encoder_attention_mask=torch.ones(1, 4, device='cuda'),
                      timestep=torch.ones(1, device='cuda'),
                      num_frames=1, height=1, width=1, return_dict=False)
        with torch.no_grad():
            reference = model(**values)[0]
        model.cpu()
        adapter = EraserDiTMemoryAdapter()
        self.addCleanup(adapter.close)
        adapter.register(modules={'transformer': model}, device=torch.device('cuda'),
                         dynamic_offload=True, pin_memory=True,
                         max_weight_usage=budget, rank=0)
        self.assertTrue(hasattr(model.transformer_blocks[0], 'flexible_extent'))
        class TransformerStage:
            @offload_component('transformer')
            def forward(self, batch, args):
                return model(**values)[0]
        controller = MemoryPhaseController(adapter, rank=0, device='cuda')
        batch = SimpleNamespace(modules={'transformer': model}, extra={'memory_phase_controller': controller})
        with torch.no_grad():
            actual = TransformerStage().forward(batch, args)
        controller.close()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        self.assert_idle(model, adapter, budget)


if __name__ == '__main__':
    unittest.main()
