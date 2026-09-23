"""Stage residency for T5/VAE and SGLang-style DiT block offload."""
import time
from collections import deque

import torch

from memory.backends.layerwise_offload import LayerwiseOffloadManager
from memory.telemetry import memory_observation


def _tensors(module):
    return list(module.parameters()) + list(module.buffers())


def _bytes(tensors):
    return sum(t.numel() * t.element_size() for t in {id(t): t for t in tensors}.values())


class LayerwiseMemoryAdapter:
    def __init__(self, *, prefetch_size=1):
        self.prefetch_size = prefetch_size
        self.active_component_name = None
        self.manager = None
        self.closed = False
        self.events = deque(maxlen=64)
        self.transfer_counts = {}
        self.components = {}
        self._closed_snapshot = None

    def register(self, *, modules, device, max_weight_usage, **kwargs):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.modules = modules
        transformer = modules['transformer']
        # TeaCache's small first-block probe stays with the non-block remainder.
        resident = {'transformer_blocks.0.scale_shift_table'}
        resident.update(f'transformer_blocks.0.norm1.{n}' for n, _ in
                        list(transformer.transformer_blocks[0].norm1.named_parameters()) +
                        list(transformer.transformer_blocks[0].norm1.named_buffers()))
        self.manager = LayerwiseOffloadManager(
            transformer, device=self.device, max_weight_usage=max_weight_usage,
            prefetch_size=self.prefetch_size, resident_names=resident,
        )
        transformer._layerwise_offload_manager = self.manager
        self.components['transformer'] = [t for t in _tensors(transformer)
                                          if id(t) not in self.manager.managed_ids]
        if 'text_encoder' in modules:
            self.components['text_encoder'] = _tensors(modules['text_encoder'])
        vae = modules.get('vae')
        if vae is not None:
            encoder = _tensors(vae.encoder)
            decoder = _tensors(vae.decoder)
            owned = {id(t) for t in encoder + decoder}
            # Root normalization and any other shared VAE state has the same stage
            # lifetime; identity deduplication preserves tied Parameters.
            shared = [t for t in _tensors(vae) if id(t) not in owned]
            self.components['vae.encoder'] = encoder + shared
            self.components['vae.decoder'] = decoder + shared
            vae._mgerase_execution_device = self.device
        self.components = {name: list({id(t): t for t in ts}.values())
                           for name, ts in self.components.items()}
        self.host = {id(t): t.detach() for ts in self.components.values() for t in ts}
        return self.snapshot()

    def _move(self, name, device):
        for t in self.components[name]:
            t.data = self.host[id(t)].to(device=device)

    def _trim_cache(self):
        # Preprocessing/VAE workspaces dwarf weights. Do not let their idle
        # allocator segments fragment the next component's allocations.
        # Never called between DiT blocks: that would serialize the pipeline.
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()

    def acquire_component_residency(self, name, *, reason):
        if self.closed or self.active_component_name is not None:
            raise RuntimeError('component adapter is closed or already active')
        start = time.perf_counter()
        try:
            self._trim_cache()
            self._move(name, self.device)
            if name == 'transformer':
                self.manager.begin()
        except BaseException:
            self._move(name, 'cpu')
            raise
        self.active_component_name = name
        self.transfer_counts[name] = self.transfer_counts.get(name, 0) + 1
        self.events.append(dict(component=name, action='acquire', reason=reason,
                                seconds=time.perf_counter()-start,
                                weight_bytes=_bytes(self.components[name]),
                                memory=memory_observation(self.device)))

    def release_component_residency(self, name, *, reason):
        if self.active_component_name != name:
            raise RuntimeError('component release does not match active phase')
        start = time.perf_counter()
        try:
            if name == 'transformer':
                self.manager.end()
            # Stage boundary only: finish kernels before dropping component
            # weights, including error exits with work still queued.
            torch.cuda.current_stream(self.device).synchronize()
        finally:
            self._move(name, 'cpu')
            self.active_component_name = None
            self._trim_cache()
        self.events.append(dict(component=name, action='release', reason=reason,
                                seconds=time.perf_counter()-start,
                                memory=memory_observation(self.device)))

    def settle_component_transfers(self):
        if self.manager and not self.manager.active:
            self.manager.release_all()

    def onload_remainder(self, name):
        raise RuntimeError('layerwise components must acquire stage residency')

    def offload_remainder(self, name):
        # Also runs if entering a phase failed part way through a transfer.
        if self.active_component_name == name:
            self.release_component_residency(name, reason='exception_cleanup')
        elif name in self.components:
            self._move(name, 'cpu')

    def snapshot(self):
        if self._closed_snapshot is not None:
            return dict(self._closed_snapshot)
        result = self.manager.snapshot() if self.manager else {}
        result.update(active_component_name=self.active_component_name,
                      dynamic_offload=True,
                      component_weight_bytes={n: _bytes(ts) for n, ts in self.components.items()},
                      component_acquire_counts=dict(self.transfer_counts),
                      component_transfers=list(self.events),
                      component_transfer_history_limit=64,
                      component_resident_bytes=(
                          _bytes(self.components[self.active_component_name])
                          if self.active_component_name else 0),
                      cpu_component_weight_bytes=_bytes(
                          [t for ts in self.components.values() for t in ts]))
        return result

    def shutdown(self, *, terminal=False):
        if self.closed:
            return self.snapshot()
        if self.active_component_name:
            self.release_component_residency(self.active_component_name, reason='shutdown')
        if self.manager:
            self.manager.close(terminal=terminal)
            del self.modules['transformer']._layerwise_offload_manager
        self._closed_snapshot = self.snapshot()
        self._closed_snapshot['shutdown_mode'] = 'terminal' if terminal else 'restored'
        if hasattr(self.modules.get('vae'), '_mgerase_execution_device'):
            del self.modules['vae']._mgerase_execution_device
        if terminal:
            for tensors in self.components.values():
                for t in tensors:
                    t.data = torch.empty(0, dtype=t.dtype, device='cpu')
        self.host.clear()
        self.components.clear()
        self.closed = True
        return self.snapshot()
