"""Persistent CPU-created replicas; each rank owns its offload state."""
from copy import deepcopy

import torch


def clone_transformer_cpu(source):
    # Populate deepcopy's memo with CPU tensors first: never transiently clone
    # a complete model on the source GPU. Preserve tied Parameter identities.
    memo = {}
    for tensor in list(source.parameters()) + list(source.buffers()):
        if id(tensor) not in memo:
            value = tensor.detach().to(device='cpu', copy=True)
            memo[id(tensor)] = (torch.nn.Parameter(value, requires_grad=tensor.requires_grad)
                                if isinstance(tensor, torch.nn.Parameter) else value)
    manager = getattr(source, '_layerwise_offload_manager', None)
    if manager is not None:
        memo[id(manager)] = None
        hook_ids = {h.id for h in manager.hooks}
        for block in source.transformer_blocks:
            for name in ('_forward_pre_hooks', '_forward_hooks', '_forward_hooks_always_called'):
                hooks = getattr(block, name)
                memo[id(hooks)] = type(hooks)((k, v) for k, v in hooks.items() if k not in hook_ids)
    for block in source.transformer_blocks:
        if hasattr(block.ff, '_eager_forward'):
            memo[id(block.ff.forward)] = None
    replica = deepcopy(source, memo).eval()
    replica.__dict__.pop('_layerwise_offload_manager', None)
    from layers.block_compile import remove_block_compile
    remove_block_compile(replica)
    return replica


class EraserDiTReplicaPool:
    def __init__(self, source, plan, args):
        self.models = [source]
        self.adapters = []
        self.plan = plan
        self.policy = args.resolve_resource_policy()
        self.closed = False
        self.active = []
        try:
            for device in plan['devices'][1:plan['sp'] * plan['cfg']]:
                model = clone_transformer_cpu(source)
                self.models.append(model)
                if self.policy.dynamic_offload:
                    from memory.adapters.layerwise_memory_adapter import LayerwiseMemoryAdapter
                    adapter = LayerwiseMemoryAdapter(prefetch_size=args.dit_offload_prefetch_size)
                    adapter.register(modules={'transformer': model}, device=device,
                                     max_weight_usage=self.policy.max_weight_usage)
                else:
                    adapter = None
                    if not self.policy.dit_cpu_offload:
                        model.to(device)
                self.adapters.append(adapter)
                if args.enable_torch_compile:
                    from layers.block_compile import configure_block_compile
                    from nodes.stages.denoising import resolve_torch_compile_mode
                    configure_block_compile(model, mode=resolve_torch_compile_mode())
        except BaseException:
            self.close()
            raise

    def acquire(self):
        if self.closed or self.active:
            raise RuntimeError('replica pool is closed or already acquired')
        try:
            for index, adapter in enumerate(self.adapters, 1):
                self.active.append(index)
                if adapter:
                    adapter.acquire_component_residency('transformer', reason='mesh_window')
                elif self.policy.dit_cpu_offload:
                    self.models[index].to(self.plan['devices'][index])
        except BaseException:
            self.release()
            raise

    def release(self):
        try:
            for index in reversed(self.active):
                adapter = self.adapters[index - 1]
                if adapter:
                    adapter.offload_remainder('transformer')
                elif self.policy.dit_cpu_offload:
                    torch.cuda.synchronize(self.plan['devices'][index])
                    self.models[index].to('cpu')
        finally:
            self.active.clear()

    def snapshot(self):
        return [adapter.snapshot() if adapter else {'dynamic_offload': False}
                for adapter in self.adapters]

    def close(self):
        self.release()
        for adapter in self.adapters:
            if adapter:
                adapter.shutdown()
        self.adapters.clear()
        self.models.clear()
        self.closed = True
