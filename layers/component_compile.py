"""Compile auxiliary model forwards without replacing modules or their weights."""
from copy import deepcopy
import time

import torch
from torch.utils._pytree import tree_flatten


class CompiledComponentForward:
    def __init__(self, forward, *, name, mode):
        self.eager = forward
        # Torch 2.6 precision-cast emulation fails on T5 relative-position
        # bucket conversion. Use standard Inductor semantics for T5 only.
        options = dict(torch._inductor.list_mode_options(mode))
        options.update({"triton.cudagraphs": False, "emulate_precision_casts": name != "text_encoder"})
        self.compiled = torch.compile(forward, options=options, fullgraph=True, dynamic=False)
        self.seen = set()
        self.report = dict(component=name, mode=mode, fullgraph=True, cudagraphs=False,
                           emulate_precision_casts=options["emulate_precision_casts"],
                           successful_forwards=0, first_call_history=[])

    def __call__(self, *args, **kwargs):
        if torch.is_grad_enabled():
            raise RuntimeError('component compile is inference-only; use torch.no_grad()')
        leaves, _ = tree_flatten((args, kwargs))
        tensors = [value for value in leaves if isinstance(value, torch.Tensor)]
        signature = tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device)) for t in tensors)
        devices = {t.device for t in tensors if t.is_cuda}
        first = signature not in self.seen
        if first:
            for device in devices:
                torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = self.compiled(*args, **kwargs)
        if first:
            for device in devices:
                torch.cuda.synchronize(device)
            self.report['first_call_history'].append(dict(
                input_shapes=[list(t.shape) for t in tensors], seconds=time.perf_counter() - started))
            self.seen.add(signature)
        self.report['successful_forwards'] += 1
        return output


class ComponentCompileManager:
    def __init__(self, modules, names, *, mode='max-autotune-no-cudagraphs'):
        if mode not in ('default', 'max-autotune-no-cudagraphs'):
            raise ValueError('component compile supports default or max-autotune-no-cudagraphs')
        self.entries = []
        try:
            for name in names:
                if name == 'text_encoder':
                    module = modules['text_encoder']
                    if any(hasattr(child, '_get_fsdp_state') for child in module.modules()):
                        raise ValueError('cannot compile a text encoder with FSDP hooks')
                elif name in ('vae_encoder', 'vae_decoder'):
                    module = getattr(modules['vae'], name.removeprefix('vae_'))
                else:
                    raise ValueError(f'unsupported compile component: {name}')
                if isinstance(module.forward, CompiledComponentForward):
                    raise ValueError(f'{name} is already compiled')
                compiled = CompiledComponentForward(module.forward, name=name, mode=mode)
                self.entries.append((module, compiled))
                module.forward = compiled
        except BaseException:
            self.close()
            raise

    def snapshot(self):
        return {compiled.report['component']: deepcopy(compiled.report) for _, compiled in self.entries}

    def close(self):
        for module, compiled in reversed(self.entries):
            if module.forward is compiled:
                module.forward = compiled.eager
        self.entries.clear()
