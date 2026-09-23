"""Inference-only DiT offload, following SGLang's per-layer/dtype CPU storage.

Unlike SGLang's cyclic hook scheduler, lookahead is bounded by the actual
execution range. Initialization never allocates GPU weights. CPU views preserve
parameter shapes; retired GPU buffers stay owned until compute completes.
"""
from collections import defaultdict, deque
from contextlib import contextmanager

import torch


class LayerwiseOffloadManager:
    def __init__(self, model, *, device, max_weight_usage, prefetch_size=1,
                 layers_attr='transformer_blocks', pin_memory=True,
                 resident_names=()):
        self.device = torch.device(device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('layerwise offload requires CUDA')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if type(prefetch_size) is not int or prefetch_size < 0:
            raise ValueError('prefetch_size must be a non-negative integer')
        self.budget = int(max_weight_usage)
        if self.budget <= 0:
            raise ValueError('max_weight_usage must be positive')
        self.model = model
        self.layers = getattr(model, layers_attr)
        self.prefetch_size = prefetch_size
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.cpu = {}
        self.targets = {}
        self.sizes = {}
        self.live = {}
        self.retired = deque()
        self.hooks = []
        self.used = self.peak = self.h2d_bytes = self.h2d_count = 0
        self.budget_waits = 0
        self.active = False
        self.closed = False
        self.range_end = len(self.layers)
        self.managed_ids = set()
        excluded = set(resident_names)
        # Validate aliases and CPU placement before changing any tensor storage.
        owners = {}
        groups = {}
        for i, layer in enumerate(self.layers):
            tensors = dict(layer.named_parameters()) | dict(layer.named_buffers())
            groups[i] = {n: t for n, t in tensors.items()
                         if f'{layers_attr}.{i}.{n}' not in excluded}
        for name, t in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False)):
            if t.device.type != 'cpu':
                raise ValueError('initialize layerwise offload from CPU weights')
            # A tied tensor crossing a layer or the resident remainder cannot
            # have two independent residency owners.
            path_owner = next((i for i in groups if name.startswith(f'{layers_attr}.{i}.')), None)
            actual = path_owner if name not in excluded else None
            key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
            if t.numel() and key in owners and owners[key] != actual:
                raise ValueError(f'cross-layer/shared resident storage: {name}')
            if t.numel():
                owners[key] = actual
        # Nonidentical overlapping views are rejected rather than silently untied.
        for i, tensors in groups.items():
            seen = defaultdict(list)
            unique = {}
            for name, t in tensors.items():
                key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
                if t.numel():
                    start = t.storage_offset() * t.element_size()
                    end = start + (1 + sum((s-1)*d for s, d in zip(t.shape, t.stride()))) * t.element_size()
                    for other, lo, hi in seen[key]:
                        if other is not t and start < hi and lo < end:
                            raise ValueError(f'overlapping tensor views in layer {i}: {name}')
                    seen[key].append((t, start, end))
                if id(t) not in unique:
                    unique[id(t)] = (name, t)
            groups[i] = dict(unique.values())
            self.sizes[i] = sum((t.numel() if t.is_contiguous() else
                                (0 if not t.numel() else 1 + sum((s-1)*d for s, d in zip(t.shape, t.stride()))))
                               * t.element_size() for t in groups[i].values())
            if self.sizes[i] > self.budget:
                raise ValueError(f'block {i} requires {self.sizes[i]} bytes; budget={self.budget}')
        try:
            for i, tensors in groups.items():
                self.cpu[i] = {}
                self.targets[i] = tensors
                by_dtype = defaultdict(list)
                for name, t in tensors.items():
                    by_dtype[t.dtype].append((name, t))
                for dtype, entries in by_dtype.items():
                    contiguous = [(n, t) for n, t in entries if t.is_contiguous()]
                    flat = torch.empty(sum(t.numel() for _, t in contiguous), dtype=dtype,
                                       device='cpu', pin_memory=pin_memory)
                    offset = 0
                    for name, t in contiguous:
                        view = flat[offset:offset+t.numel()].view(t.shape)
                        view.copy_(t.detach())
                        self.cpu[i][name] = view
                        t.data = view
                        offset += t.numel()
                    for name, t in entries:
                        if name in self.cpu[i]:
                            continue
                        view = torch.empty_strided(t.shape, t.stride(), dtype=dtype,
                                                   device='cpu', pin_memory=pin_memory)
                        view.copy_(t.detach())
                        self.cpu[i][name] = view
                        t.data = view
                self.managed_ids.update(id(t) for t in tensors.values())
            for i, layer in enumerate(self.layers):
                self.hooks.append(layer.register_forward_pre_hook(self._pre(i)))
                self.hooks.append(layer.register_forward_hook(self._post(i), always_call=True))
        except BaseException:
            for hook in self.hooks:
                hook.remove()
            raise

    def _reap(self, wait=False):
        while self.retired:
            event, buffers, size = self.retired[0]
            if wait:
                event.synchronize()
            elif not event.query():
                break
            self.retired.popleft()
            self.used -= size
            del buffers
            if wait:
                break

    @torch.compiler.disable
    def prefetch_layer(self, i, *, required=False):
        if i in self.live:
            return True
        self._reap()
        size = self.sizes[i]
        if required and self.used + size > self.budget:
            # Unused lookahead from a previous execution range is expendable.
            for other in list(self.live):
                self.release_layer(other)
        while self.used + size > self.budget:
            if not required:
                return False
            if not self.retired:
                raise RuntimeError('layerwise budget cannot make progress')
            self.budget_waits += 1
            self._reap(wait=True)
        buffers = []
        views = {}
        try:
            with torch.cuda.stream(self.copy_stream):
                # Copy each consolidated storage once, then recover its views.
                copied = {}
                for name, cpu in self.cpu[i].items():
                    storage = cpu.untyped_storage()
                    key = (storage.data_ptr(), cpu.dtype)
                    if key not in copied:
                        host = torch.empty(0, dtype=cpu.dtype, device='cpu').set_(storage, 0, (storage.nbytes() // cpu.element_size(),))
                        gpu = torch.empty_like(host, device=self.device)
                        gpu.copy_(host, non_blocking=True)
                        copied[key] = gpu
                        buffers.append(gpu)
                    views[name] = copied[key].as_strided(cpu.shape, cpu.stride(), cpu.storage_offset())
                ready = torch.cuda.Event()
                ready.record(self.copy_stream)
            for name, view in views.items():
                self.targets[i][name].data = view
        except BaseException:
            self.copy_stream.synchronize()
            for name, cpu in self.cpu[i].items():
                self.targets[i][name].data = cpu
            raise
        self.live[i] = (ready, buffers)
        self.used += size
        self.peak = max(self.peak, self.used)
        self.h2d_bytes += size
        self.h2d_count += 1
        return True

    @torch.compiler.disable
    def release_layer(self, i):
        item = self.live.pop(i, None)
        if item is None:
            return
        ready, buffers = item
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(ready)
        done = torch.cuda.Event()
        done.record(stream)
        for name, cpu in self.cpu[i].items():
            self.targets[i][name].data = cpu
        # Retain allocations (and account for them) until compute completes.
        self.retired.append((done, buffers, self.sizes[i]))

    def _pre(self, i):
        def hook(module, inputs):
            if not self.active or self.closed:
                raise RuntimeError('DiT block called outside active offload phase')
            if torch.is_grad_enabled():
                raise RuntimeError('layerwise offload is inference-only')
            self.prefetch_layer(i, required=True)
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(self.live[i][0])
            # record_stream also protects against allocator reuse after an error.
            for buffer in self.live[i][1]:
                buffer.record_stream(stream)
            for j in range(i+1, min(self.range_end, i+1+self.prefetch_size)):
                if not self.prefetch_layer(j):
                    break
        return hook

    def _post(self, i):
        def hook(module, inputs, output):
            self.release_layer(i)
        return hook

    @contextmanager
    def execution_range(self, start, end):
        previous = self.range_end
        self.range_end = end
        try:
            for i in list(self.live):
                if not start <= i < end:
                    self.release_layer(i)
            yield
        finally:
            self.range_end = previous

    @contextmanager
    def layer_residency(self, index):
        """Scoped local compute (e.g. FFN compile warmup) with ordinary ownership."""
        with self.execution_range(index, index + 1):
            try:
                self._pre(index)(self.layers[index], ())
                yield
            finally:
                self.release_layer(index)

    def begin(self):
        if self.closed or self.active:
            raise RuntimeError('offload manager is closed or already active')
        self.active = True

    def release_all(self):
        for i in list(self.live):
            self.release_layer(i)
        while self.retired:
            self._reap(wait=True)
        self.copy_stream.synchronize()

    def end(self):
        try:
            self.release_all()
        finally:
            self.active = False

    def snapshot(self):
        return dict(backend='sglang_layerwise', weight_budget_scope='dit_blocks_only',
                    max_weight_usage=self.budget, prefetch_size=self.prefetch_size,
                    managed_weight_bytes=sum(self.sizes.values()),
                    largest_block_bytes=max(self.sizes.values(), default=0),
                    resident_bytes=self.used, peak_resident_bytes=self.peak,
                    live_layers=len(self.live), pending_releases=len(self.retired),
                    h2d_bytes=self.h2d_bytes, h2d_count=self.h2d_count,
                    budget_waits=self.budget_waits,
                    pinned_cpu_bytes=sum(self.sizes.values()) if any(
                        t.is_pinned() for tensors in self.cpu.values() for t in tensors.values()) else 0)

    def close(self, *, terminal=False):
        if self.closed:
            return
        self.end()
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        # Drop pinned mirrors on shutdown. A nonterminal close restores ordinary
        # CPU views, allowing the caller to use/save the model without hooks.
        for i, tensors in self.cpu.items():
            restored = {}
            for name, cpu in tensors.items():
                if terminal:
                    value = torch.empty(0, dtype=cpu.dtype, device='cpu')
                else:
                    storage = cpu.untyped_storage()
                    key = (storage.data_ptr(), cpu.dtype)
                    if key not in restored:
                        host = torch.empty(0, dtype=cpu.dtype, device='cpu').set_(
                            storage, 0, (storage.nbytes() // cpu.element_size(),))
                        ordinary = torch.empty(host.shape, dtype=host.dtype, device='cpu')
                        ordinary.copy_(host)
                        restored[key] = ordinary
                    value = restored[key].as_strided(cpu.shape, cpu.stride(), cpu.storage_offset())
                self.targets[i][name].data = value
            tensors.clear()
        self.cpu.clear()
        self.targets.clear()
        self.closed = True
