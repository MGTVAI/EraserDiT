"""Single-process device mesh for exact CFG and Ulysses sequence partitioning.

Collectives use peer tensor copies and bounded thread barriers, not NCCL. Each
rank owns a model replica; weights are neither tensor nor pipeline partitioned.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from contextlib import contextmanager
from threading import Barrier
import time
import sys

import torch
from torch.utils._pytree import tree_map


def resolve_mesh(args, batch=None):
    config = args.pipeline_config
    from config.dit_parallel import validate_nccl_dit
    topology = validate_nccl_dit(args, batch)
    if topology is not None:
        size = max(topology.world_size, config.vae_degree)
        primary = torch.device(args.device)
        if primary.type != 'cuda':
            raise ValueError('NCCL DiT requires CUDA')
        index = primary.index if primary.index is not None else torch.cuda.current_device()
        indexes = tuple(range(size)) if config.parallel_devices is None else config.parallel_devices
        if (len(indexes) != size or len(set(indexes)) != size or indexes[0] != index
                or any(type(i) is not int or not 0 <= i < torch.cuda.device_count() for i in indexes)):
            raise ValueError('parallel_devices must match the topology and start with the primary CUDA device')
        return dict(sp=topology.sp, cfg=topology.cfg, vae=config.vae_degree,
                    topology=topology, backend='nccl', ring_attention_mode=config.ring_attention_mode,
                    devices=[torch.device('cuda', i) for i in indexes], linear_mode=config.sp_linear_mode)
    sp = getattr(config, "sp_degree", 1)
    cfg = getattr(config, "cfg_degree", 1)
    vae = getattr(config, "vae_degree", 1)
    tiling = getattr(config, "vae_tiling", False)
    if any(type(value) is not int for value in (sp, cfg, vae)):
        raise TypeError("parallel degrees must be integers")
    if sp not in (1, 2, 4) or cfg not in (1, 2) or vae not in (1, 2, 4):
        raise ValueError("SP/VAE degrees must be 1,2,4; CFG degree must be 1,2")
    attention_mode = getattr(config, 'sp_attention_mode', 'ulysses')
    if attention_mode not in ('ulysses', 'ring'):
        raise ValueError('sp_attention_mode must be ulysses or ring')
    if attention_mode == 'ring' and (sp != 2 or getattr(args, 'attention_backend', 'sdpa') != 'sdpa'):
        raise ValueError('experimental Ring currently supports SP2 with explicit sdpa only')
    size = max(sp * cfg, vae)
    active = size > 1 or tiling
    if not active:
        return None
    if sp * cfg > 4:
        raise ValueError("at most four devices per EraserDiT mesh")
    linear_mode = getattr(config, "sp_linear_mode", "reference")
    if linear_mode not in ("reference", "sharded"):
        raise ValueError("sp_linear_mode must be reference or sharded")
    if getattr(config, "cfg_parallel_device", None) is not None and sp * cfg > 1:
        raise ValueError("use cfg_degree instead of cfg_parallel_device with a mesh")
    if getattr(args, "use_fsdp_inference", False):
        raise ValueError("EraserDiT peer mesh cannot wrap FSDP models")
    memory = args.resolve_resource_policy()
    if size > 1 and memory.enabled and (memory.dit_cpu_offload or memory.dit_layerwise_offload or vae > 1):
        raise ValueError("DiT/VAE parallel offload currently requires single-GPU EraserDiT; disable DiT offload for CFG/SP mesh")
    tile = getattr(config, "vae_tile_size", 512)
    stride = getattr(config, "vae_tile_stride", 448)
    if tile < 64 or stride < 32 or stride >= tile or tile % 32 or stride % 32:
        raise ValueError("VAE tile and stride must be multiples of 32; 32 <= stride < tile")
    primary = torch.device(args.device)
    if primary.type != "cuda":
        raise ValueError("device mesh requires CUDA")
    primary_index = primary.index if primary.index is not None else torch.cuda.current_device()
    indexes = getattr(config, "parallel_devices", None)
    indexes = list(range(size)) if indexes is None else list(indexes)
    if len(indexes) != size or len(set(indexes)) != size or indexes[0] != primary_index:
        raise ValueError("parallel_devices must be distinct, match mesh size, and start with primary")
    if any(type(i) is not int or i < 0 or i >= torch.cuda.device_count() for i in indexes):
        raise ValueError("parallel_devices contains unavailable local CUDA indices")
    return dict(sp=sp, cfg=cfg, vae=vae, devices=[torch.device("cuda", i) for i in indexes],
                linear_mode=linear_mode, attention_mode=attention_mode)


class PeerExchange:
    """Ordered exchange: producers and all consumers finish before slot reuse."""
    def __init__(self, degree, timeout=120):
        self.degree = degree
        self.barrier = Barrier(degree, timeout=timeout)
        self.slots = [None] * degree
        self.calls = [0] * degree
        self.direct_copies = [0] * degree

    def abort(self):
        self.barrier.abort()

    def exchange(self, rank, values, select, dim):
        device = values[0].device
        try:
            torch.cuda.current_stream(device).synchronize()
            self.slots[rank] = values
            self.barrier.wait()
            result = []
            for k in range(len(values)):
                parts = [select(peer[k]) for peer in self.slots]
                # Large transfers can write directly into the final tensor,
                # avoiding destination temporaries followed by another cat.
                # Small transfers retain cat: strided copy launch overhead
                # outweighed the saved bandwidth on the short-window benchmark.
                size = sum(part.numel() * part.element_size() for part in parts)
                if device.type == 'cuda' and size >= 32 * 1024**2:
                    shape = list(parts[0].shape)
                    shape[dim] = sum(part.shape[dim] for part in parts)
                    output = torch.empty(shape, dtype=parts[0].dtype, device=device)
                    offset = 0
                    for part in parts:
                        output.narrow(dim, offset, part.shape[dim]).copy_(part, non_blocking=True)
                        offset += part.shape[dim]
                    result.append(output)
                    self.direct_copies[rank] += 1
                else:
                    result.append(torch.cat([part.to(device) for part in parts], dim=dim))
            torch.cuda.current_stream(device).synchronize()
            self.barrier.wait()
            self.calls[rank] += 1
            return tuple(result)
        except BaseException:
            self.abort()
            raise


    def rotate(self, rank, values):
        """One ring hop; no full-sequence K/V allocation."""
        device = values[0].device
        try:
            torch.cuda.current_stream(device).synchronize()
            self.slots[rank] = values
            self.barrier.wait()
            result = tuple(x.to(device) for x in self.slots[(rank - 1) % self.degree])
            torch.cuda.current_stream(device).synchronize()
            self.barrier.wait()
            self.calls[rank] += 1
            return result
        except BaseException:
            self.abort()
            raise


class PeerCacheCoordinator:
    """Cache consensus shares the ordered SP transport with attention."""
    def __init__(self, exchange, rank):
        self.exchange, self.rank = exchange, rank
        self.world_size = exchange.degree

    def all_reduce(self, tensor):
        values = self.exchange.exchange(self.rank, (tensor.unsqueeze(0),), lambda x: x, dim=0)[0]
        return values.sum(dim=0)


class SequenceRank:
    def __init__(self, exchange, rank, attention_mode='ulysses'):
        self.exchange, self.rank = exchange, rank
        self.attention_mode = attention_mode
        self.degree = exchange.degree
        self.start = self.end = 0

    def partition(self, length):
        if length < self.degree:
            raise ValueError("sequence length must be at least SP degree")
        self.start = length * self.rank // self.degree
        self.end = length * (self.rank + 1) // self.degree
        self.length = length
        return slice(self.start, self.end)

    @contextmanager
    def linear_scope(self, model, mode):
        """Reference mode preserves GEMM M and row offsets to avoid BF16 drift.

        Only local rows survive each projection. This deliberately computes
        dummy rows and sacrifices linear-layer speed/memory for compatibility;
        the attention heads and resident hidden states remain partitioned.
        Aligned mode skips padding only for a screened numerical profile, with
        separate protection for the final output and short-SP4 FFN down GEMM.
        """
        saved = []
        pad_sequence = None
        aligned_lengths, fallback = (), None
        if mode == 'aligned':
            from layers.sequence_linear import aligned_sequence_lengths
            aligned_lengths, fallback = aligned_sequence_lengths(model, self.degree)
        self.linear_report = dict(requested=mode, effective='reference' if mode == 'aligned' else mode,
                                  fallback_reason=fallback, local_calls=0, reference_calls=0,
                                  protected_ffn_down_calls=0, compact_ffn_down_calls=0)
        try:
            if mode in ("reference", "aligned"):
                from layers.sequence_padding import ReferenceSequencePadding
                pad_sequence = ReferenceSequencePadding()
                modules = [(model.proj_out, 'output')]
                for block in model.transformer_blocks:
                    modules.extend((module, 'linear') for module in (
                        block.attn1.to_q, block.attn1.to_k, block.attn1.to_v,
                        block.attn1.to_out[0], block.attn2.to_q, block.attn2.to_out[0]))
                    modules.append((block.ff, 'ffn'))
                    if mode == 'aligned':
                        modules.append((block.ff.net[2], 'ffn_down'))
                for module, role in modules:
                    saved.append((module, module.__dict__.get("forward")))
                    original = module.forward
                    def forward(value, original=original, role=role):
                        aligned = (self.length in aligned_lengths and value.ndim == 3
                                   and value.shape[0] == 1 and value.dtype == torch.bfloat16
                                   and value.shape[1] == self.end - self.start)
                        if mode == 'aligned':
                            self.linear_report['tokens'] = self.length
                            if not aligned and not self.linear_report['local_calls']:
                                self.linear_report['fallback_reason'] = (
                                    fallback or 'unvalidated sequence shape or activation dtype')
                        # Unknown profiles keep the entire FFN reference path;
                        # its nested down projection must not pad a second time.
                        if role == 'ffn_down':
                            if not (aligned and self.degree == 4 and self.length == 10200):
                                return original(value)
                            self.linear_report['protected_ffn_down_calls'] += 1
                            if getattr(self, 'compact_ffn_down', True):
                                # L40S's 2550-row BF16 down GEMM selects a different
                                # reduction. 3072 rows reproduce the screened full
                                # 10200-row result without calculating every shard.
                                self.linear_report['compact_ffn_down_calls'] += 1
                                self.linear_report['reference_calls'] += 1
                                padded = pad_sequence(value, 0, 3072)
                                return original(padded)[:, :value.shape[1]].contiguous()
                        elif aligned and role != 'output':
                            self.linear_report['local_calls'] += 1
                            self.linear_report.update(effective='aligned', fallback_reason=None)
                            return original(value)
                        self.linear_report['reference_calls'] += 1
                        padded = pad_sequence(value, self.start, self.length)
                        return original(padded)[:, self.start:self.end].contiguous()
                    module.forward = forward
            yield
        finally:
            for module, previous in saved:
                if previous is None:
                    del module.forward
                else:
                    module.forward = previous
            if pad_sequence is not None:
                pad_sequence.clear()

    def attention(self, query, key, value, impl, metadata):
        if self.attention_mode == 'ring':
            from models.adapters.eraserdit.ring import ring_attention
            return ring_attention(self, query, key, value, impl, metadata)
        heads = query.shape[2]
        if heads % self.degree:
            raise ValueError("attention heads must be divisible by SP degree")
        head_slice = slice(heads * self.rank // self.degree, heads * (self.rank + 1) // self.degree)
        from layers.attention.backends.sage_attn import SageAttentionImpl
        if isinstance(impl, SageAttentionImpl):
            # Sage 1.x subtracts K.mean(sequence) in BF16. Changing the number
            # of heads changes PyTorch's reduction layout. Preserve the full
            # layout for this reduction before selecting the local heads.
            full_key = self.exchange.exchange(self.rank, (key,), lambda x: x, dim=1)[0]
            full_key.sub_(full_key.mean(dim=1, keepdim=True))
            q, v = self.exchange.exchange(self.rank, (query, value),
                                          lambda x: x[:, :, head_slice], dim=1)
            k = full_key[:, :, head_slice].contiguous()
            output = impl.forward(q, k, v, metadata, key_already_smoothed=True)
        else:
            q, k, v = self.exchange.exchange(self.rank, (query, key, value),
                                            lambda x: x[:, :, head_slice], dim=1)
            output = impl.forward(q, k, v, metadata)
        return self.exchange.exchange(self.rank, (output,),
                                      lambda x: x[:, self.start:self.end], dim=2)[0]


class EraserDiTMeshWindow:
    def __init__(self, transformer, plan, *, pool=None, batch=None, total_steps=0):
        self.source, self.plan = transformer, plan
        self.pool, self.batch, self.total_steps = pool, batch, total_steps
        self.caches = []
        self.cache_batches = []
        self.models, self.groups, self.ranks, self.static = [], [], [], []
        self.rotary = []
        self.executor = None
        self.setup_seconds = 0.0
        self.steps = 0
        self.compile_preparation_seconds = 0.0

    @property
    def active(self):
        return self.plan is not None and self.plan["sp"] * self.plan["cfg"] > 1

    def __enter__(self):
        if not self.active:
            return self
        started = time.perf_counter()
        sp, cfg = self.plan["sp"], self.plan["cfg"]
        try:
            if self.pool is not None:
                self.pool.acquire()
                self.models = list(self.pool.models)
            else:
                from models.adapters.eraserdit.replicas import clone_transformer_cpu
                if getattr(self.source, '_layerwise_offload_manager', None) is not None:
                    raise ValueError('offloaded mesh requires a persistent replica pool')
                for index, device in enumerate(self.plan['devices'][:sp * cfg]):
                    model = self.source if index == 0 else clone_transformer_cpu(self.source).to(device)
                    self.models.append(model)
                    torch.cuda.synchronize(device)
            self.groups = [PeerExchange(sp) for _ in range(cfg)]
            self.ranks = [SequenceRank(self.groups[i // sp], i % sp, self.plan.get("attention_mode", "ulysses")) if sp > 1 else None
                          for i in range(sp * cfg)]
            if self.batch is not None:
                from cache.eraserdit import EraserDiTCacheWindow
                for index in range(sp * cfg):
                    batch = copy(self.batch)
                    batch.extra = dict(self.batch.extra)
                    coordinator = PeerCacheCoordinator(self.groups[index // sp], index % sp) if sp > 1 else None
                    cache = EraserDiTCacheWindow(batch, total_steps=self.total_steps,
                        num_blocks=len(self.source.transformer_blocks), sp_degree=sp, sp_rank=index % sp,
                        cfg_degree=cfg, cfg_rank=index // sp, coordinator=coordinator)
                    self.cache_batches.append(batch)
                    self.caches.append(cache)
                    cache.__enter__()
            self.static = [{} for _ in self.models]
            self.rotary = [None for _ in self.models]
            self.executor = ThreadPoolExecutor(max_workers=sp * cfg, thread_name_prefix="eraserdit-mesh")
            self.setup_seconds = time.perf_counter() - started
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def _forward(self, index, branch, kwargs):
        device = self.plan["devices"][index]
        model, rank = self.models[index], self.ranks[index]
        try:
            with torch.cuda.device(device), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hidden_states = kwargs['hidden_states'].to(device)
                if branch not in self.static[index]:
                    self.static[index][branch] = tree_map(
                        lambda value: value.to(device) if isinstance(value, torch.Tensor) else value,
                        {k: v for k, v in kwargs.items()
                         if k not in ("hidden_states", "timestep", "cache_adapter", "text_cache")})
                    if 'image_rotary_emb' not in self.static[index][branch]:
                        if self.rotary[index] is None:
                            self.rotary[index] = model.rope(
                                hidden_states, kwargs['num_frames'], kwargs['height'], kwargs['width'],
                                kwargs.get('rope_interpolation_scale'), kwargs.get('video_coords'))
                        self.static[index][branch]['image_rotary_emb'] = self.rotary[index]
                for block in model.transformer_blocks:
                    block.attn1.processor.sequence_parallel = rank
                from contextlib import nullcontext
                scope = rank.linear_scope(model, self.plan.get("linear_mode", "reference")) if rank else nullcontext()
                cache_kwargs = self.caches[index].kwargs(branch, self.steps) if self.caches else {}
                # The pool owns one persistent compiled callable per device.
                # Transfers, CFG scheduling and all mesh state stay eager.
                compiled = getattr(self.pool, 'compiled_forwards', ())
                forward = compiled[index] if compiled else model
                with scope:
                    output = forward(**self.static[index][branch], **cache_kwargs,
                                   hidden_states=hidden_states,
                                   timestep=kwargs["timestep"].to(device), sequence_parallel=rank)[0]
                torch.cuda.current_stream(device).synchronize()
                return output
        except BaseException:
            for group in self.groups:
                group.abort()
            raise
        finally:
            for block in model.transformer_blocks:
                block.attn1.processor.sequence_parallel = None

    def predict(self, negative, positive):
        torch.cuda.current_stream(positive["hidden_states"].device).synchronize()
        sp, cfg = self.plan["sp"], self.plan["cfg"]
        if self.steps == 0:
            from layers.block_compile import prepare_block_compile
            config = self.source.config
            tokens = (positive['num_frames'] // config.patch_size_t
                      * (positive['height'] // config.patch_size) * (positive['width'] // config.patch_size))
            for index, model in enumerate(self.models):
                length = tokens
                if sp > 1 and self.plan.get('linear_mode', 'reference') == 'sharded':
                    rank = index % sp
                    length = tokens * (rank + 1) // sp - tokens * rank // sp
                self.compile_preparation_seconds += prepare_block_compile(model,
                    batch_size=positive['hidden_states'].shape[0], sequence_length=length,
                    device=self.plan['devices'][index], dtype=positive['hidden_states'].dtype)
        if cfg == 2:
            if sp == 1 and self.steps == 0:
                # Initialize lazy kernels/device handles serially for each new
                # window shape. These are real first-step outputs, not dummy
                # cache updates. Later CFG steps execute concurrently.
                outputs = [self._forward(0, 'positive', positive),
                           self._forward(1, 'negative', negative)]
            else:
                futures = [self.executor.submit(self._forward, i, "positive" if i < sp else "negative",
                                               positive if i < sp else negative) for i in range(2 * sp)]
                outputs = [f.result() for f in futures]
            pos, neg = outputs[:sp], outputs[sp:]
        else:
            neg = [f.result() for f in [self.executor.submit(self._forward, i, "negative", negative)
                                        for i in range(sp)]]
            pos = [f.result() for f in [self.executor.submit(self._forward, i, "positive", positive)
                                        for i in range(sp)]]
        device = positive["hidden_states"].device
        def gather(parts):
            if sp == 1:
                return parts[0].to(device).float()
            from models.dits.eraserdit_transformer import unpack_latents
            packed = torch.cat([p.to(device) for p in parts], dim=1)
            return unpack_latents(packed, positive["num_frames"], positive["height"], positive["width"],
                                  self.source.config.patch_size, self.source.config.patch_size_t).float()
        self.steps += 1
        return gather(neg), gather(pos)

    def report(self):
        from memory.telemetry import memory_observation
        return {"sp_degree": self.plan["sp"], "cfg_degree": self.plan["cfg"],
                "devices": [str(d) for d in self.plan["devices"][:len(self.models)]],
                "steps": self.steps, "replica_setup_seconds": self.setup_seconds,
                "compile_preparation_seconds": self.compile_preparation_seconds,
                "collectives_per_rank": [g.calls[:] for g in self.groups],
                "direct_copy_tensors_per_rank": [g.direct_copies[:] for g in self.groups],
                "linear_mode": self.plan.get("linear_mode", "reference"),
                "transport": "peer_copy_" + self.plan.get("attention_mode", "ulysses"),
                "persistent_replicas": self.pool is not None,
                "rotary_cache_entries": sum(value is not None for value in self.rotary),
                "serial_first_cfg_step": self.plan['sp'] == 1 and self.plan['cfg'] == 2,
                "rank_attention": [m.transformer_blocks[0].attn1.processor.attention_backend_report()
                                   for m in self.models],
                "rank_memory": [memory_observation(d) for d in self.plan["devices"][:len(self.models)]],
                "replica_offload": self.pool.snapshot() if self.pool else [],
                "transformer_cache_mode": self.caches[0].mode if self.caches else "off"}

    def __exit__(self, *exc):
        for group in self.groups:
            group.abort()
        if self.executor is not None:
            self.executor.shutdown(wait=True)
        self.executor = None
        try:
            for cache in self.caches:
                cache.__exit__(*exc)
            if self.batch is not None and self.cache_batches:
                reports = [batch.extra['transformer_cache'] for batch in self.cache_batches]
                from cache.eraserdit import aggregate_rank_cache_reports
                self.batch.extra['transformer_cache'] = aggregate_rank_cache_reports(reports, sp_degree=self.plan['sp'])
        finally:
            self.caches.clear()
            self.cache_batches.clear()
            try:
                if self.pool is not None:
                    self.pool.release()
            finally:
                self.models.clear()
                self.static.clear()
                self.rotary.clear()
                for group in self.groups:
                    group.slots.clear()
        return False
