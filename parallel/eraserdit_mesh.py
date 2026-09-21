"""Single-process device mesh for exact CFG and Ulysses sequence partitioning.

Collectives use peer tensor copies and bounded thread barriers, not NCCL. Each
rank owns a model replica; weights are neither tensor nor pipeline partitioned.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from contextlib import contextmanager
from threading import Barrier
import time

import torch


def resolve_mesh(args, batch=None):
    config = args.pipeline_config
    sp = getattr(config, "sp_degree", 1)
    cfg = getattr(config, "cfg_degree", 1)
    vae = getattr(config, "vae_degree", 1)
    tiling = getattr(config, "vae_tiling", False)
    if any(type(value) is not int for value in (sp, cfg, vae)):
        raise TypeError("parallel degrees must be integers")
    if sp not in (1, 2, 4) or cfg not in (1, 2) or vae not in (1, 2, 4):
        raise ValueError("SP/VAE degrees must be 1,2,4; CFG degree must be 1,2")
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
    if args.resource_policy != "fullgpu" or args.enable_torch_compile:
        raise ValueError("EraserDiT mesh requires fullgpu and compile disabled")
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
    if batch is not None and (getattr(batch, "transformer_cache_mode", "off") != "off"
                              or getattr(batch, "cache_text_projections", False)):
        raise ValueError("device mesh requires all transformer caches disabled")
    return dict(sp=sp, cfg=cfg, vae=vae, devices=[torch.device("cuda", i) for i in indexes],
                linear_mode=linear_mode)


class PeerExchange:
    """Ordered exchange: producers and all consumers finish before slot reuse."""
    def __init__(self, degree, timeout=120):
        self.degree = degree
        self.barrier = Barrier(degree, timeout=timeout)
        self.slots = [None] * degree
        self.calls = [0] * degree

    def abort(self):
        self.barrier.abort()

    def exchange(self, rank, values, select, dim):
        device = values[0].device
        torch.cuda.current_stream(device).synchronize()
        self.slots[rank] = values
        self.barrier.wait()
        try:
            result = tuple(torch.cat([select(peer[k]).to(device) for peer in self.slots], dim=dim)
                           for k in range(len(values)))
            torch.cuda.current_stream(device).synchronize()
            self.barrier.wait()
            self.calls[rank] += 1
            return result
        except BaseException:
            self.abort()
            raise


class SequenceRank:
    def __init__(self, exchange, rank):
        self.exchange, self.rank = exchange, rank
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
        """
        saved = []
        try:
            if mode == "reference":
                modules = [model.proj_out]
                for block in model.transformer_blocks:
                    modules.extend([block.attn1.to_q, block.attn1.to_k, block.attn1.to_v,
                                    block.attn1.to_out[0], block.attn2.to_q, block.attn2.to_out[0],
                                    block.ff.net[0].proj, block.ff.net[2]])
                for module in modules:
                    saved.append((module, module.__dict__.get("forward")))
                    original = module.forward
                    def forward(value, original=original):
                        padded = torch.nn.functional.pad(value, (0, 0, self.start, self.length - self.end))
                        return original(padded)[:, self.start:self.end].contiguous()
                    module.forward = forward
            yield
        finally:
            for module, previous in saved:
                if previous is None:
                    del module.forward
                else:
                    module.forward = previous

    def attention(self, query, key, value, impl, metadata):
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
    def __init__(self, transformer, plan):
        self.source, self.plan = transformer, plan
        self.models, self.groups, self.ranks, self.static = [], [], [], []
        self.executor = None
        self.setup_seconds = 0.0
        self.steps = 0

    @property
    def active(self):
        return self.plan is not None and self.plan["sp"] * self.plan["cfg"] > 1

    def __enter__(self):
        if not self.active:
            return self
        started = time.perf_counter()
        sp, cfg = self.plan["sp"], self.plan["cfg"]
        try:
            for index, device in enumerate(self.plan["devices"][:sp * cfg]):
                model = self.source if index == 0 else deepcopy(self.source).to(device).eval()
                self.models.append(model)
                torch.cuda.synchronize(device)
            self.groups = [PeerExchange(sp) for _ in range(cfg)]
            self.ranks = [SequenceRank(self.groups[i // sp], i % sp) if sp > 1 else None
                          for i in range(sp * cfg)]
            self.static = [{} for _ in self.models]
            self.executor = ThreadPoolExecutor(max_workers=sp * cfg, thread_name_prefix="eraserdit-mesh")
            self.setup_seconds = time.perf_counter() - started
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _forward(self, index, branch, kwargs):
        device = self.plan["devices"][index]
        model, rank = self.models[index], self.ranks[index]
        try:
            with torch.cuda.device(device), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                if branch not in self.static[index]:
                    self.static[index][branch] = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in kwargs.items() if k not in ("hidden_states", "timestep")
                    }
                for block in model.transformer_blocks:
                    block.attn1.processor.sequence_parallel = rank
                from contextlib import nullcontext
                scope = rank.linear_scope(model, self.plan.get("linear_mode", "reference")) if rank else nullcontext()
                with scope:
                    output = model(**self.static[index][branch],
                                   hidden_states=kwargs["hidden_states"].to(device),
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
        if cfg == 2:
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
        return {"sp_degree": self.plan["sp"], "cfg_degree": self.plan["cfg"],
                "devices": [str(d) for d in self.plan["devices"][:len(self.models)]],
                "steps": self.steps, "replica_setup_seconds": self.setup_seconds,
                "collectives_per_rank": [g.calls[:] for g in self.groups],
                "linear_mode": self.plan.get("linear_mode", "reference"),
                "transport": "peer_copy_ulysses", "transformer_cache_mode": "off"}

    def __exit__(self, *exc):
        for group in self.groups:
            group.abort()
        if self.executor is not None:
            self.executor.shutdown(wait=True)
        self.executor = None
        self.models.clear()
        self.static.clear()
        for group in self.groups:
            group.slots.clear()
        return False
