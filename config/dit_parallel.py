"""Validated DiT worker topology. Ulysses is the low-order SP axis."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DiTTopology:
    tp: int = 1
    ulysses: int = 1
    ring: int = 1
    cfg: int = 1
    replicas: int = 1

    def __post_init__(self):
        if any(type(n) is not int or n < 1 for n in self.sizes):
            raise ValueError('parallel degrees must be positive integers')
        if self.cfg not in (1, 2):
            raise ValueError('CFG degree must be 1 or 2')

    @property
    def sizes(self):
        return (self.tp, self.ulysses, self.ring, self.cfg, self.replicas)

    @property
    def world_size(self):
        import math
        return math.prod(self.sizes)

    @property
    def sp(self):
        return self.ulysses * self.ring

    def coordinates(self, rank):
        if not 0 <= rank < self.world_size:
            raise ValueError('rank outside topology')
        values = []
        for size in self.sizes:
            values.append(rank % size)
            rank //= size
        return tuple(values)

    def rank(self, coordinates):
        value, stride = 0, 1
        for index, size in zip(coordinates, self.sizes):
            if not 0 <= index < size:
                raise ValueError('coordinate outside topology')
            value += index * stride
            stride *= size
        return value

    def groups(self, axis):
        axes = {'tp': (0,), 'ulysses': (1,), 'ring': (2,), 'sp': (1, 2), 'cfg': (3,)}[axis]
        groups = {}
        for rank in range(self.world_size):
            coords = self.coordinates(rank)
            key = tuple(c for i, c in enumerate(coords) if i not in axes)
            groups.setdefault(key, []).append(rank)
        return tuple(tuple(ranks) for ranks in groups.values())


def resolve_dit_topology(config):
    sp = config.sp_degree
    if type(sp) is not int or sp < 1:
        raise ValueError('sp_degree must be a positive integer')
    ring = getattr(config, 'ring_degree', 1)
    ulysses = getattr(config, 'ulysses_degree', None)
    if ulysses is None:
        if config.sp_attention_mode == 'ring' and ring == 1:
            ring = sp
        if type(ring) is not int or ring < 1 or sp % ring:
            raise ValueError('sp_degree must be divisible by ring_degree')
        ulysses = sp // ring
    if sp != ulysses * ring:
        raise ValueError('sp_degree must equal ulysses_degree * ring_degree')
    tp = getattr(config, 'tp_degree', 1)
    base = DiTTopology(tp=tp, ulysses=ulysses, ring=ring, cfg=config.cfg_degree)
    shard = getattr(config, 'dit_fsdp_shard_degree', 1)
    replicate = getattr(config, 'dit_fsdp_replicate_degree', 1)
    if any(type(v) is not int or v < 1 for v in (shard, replicate)):
        raise ValueError('FSDP degrees must be positive integers')
    fsdp = shard * replicate
    if fsdp > 1:
        if tp != 1:
            raise ValueError('FSDP and TP cannot manage the same weights')
        if fsdp % base.world_size:
            raise ValueError('FSDP mesh size must be a multiple of CFG * SP')
    return DiTTopology(tp=tp, ulysses=ulysses, ring=ring, cfg=config.cfg_degree,
                       replicas=max(1, fsdp // base.world_size))


def validate_nccl_dit(args, batch=None):
    config = getattr(args, 'pipeline_config', None)
    if getattr(config, 'dit_parallel_backend', 'peer') not in ('peer', 'nccl'):
        raise ValueError('dit_parallel_backend must be peer or nccl')
    if getattr(config, 'dit_parallel_backend', 'peer') != 'nccl':
        if any(getattr(config, name, 1) != 1 for name in
               ('tp_degree', 'ring_degree', 'dit_fsdp_shard_degree', 'dit_fsdp_replicate_degree')) or getattr(config, 'ulysses_degree', None) is not None:
            raise ValueError('explicit TP/USP/FSDP degrees require --dit-parallel-backend nccl')
        return None
    topology = resolve_dit_topology(config)
    import torch
    if args.resolve_component_dtype('transformer') is not torch.bfloat16:
        raise ValueError('NCCL DiT currently requires bf16 model precision')
    if config.sp_linear_mode not in ('reference', 'sharded'):
        raise ValueError('sp_linear_mode must be reference or sharded')
    if config.tp_linear_mode not in ('reference', 'sharded', 'aligned'):
        raise ValueError('tp_linear_mode must be reference, sharded or aligned')
    if type(config.vae_degree) is not int or config.vae_degree not in (1, 2, 4):
        raise ValueError('vae_degree must be 1, 2 or 4')
    tile, stride = config.vae_tile_size, config.vae_tile_stride
    if tile < 64 or stride < 32 or stride >= tile or tile % 32 or stride % 32:
        raise ValueError('VAE tile and stride must be multiples of 32; 32 <= stride < tile')
    if config.ring_attention_mode not in ('reference', 'online', 'streaming'):
        raise ValueError('ring_attention_mode must be reference, online or streaming')
    if args.dit_cpu_offload or args.dit_layerwise_offload:
        raise ValueError('NCCL DiT workers currently require resident or FSDP-sharded weights')
    from config.torch_compile import normalize_compile_components
    if (args.enable_torch_compile or normalize_compile_components(getattr(args, 'compile_components', ()))
            or str(args.transformer_quantization).strip().lower() != 'none'):
        raise ValueError('NCCL quality-alignment path requires compile off and unquantized DiT')
    if args.attention_backend != 'sdpa' or args.operator_fusion_backend != 'disabled':
        raise ValueError('NCCL quality-alignment path requires SDPA and operator fusion disabled')
    if config.cfg_parallel_device is not None:
        raise ValueError('use cfg_degree with NCCL DiT workers')
    if batch is not None:
        get = batch.get if isinstance(batch, dict) else lambda key, default: getattr(batch, key, default)
        if get('transformer_cache_mode', 'off') != 'off' or get('cache_text_projections', False):
            raise ValueError('NCCL quality-alignment path requires transformer caches off')
    return topology
