"""Fail unsupported migrated-memory combinations before loading or execution."""
import torch


def validate_memory_config(args, batch=None):
    from config.dit_parallel import validate_nccl_dit
    validate_nccl_dit(args, batch)
    from config.torch_compile import validate_transformer_compile
    validate_transformer_compile(args, batch)
    from config.torch_compile import validate_component_compile
    validate_component_compile(args)
    memory = args.resolve_resource_policy()
    if memory.enabled and (torch.device(args.device).type != 'cuda' or not torch.cuda.is_available()):
        raise ValueError('SGLang offload requires an available CUDA device')
    if args.use_fsdp_inference:
        raise ValueError('DiT FSDP inference is not supported by this single-GPU integration; T5 CPU offload uses its own FSDP mesh')
    if memory.enabled:
        config = args.pipeline_config
        dit_parallel = max(getattr(config, k, 1) for k in ('sp_degree', 'cfg_degree')) > 1
        if ((dit_parallel and (memory.dit_cpu_offload or memory.dit_layerwise_offload))
                or getattr(config, 'vae_degree', 1) > 1
                or getattr(config, 'cfg_parallel_device', None) is not None):
            raise ValueError('SGLang offload currently requires single-GPU EraserDiT')
        # EraserDiT compilation covers only local FFNs, with per-layer warmup
        # residency and CUDA graphs disabled. Transfer hooks remain eager.
        if args.transformer_quantization != 'none' and (memory.dit_cpu_offload or memory.dit_layerwise_offload):
            raise ValueError('INT8 with DiT weight offload is unsupported; keep DiT resident (T5/VAE CPU offload is allowed)')
    if batch is not None and args.dit_layerwise_offload:
        if getattr(batch, 'transformer_cache_mode', 'off') not in ('off', 'teacache'):
            raise ValueError('cache_dit cannot be combined with SGLang cyclic layerwise offload')
