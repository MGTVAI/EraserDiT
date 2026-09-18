"""LTX0.9.5 CFG-parallel denoising adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from config.transformer_cache import TransformerCacheMode
from layers.attention import AttentionBackendEnum, SequenceParallelMetadata
from models.dits.ltx095_parallel import LTX095SequenceParallelContract
from nodes.schedule_batch import Req
from distributed.group_coordinator import GroupCoordinator
from distributed.parallel_state import ParallelContext, RuntimeGroup
from cache import (
    CacheBranch,
    build_ltx095_transformer_cache_controller,
    ltx095_transformer_cache_window_scope,
)
from parallel.guidance_parallel import (
    GuidanceBranch,
    GuidanceParallelBinding,
    GuidanceParallelEngine,
    resolve_guidance_parallel_binding,
)
from parallel.sequence_sharding import (
    SequenceShardPlan,
    gather_sequence_to_owner,
    plan_sequence_shards,
    shard_sequence_tensor,
)
from service.control import service_checkpoint
from utils.dynamic_cfg import calc_current_cfg
from utils.mask import concrete_mask
from utils.resource_policy import module_device, module_dtype
from utils.runtime_progress import RuntimeProgressState

_GLOBAL_NOISE_KEY = "ltx095_sequence_parallel_global_noise"


@dataclass
class LTX095CFGParallelState:
    """Rank-local tensors used by one LTX CFG branch."""

    binding: GuidanceParallelBinding
    engine: GuidanceParallelEngine
    sequence_plan: SequenceShardPlan
    attention_metadata: SequenceParallelMetadata | None
    sp_coordinator: GroupCoordinator | None
    latents: torch.Tensor
    conditioning: torch.Tensor
    mask: torch.Tensor
    noise: torch.Tensor
    rope_interpolation_scale: tuple[torch.Tensor, torch.Tensor]
    guss_cond: torch.Tensor | None
    prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor
    latent_num_frames: int
    latent_height: int
    latent_width: int


def _positive_int(value, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive non-bool int")
    return value


def _latent_shape(batch: Req) -> tuple[int, int, int, int, int]:
    shape = batch.latent_shape
    if type(shape) is not tuple or len(shape) != 5:
        raise TypeError("batch.latent_shape must be an explicit rank-5 tuple")
    if any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("batch.latent_shape dimensions must be positive ints")
    return shape


def _mask_shape(
    batch: Req,
    latent_shape: tuple[int, int, int, int, int],
) -> tuple[int, int, int, int, int]:
    shape = batch.extra.get("ltx095_sp_mask_shape")
    if type(shape) is not tuple or len(shape) != 5:
        raise TypeError(
            "batch.extra['ltx095_sp_mask_shape'] must be an explicit rank-5 tuple"
        )
    if any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("synchronized mask shape dimensions must be positive ints")
    if shape[0] != latent_shape[0] or shape[2:] != latent_shape[2:]:
        raise ValueError("synchronized mask shape must match latent batch and grid")
    return shape


def _required_tensor(
    name: str,
    tensor,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return tensor.to(device=device, dtype=dtype)


def _packed_template(
    *,
    batch_size: int,
    sequence_length: int,
    feature_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(
        (batch_size, sequence_length, feature_size),
        device=device,
        dtype=dtype,
    )


def _replicate_positive(
    state_engine: GuidanceParallelEngine,
    branch: GuidanceBranch,
    positive_tensor: torch.Tensor | None,
    *,
    template: torch.Tensor,
) -> torch.Tensor:
    return state_engine.replicate_from_positive(
        positive_tensor if branch is GuidanceBranch.POSITIVE else None,
        receive_template=template,
    )


def _sequence_parallel_backend(name: str) -> AttentionBackendEnum:
    mapping = {
        "sdpa": AttentionBackendEnum.TORCH_SDPA,
        "flash_attn": AttentionBackendEnum.FLASH_ATTN,
        "sage_attn": AttentionBackendEnum.SAGE_ATTN,
        "sage_fp8": AttentionBackendEnum.SAGE_FP8,
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise ValueError(
            "active LTX095 CFG parallel attention backend must be 'sdpa', "
            f"'flash_attn', 'sage_attn', or 'sage_fp8', got {name!r}"
        ) from error


def _bind_sequence_parallel_control(
    transformer,
    group: RuntimeGroup,
) -> GroupCoordinator:
    current_device = module_device(transformer)
    binding = getattr(transformer, "sequence_parallel_control_binding", None)
    if binding is None:
        coordinator = GroupCoordinator(group)
        bind = getattr(transformer, "bind_sequence_parallel_control", None)
        if not callable(bind):
            raise TypeError(
                "LTX095 transformer must provide bind_sequence_parallel_control"
            )
        bind(coordinator)
        binding = getattr(transformer, "sequence_parallel_control_binding", None)
        if binding is None:
            raise RuntimeError("transformer did not retain trusted SP control binding")
    else:
        coordinator = getattr(binding, "coordinator", None)

    if coordinator is None:
        raise RuntimeError("trusted SP control binding has no coordinator")
    if tuple(getattr(binding, "group_ranks", ())) != group.spec.ranks:
        raise RuntimeError("trusted SP control binding group ranks changed")
    if getattr(binding, "group_slot", None) != group.group_rank:
        raise RuntimeError("trusted SP control binding group slot changed")
    if torch.device(getattr(binding, "control_device", "cpu")) != current_device:
        raise RuntimeError(
            "trusted SP control device must match the transformer's final "
            f"onload device: binding={getattr(binding, 'control_device', None)}, "
            f"transformer={current_device}"
        )
    if getattr(coordinator, "rank", None) != group.group_rank:
        raise RuntimeError("SP data coordinator rank must match group slot")
    if getattr(coordinator, "world_size", None) != group.world_size:
        raise RuntimeError("SP data coordinator world_size must match SP group")
    return coordinator


def _validate_sp_mesh(
    binding: GuidanceParallelBinding,
    contract: LTX095SequenceParallelContract,
) -> None:
    group = binding.sp_group
    if group.world_size != contract.sp_degree:
        raise RuntimeError("SP group world_size must match frozen sp_degree")
    if group.group_rank != binding.sequence_shard_index:
        raise RuntimeError("SP group slot must match sequence shard index")
    expected_first_rank = binding.branch_index * contract.sp_degree
    expected_ranks = tuple(
        range(expected_first_rank, expected_first_rank + contract.sp_degree)
    )
    if group.spec.ranks != expected_ranks:
        raise RuntimeError("SP group must match deterministic CFG branch mesh")
    if (
        binding.branch is GuidanceBranch.POSITIVE
        and group.spec.ranks[0] != contract.writer_rank
    ):
        raise RuntimeError("writer must own positive SP shard zero")


def _positive_sp_shard(
    tensor: torch.Tensor | None,
    *,
    template: torch.Tensor,
    binding: GuidanceParallelBinding,
    plan: SequenceShardPlan,
    coordinator: GroupCoordinator | None,
) -> torch.Tensor | None:
    if binding.branch is GuidanceBranch.NEGATIVE:
        return None
    if plan.sp_degree == 1:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("positive CFG owner must provide a global tensor")
        return tensor
    if coordinator is None:
        raise RuntimeError("SP>1 requires a sequence-parallel coordinator")
    source = tensor if plan.rank == plan.owner_rank else template
    if not isinstance(source, torch.Tensor):
        raise TypeError("positive SP owner must provide a global tensor")
    return shard_sequence_tensor(
        source,
        plan,
        coordinator,
        sequence_dim=1,
    )


def _prepare_cfg_parallel_state(
    *,
    batch: Req,
    server_args,
    transformer,
    contract: LTX095SequenceParallelContract,
) -> LTX095CFGParallelState:
    if contract.cfg_degree != 2:
        raise ValueError("LTX095 CFG adapter requires CFG=2")
    context = getattr(server_args, "parallel_context", None)
    if not isinstance(context, ParallelContext) or not context.enabled:
        raise RuntimeError("active LTX095 CFG parallel requires ParallelContext")
    binding = resolve_guidance_parallel_binding(
        context,
        writer_rank=contract.writer_rank,
    )
    _validate_sp_mesh(binding, contract)
    cfg_coordinator = GroupCoordinator(binding.cfg_group)
    world_control_coordinator = GroupCoordinator(binding.world_control_group)
    engine = GuidanceParallelEngine(
        binding,
        cfg_coordinator=cfg_coordinator,
        world_control_coordinator=world_control_coordinator,
        control_device="cpu",
    )
    sp_coordinator = None
    if contract.sp_degree > 1:
        sp_coordinator = _bind_sequence_parallel_control(
            transformer,
            binding.sp_group,
        )

    transformer_device = module_device(transformer)
    transformer_dtype = module_dtype(transformer)
    latent_shape = _latent_shape(batch)
    synchronized_mask_shape = _mask_shape(batch, latent_shape)
    batch_size, latent_channels, latent_num_frames, latent_height, latent_width = (
        latent_shape
    )
    patch_size = _positive_int(
        getattr(getattr(transformer, "config", None), "patch_size", None),
        name="transformer.config.patch_size",
    )
    patch_size_t = _positive_int(
        getattr(getattr(transformer, "config", None), "patch_size_t", None),
        name="transformer.config.patch_size_t",
    )
    if (
        latent_num_frames % patch_size_t
        or latent_height % patch_size
        or latent_width % patch_size
    ):
        raise ValueError("latent shape must be divisible by transformer patch sizes")
    sequence_length = (
        latent_num_frames
        // patch_size_t
        * (latent_height // patch_size)
        * (latent_width // patch_size)
    )
    sequence_plan = plan_sequence_shards(
        sequence_length,
        contract.sp_degree,
        owner_rank=0,
    )[binding.sequence_shard_index]
    backend = _sequence_parallel_backend(contract.attention_backend)
    if backend in {
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.SAGE_ATTN,
    } and sequence_plan.padded_length != sequence_plan.global_length:
        raise ValueError(
            f"LTX095 CFG {backend.value} does not support padded sequence shards: "
            f"global_length={sequence_plan.global_length}, "
            f"padded_length={sequence_plan.padded_length}"
        )
    attention_metadata = (
        SequenceParallelMetadata.from_shard_plan(sequence_plan, backend=backend)
        if contract.sp_degree > 1
        else None
    )
    latent_feature_size = (
        latent_channels * patch_size_t * patch_size * patch_size
    )
    mask_feature_size = (
        synchronized_mask_shape[1] * patch_size_t * patch_size * patch_size
    )
    hidden_size = _positive_int(
        contract.hidden_size,
        name="contract.hidden_size",
    )
    branch = binding.branch

    is_positive_owner = (
        branch is GuidanceBranch.POSITIVE
        and binding.sequence_shard_index == sequence_plan.owner_rank
    )
    if is_positive_owner:
        expected_latent_shape = tuple(latent_shape)
        for name in ("latents", "cond_latents"):
            value = getattr(batch, name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_latent_shape:
                raise ValueError(f"writer {name} must match batch.latent_shape")
        if (
            not isinstance(batch.mask_values, torch.Tensor)
            or tuple(batch.mask_values.shape) != synchronized_mask_shape
        ):
            raise ValueError("writer mask_values must match synchronized mask shape")
        global_noise = batch.extra.get(_GLOBAL_NOISE_KEY)
        if (
            not isinstance(global_noise, torch.Tensor)
            or tuple(global_noise.shape) != expected_latent_shape
        ):
            raise ValueError("writer global noise must match batch.latent_shape")

        latents = transformer.sequence_latent(
            batch.latents.to(device=transformer_device, dtype=transformer_dtype)
        )
        conditioning = transformer.sequence_latent(
            batch.cond_latents.to(
                device=transformer_device,
                dtype=transformer_dtype,
            )
        )
        mask = transformer.sequence_latent(
            batch.mask_values.to(
                device=transformer_device,
                dtype=transformer_dtype,
            )
        )
        noise = transformer.sequence_latent(
            global_noise.to(device=transformer_device, dtype=transformer_dtype)
        )
        rope_cos, rope_sin = transformer.rope(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            batch.rope_interpolation_scale,
            None,
        )
        guss_cond = None
        if bool(batch.dynamic_cfg) and bool(
            getattr(batch, "enable_dynamic_cfg_space", True)
        ):
            global_mask = batch.mask_values.to(
                device=transformer_device,
                dtype=transformer_dtype,
            )
            guss_cond = transformer.sequence_latent(
                concrete_mask(
                    global_mask.movedim(1, 2),
                    ksize=3,
                    sigma=0.8,
                )
                .movedim(1, 2)
                .to(device=transformer_device, dtype=transformer_dtype)
            )
    else:
        latents = None
        conditioning = None
        mask = None
        noise = None
        rope_cos = None
        rope_sin = None
        guss_cond = None

    local_sequence_length = sequence_plan.local_length
    latent_template = _packed_template(
        batch_size=batch_size,
        sequence_length=local_sequence_length,
        feature_size=latent_feature_size,
        device=transformer_device,
        dtype=transformer_dtype,
    )
    mask_template = _packed_template(
        batch_size=batch_size,
        sequence_length=local_sequence_length,
        feature_size=mask_feature_size,
        device=transformer_device,
        dtype=transformer_dtype,
    )
    rope_template = _packed_template(
        batch_size=batch_size,
        sequence_length=local_sequence_length,
        feature_size=hidden_size,
        device=transformer_device,
        dtype=torch.float32,
    )

    latents = _positive_sp_shard(
        latents,
        template=latent_template,
        binding=binding,
        plan=sequence_plan,
        coordinator=sp_coordinator,
    )
    conditioning = _positive_sp_shard(
        conditioning,
        template=latent_template,
        binding=binding,
        plan=sequence_plan,
        coordinator=sp_coordinator,
    )
    mask = _positive_sp_shard(
        mask,
        template=mask_template,
        binding=binding,
        plan=sequence_plan,
        coordinator=sp_coordinator,
    )
    noise = _positive_sp_shard(
        noise,
        template=latent_template,
        binding=binding,
        plan=sequence_plan,
        coordinator=sp_coordinator,
    )
    rope_cos = _positive_sp_shard(
        rope_cos,
        template=rope_template,
        binding=binding,
        plan=sequence_plan,
        coordinator=sp_coordinator,
    )
    rope_sin = _positive_sp_shard(
        rope_sin,
        template=rope_template,
        binding=binding,
        plan=sequence_plan,
        coordinator=sp_coordinator,
    )
    if bool(batch.dynamic_cfg) and bool(
        getattr(batch, "enable_dynamic_cfg_space", True)
    ):
        guss_cond = _positive_sp_shard(
            guss_cond,
            template=mask_template,
            binding=binding,
            plan=sequence_plan,
            coordinator=sp_coordinator,
        )

    latents = _replicate_positive(
        engine,
        branch,
        latents,
        template=latent_template,
    )
    conditioning = _replicate_positive(
        engine,
        branch,
        conditioning,
        template=torch.empty_like(latents),
    )
    mask = _replicate_positive(
        engine,
        branch,
        mask,
        template=mask_template,
    )
    noise = _replicate_positive(
        engine,
        branch,
        noise,
        template=torch.empty_like(latents),
    )
    rope_cos = _replicate_positive(
        engine,
        branch,
        rope_cos,
        template=rope_template,
    )
    rope_sin = _replicate_positive(
        engine,
        branch,
        rope_sin,
        template=torch.empty_like(rope_cos),
    )
    if bool(batch.dynamic_cfg) and bool(
        getattr(batch, "enable_dynamic_cfg_space", True)
    ):
        guss_cond = _replicate_positive(
            engine,
            branch,
            guss_cond,
            template=mask_template,
        )

    if branch is GuidanceBranch.POSITIVE:
        prompt_embeds = _required_tensor(
            "prompt_embeds",
            batch.prompt_embeds,
            device=transformer_device,
            dtype=transformer_dtype,
        )
        prompt_attention_mask = _required_tensor(
            "prompt_attention_mask",
            batch.prompt_attention_mask,
            device=transformer_device,
        )
    else:
        prompt_embeds = _required_tensor(
            "negative_prompt_embeds",
            batch.negative_prompt_embeds,
            device=transformer_device,
            dtype=transformer_dtype,
        )
        prompt_attention_mask = _required_tensor(
            "negative_attention_mask",
            batch.negative_attention_mask,
            device=transformer_device,
        )

    return LTX095CFGParallelState(
        binding=binding,
        engine=engine,
        sequence_plan=sequence_plan,
        attention_metadata=attention_metadata,
        sp_coordinator=sp_coordinator,
        latents=latents,
        conditioning=conditioning,
        mask=mask,
        noise=noise,
        rope_interpolation_scale=(rope_cos, rope_sin),
        guss_cond=guss_cond,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        latent_num_frames=latent_num_frames,
        latent_height=latent_height,
        latent_width=latent_width,
    )


def forward_ltx095_cfg_parallel_denoising(
    *,
    batch: Req,
    server_args,
    transformer,
    scheduler,
    contract: LTX095SequenceParallelContract,
    log_info: Callable[..., None],
) -> Req:
    """Run a CFG-only branch pair while preserving all-rank scheduler state."""

    state = _prepare_cfg_parallel_state(
        batch=batch,
        server_args=server_args,
        transformer=transformer,
        contract=contract,
    )
    batch.extra.pop(_GLOBAL_NOISE_KEY, None)
    batch.latents = None
    batch.noisy_latents = None
    batch.noise_pred = None
    batch.cond_latents = None
    batch.cond_masks = None
    batch.mask_values = None

    progress_state = batch.extra.get("runtime_progress_state")
    if progress_state is not None and not isinstance(
        progress_state,
        RuntimeProgressState,
    ):
        progress_state = None
    total_steps = len(batch.timesteps) if batch.timesteps is not None else 0
    cfg_step = int(batch.cfg_step or 0)
    dynamic_cfg = bool(batch.dynamic_cfg)
    enable_dynamic_cfg_space = bool(
        getattr(batch, "enable_dynamic_cfg_space", True)
    )
    transformer_dtype = state.latents.dtype
    transformer_device = state.latents.device

    noise_pred = None
    step_audit: list[dict[str, object]] = []
    batch.extra["ltx095_cfg_parallel_steps"] = step_audit
    cfg_metric_names = (
        "cfg_on_step",
        "cfg_off_step",
        "cfg_transformer_forward",
        "cfg_negative_skipped_step",
        "cfg_prediction_all_reduce",
        "cfg_prediction_broadcast",
        "cfg_scheduler_step",
    )
    if batch.metrics is not None:
        for metric_name in cfg_metric_names:
            batch.metrics.ensure_operation(metric_name)
    cache_branch = (
        CacheBranch.POSITIVE
        if state.binding.branch is GuidanceBranch.POSITIVE
        else CacheBranch.NEGATIVE
    )
    transformer_cache_controller = build_ltx095_transformer_cache_controller(
        batch=batch,
        total_steps=total_steps,
        sp_degree=state.sequence_plan.sp_degree,
        sp_rank=state.sequence_plan.rank,
        cfg_degree=2,
        cfg_rank=state.binding.branch_index,
        coordinator=state.sp_coordinator,
        sp_group_identity=(
            f"{state.binding.sp_group.spec.name}:"
            f"{state.binding.sp_group.spec.ranks}"
        ),
        cfg_group_identity=(
            f"{state.binding.cfg_group.spec.name}:"
            f"{state.binding.cfg_group.spec.ranks}"
        ),
    )
    cache_adapter = (
        transformer_cache_controller.adapter(cache_branch)
        if transformer_cache_controller is not None
        else None
    )
    teacache = (
        cache_adapter
        if (
            transformer_cache_controller is not None
            and transformer_cache_controller.mode is TransformerCacheMode.TEACACHE
        )
        else None
    )
    cache_dit = (
        cache_adapter
        if (
            transformer_cache_controller is not None
            and transformer_cache_controller.mode is TransformerCacheMode.CACHE_DIT
        )
        else None
    )
    log_info(
        "dynamic_cfg=%s cfg_step=%s enable_dynamic_cfg_space=%s "
        "transformer_cache=%s cfg_branch=%s sequence_parallel_rank=%s/%s",
        dynamic_cfg,
        cfg_step,
        enable_dynamic_cfg_space,
        (
            transformer_cache_controller.mode.value
            if transformer_cache_controller is not None
            else "off"
        ),
        state.binding.branch.value,
        state.sequence_plan.rank,
        state.sequence_plan.sp_degree,
    )

    with ltx095_transformer_cache_window_scope(
        batch, transformer_cache_controller
    ), torch.no_grad():
        for step_index, timestep in enumerate(batch.timesteps):
            do_cfg = False
            guidance_scale: float | torch.Tensor = 1.0
            branch_prediction = None
            transformer_executed = False
            audit_entry: dict[str, object] = {
                "step_index": step_index,
                "branch": state.binding.branch.value,
                "do_cfg": False,
                "transformer_executed": False,
                "negative_skipped": False,
                "merge_collective": "pending",
                "scheduler_step": None,
            }
            step_audit.append(audit_entry)

            phase_error: BaseException | None = None
            try:
                batch.step_index = step_index
                do_cfg, guidance_scale = calc_current_cfg(
                    max_cfg=float(batch.guidance_scale),
                    current_step=step_index,
                    max_step=cfg_step if cfg_step > 0 else total_steps,
                    min_cfg=1.0,
                    dynamic_cfg=dynamic_cfg,
                    do_space=enable_dynamic_cfg_space,
                    guss_tensor=state.guss_cond,
                )
                batch.extra["current_step_do_cfg"] = do_cfg
                audit_entry["do_cfg"] = do_cfg
                if batch.metrics is not None:
                    batch.metrics.record_operation(
                        "cfg_on_step" if do_cfg else "cfg_off_step"
                    )
                negative_skipped = (
                    state.binding.branch is GuidanceBranch.NEGATIVE and not do_cfg
                )
                audit_entry["negative_skipped"] = negative_skipped
                if negative_skipped and batch.metrics is not None:
                    batch.metrics.record_operation("cfg_negative_skipped_step")
                if isinstance(guidance_scale, torch.Tensor):
                    batch.extra["current_step_guidance_scale"] = {
                        "mean": float(guidance_scale.mean().item()),
                        "min": float(guidance_scale.min().item()),
                        "max": float(guidance_scale.max().item()),
                    }
                    progress_guidance = float(guidance_scale.mean().item())
                else:
                    batch.extra["current_step_guidance_scale"] = float(
                        guidance_scale
                    )
                    progress_guidance = float(guidance_scale)
                if progress_state is not None:
                    progress_state.update_denoise(
                        step_index=step_index,
                        total_steps=total_steps,
                        timestep_value=float(timestep.item()),
                        cfg_enabled=do_cfg,
                        guidance_scale=progress_guidance,
                    )

                execute_transformer = (
                    state.binding.branch is GuidanceBranch.POSITIVE or do_cfg
                )
                if execute_transformer:
                    timestep_tensor = timestep.expand(state.latents.shape[0]).to(
                        device=transformer_device
                    )
                    branch_prediction = transformer(
                        hidden_states=state.latents,
                        encoder_hidden_states=state.prompt_embeds,
                        timestep=timestep_tensor,
                        encoder_attention_mask=state.prompt_attention_mask,
                        num_frames=state.latent_num_frames,
                        height=state.latent_height,
                        width=state.latent_width,
                        rope_interpolation_scale=batch.rope_interpolation_scale,
                        return_dict=False,
                        cond_latents=state.conditioning,
                        mask_values=state.mask,
                        image_rotary_emb=state.rope_interpolation_scale,
                        time_stemp_index=step_index,
                        teacache=teacache,
                        cache_dit=cache_dit,
                        sequence_parallel_metadata=state.attention_metadata,
                        sequence_parallel_coordinator=state.sp_coordinator,
                    )[0]
                    transformer_executed = True
                    audit_entry["transformer_executed"] = True
                    if batch.metrics is not None:
                        batch.metrics.record_operation("cfg_transformer_forward")
            except BaseException as error:
                phase_error = error
            state.engine.synchronize_phase_error(
                phase_error,
                phase=f"timestep {step_index} transformer",
            )

            phase_error = None
            try:
                noise_pred = state.engine.merge_predictions(
                    branch_prediction if transformer_executed else None,
                    reference_tensor=state.latents,
                    guidance_scale=guidance_scale,
                    do_cfg=do_cfg,
                )
                audit_entry["merge_collective"] = (
                    "all_reduce" if do_cfg else "broadcast"
                )
                if batch.metrics is not None:
                    batch.metrics.record_operation(
                        "cfg_prediction_all_reduce"
                        if do_cfg
                        else "cfg_prediction_broadcast"
                    )
            except BaseException as error:
                phase_error = error
            state.engine.synchronize_phase_error(
                phase_error,
                phase=f"timestep {step_index} CFG merge",
            )

            phase_error = None
            try:
                state.latents = scheduler.step(
                    noise_pred,
                    timestep,
                    state.latents.float(),
                    return_dict=False,
                )[0].to(device=transformer_device, dtype=transformer_dtype)
                audit_entry["scheduler_step"] = step_index
                if batch.metrics is not None:
                    batch.metrics.record_operation("cfg_scheduler_step")
            except BaseException as error:
                phase_error = error
            state.engine.synchronize_phase_error(
                phase_error,
                phase=f"timestep {step_index} scheduler",
            )
            service_checkpoint(
                batch,
                server_args,
                phase=f"denoise_step_{step_index}",
            )

    if noise_pred is None:
        raise RuntimeError("denoising stage produced no noise prediction")
    if state.binding.branch is GuidanceBranch.POSITIVE:
        gathered_latents = (
            gather_sequence_to_owner(
                state.latents,
                state.sequence_plan,
                state.sp_coordinator,
                sequence_dim=1,
            )
            if state.sequence_plan.sp_degree > 1
            else state.latents
        )
    else:
        gathered_latents = None
    if state.binding.global_rank == state.binding.writer_rank:
        if gathered_latents is None:
            raise RuntimeError("CFG writer did not receive gathered positive latents")
        batch.latents = transformer.unsequence_latent(
            gathered_latents,
            num_frames=state.latent_num_frames,
            height=state.latent_height,
            width=state.latent_width,
        )
    else:
        batch.latents = None
    batch.noise_pred = None
    log_info("CFG-parallel denoising complete on %s", state.binding.branch.value)
    return batch


__all__ = (
    "LTX095CFGParallelState",
    "forward_ltx095_cfg_parallel_denoising",
)
