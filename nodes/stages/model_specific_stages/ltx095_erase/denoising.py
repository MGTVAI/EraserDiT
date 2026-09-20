"""LTX095 erase - denoising stage."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import torch

from config.server_args import ServerArgs, get_global_server_args
from config.transformer_cache import TransformerCacheMode
from layers.attention import AttentionBackendEnum, SequenceParallelMetadata
from models.dits.ltx095_parallel import LTX095SequenceParallelContract
from nodes.schedule_batch import Req
from nodes.stages.denoising import DenoisingStage
from nodes.stages.model_specific_stages.ltx095_erase._common import (
    _component_residency_scope,
    ltx095_memory_phase_scope,
    _distributed_context,
    _field_summary,
    _official_vae_parallel_active,
    _onload_module,
    _offload_module,
    _record_official_parallel_event,
    _should_skip_writer_only_stage,
)
from memory.policies.memory_phase_controller import MemoryPhase
from nodes.stages.model_specific_stages.ltx095_erase.cfg_parallel_denoising import (
    forward_ltx095_cfg_parallel_denoising,
)
from distributed.group_coordinator import GroupCoordinator
from distributed.parallel_state import ParallelContext, RuntimeGroup
from parallel.layouts import (
    ShardMetadata,
    TensorLayout,
    TensorLayoutKind,
)
from parallel.sequence_sharding import (
    SequenceShardPlan,
    gather_sequence_to_owner,
    plan_sequence_shards,
    shard_sequence_tensor,
)
from parallel.stage_policy import synchronize_stage_error
from entrypoints.server.control import service_checkpoint
from utils.dynamic_cfg import calc_current_cfg
from utils.distributed_runtime import broadcast_tensor_from_rank
from cache import (
    CacheBranch,
    build_ltx095_transformer_cache_controller,
    ltx095_transformer_cache_window_scope,
)
from utils.mask import concrete_mask
from utils.resource_policy import (
    module_device,
    module_dtype,
)
from utils.runtime_progress import RuntimeProgressState

_P3_GLOBAL_NOISE_KEY = "ltx095_sequence_parallel_global_noise"
_P3_RUNTIME_KEY = "ltx095_sequence_parallel_denoising"
_P3_CONTROL_PROTOCOL_VERSION = 2
_TORCH_DTYPE_SIGNATURES = {
    name: index
    for index, name in enumerate(
        sorted(
            {
                str(getattr(torch, attribute))
                for attribute in dir(torch)
                if isinstance(getattr(torch, attribute), torch.dtype)
            }
        ),
        start=1,
    )
}


@dataclass(frozen=True)
class LTX095SequenceParallelDenoisingRuntime:
    """Explicit immutable metadata for one window's sharded denoising."""

    shard_plan: SequenceShardPlan
    sequence_layout: TensorLayout
    attention_metadata: SequenceParallelMetadata
    data_coordinator: GroupCoordinator
    latent_num_frames: int
    latent_height: int
    latent_width: int


@dataclass(frozen=True)
class _PreparedSequenceParallelInputs:
    runtime: LTX095SequenceParallelDenoisingRuntime
    mask_shape: tuple[int, int, int, int, int]
    latents: torch.Tensor
    cond_latents: torch.Tensor
    mask_values: torch.Tensor
    noise: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    guss_cond_value: torch.Tensor | None
    prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    negative_attention_mask: torch.Tensor
    transformer_cache_controller: object | None


def _branch_cache_adapters(controller, branch: CacheBranch):
    if controller is None:
        return None, None
    adapter = controller.adapter(branch)
    if controller.mode is TransformerCacheMode.TEACACHE:
        return adapter, None
    if controller.mode is TransformerCacheMode.CACHE_DIT:
        return None, adapter
    raise ValueError(f"unsupported Transformer cache mode: {controller.mode}")


@dataclass(frozen=True)
class _PreparedSequenceParallelExecutionSignature:
    static: torch.Tensor
    static_gathered: torch.Tensor
    timesteps: torch.Tensor
    timesteps_gathered: torch.Tensor


def _dtype_signature(dtype: torch.dtype) -> int:
    """Return a collision-free code for every dtype exposed by this torch."""

    name = str(dtype)
    try:
        return _TORCH_DTYPE_SIGNATURES[name]
    except KeyError as error:
        raise TypeError(
            f"unsupported torch dtype in LTX095 SP signature: {name}"
        ) from error


def _float_signature(value: float) -> int:
    return struct.unpack("!q", struct.pack("!d", float(value)))[0]


def _fixed_tensor_signature(tensor: torch.Tensor | None) -> tuple[int, ...]:
    """Encode a tensor into a fixed-width rank/shape/dtype signature."""

    if tensor is None:
        return (0, 0, 0, 0, 0, 0, 0)
    shape = tuple(int(dimension) for dimension in tensor.shape)
    first_four = shape[:4] + (0,) * max(0, 4 - len(shape))
    return (
        1,
        len(shape),
        *first_four,
        _dtype_signature(tensor.dtype),
    )


def _packed_tensor_signature(
    tensor: torch.Tensor | None,
    *,
    global_length: int,
) -> tuple[int, ...]:
    """Encode the semantic global BSC shape shared by owner and peer ranks."""

    if tensor is None:
        return (0, 0, 0, 0, 0)
    return (
        1,
        int(tensor.shape[0]),
        global_length,
        int(tensor.shape[2]),
        _dtype_signature(tensor.dtype),
    )


def _all_gather_sequence_parallel_control(
    local: torch.Tensor,
    coordinator: GroupCoordinator,
) -> torch.Tensor:
    """Gather a fixed-size control record through the trusted coordinator."""

    local = local.contiguous().reshape(-1)
    output = torch.empty(
        local.numel() * coordinator.world_size,
        device=local.device,
        dtype=local.dtype,
    )
    coordinator.all_gather_into_tensor(output, local)
    return output.reshape(coordinator.world_size, local.numel())


def _required_signature_int(name: str, value) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int in the LTX095 SP signature")
    return value


def _optional_signature_int(name: str, value) -> tuple[int, int]:
    if value is None:
        return (0, 0)
    return (1, _required_signature_int(name, value))


def _scheduler_index_signature(
    scheduler,
    *,
    public_name: str,
    private_name: str,
) -> tuple[int, int]:
    value = getattr(scheduler, public_name, None)
    if value is None:
        value = getattr(scheduler, private_name, None)
    return _optional_signature_int(f"scheduler.{public_name}", value)


def _prepare_sequence_parallel_execution_signature(
    batch: Req,
    prepared: _PreparedSequenceParallelInputs,
    scheduler,
) -> _PreparedSequenceParallelExecutionSignature:
    """Build all rank-local control payloads before the first collective."""

    runtime = prepared.runtime
    plan = runtime.shard_plan
    latent_shape = tuple(int(dimension) for dimension in batch.latent_shape)
    mask_shape = prepared.mask_shape
    if not isinstance(batch.timesteps, torch.Tensor):
        raise TypeError("batch.timesteps must be a torch.Tensor in active LTX095 SP")
    if batch.timesteps.ndim != 1:
        raise ValueError("batch.timesteps must be rank-1 in active LTX095 SP")
    timesteps = (
        batch.timesteps.detach()
        .to(device=prepared.latents.device)
        .contiguous()
        .reshape(-1)
        .clone()
    )
    if timesteps.numel() == 0:
        raise ValueError("batch.timesteps must be non-empty in active LTX095 SP")
    if not isinstance(batch.latent_timestep, torch.Tensor):
        raise TypeError(
            "batch.latent_timestep must be a torch.Tensor in active LTX095 SP"
        )
    interpolation_scale = tuple(batch.rope_interpolation_scale)
    if len(interpolation_scale) != 3:
        raise ValueError("batch.rope_interpolation_scale must contain three values")
    effective_inference_steps = _required_signature_int(
        "batch.extra['effective_inference_steps']",
        batch.extra.get("effective_inference_steps"),
    )
    scheduler_order = _required_signature_int(
        "scheduler.order",
        getattr(scheduler, "order", None),
    )
    scheduler_begin_index = _scheduler_index_signature(
        scheduler,
        public_name="begin_index",
        private_name="_begin_index",
    )
    scheduler_step_index = _scheduler_index_signature(
        scheduler,
        public_name="step_index",
        private_name="_step_index",
    )
    fields = (
        _P3_CONTROL_PROTOCOL_VERSION,
        int(bool(batch.dynamic_cfg)),
        int(batch.cfg_step or 0),
        int(bool(getattr(batch, "enable_dynamic_cfg_space", True))),
        timesteps.numel(),
        _dtype_signature(timesteps.dtype),
        scheduler_order,
        *scheduler_begin_index,
        *scheduler_step_index,
        effective_inference_steps,
        *_fixed_tensor_signature(batch.latent_timestep),
        _float_signature(float(batch.guidance_scale)),
        *(_float_signature(value) for value in interpolation_scale),
        *latent_shape,
        *mask_shape,
        plan.sp_degree,
        plan.owner_rank,
        plan.global_length,
        plan.padded_length,
        plan.local_length,
        *_packed_tensor_signature(
            prepared.latents,
            global_length=plan.global_length,
        ),
        *_packed_tensor_signature(
            prepared.cond_latents,
            global_length=plan.global_length,
        ),
        *_packed_tensor_signature(
            prepared.mask_values,
            global_length=plan.global_length,
        ),
        *_packed_tensor_signature(
            prepared.noise,
            global_length=plan.global_length,
        ),
        *_packed_tensor_signature(
            prepared.rope_cos,
            global_length=plan.global_length,
        ),
        *_packed_tensor_signature(
            prepared.rope_sin,
            global_length=plan.global_length,
        ),
        *_packed_tensor_signature(
            prepared.guss_cond_value,
            global_length=plan.global_length,
        ),
        *_fixed_tensor_signature(prepared.prompt_embeds),
        *_fixed_tensor_signature(prepared.prompt_attention_mask),
        *_fixed_tensor_signature(prepared.negative_prompt_embeds),
        *_fixed_tensor_signature(prepared.negative_attention_mask),
    )
    local = torch.tensor(
        fields,
        device=prepared.latents.device,
        dtype=torch.int64,
    )
    world_size = runtime.data_coordinator.world_size
    return _PreparedSequenceParallelExecutionSignature(
        static=local,
        static_gathered=torch.empty(
            local.numel() * world_size,
            device=local.device,
            dtype=local.dtype,
        ),
        timesteps=timesteps,
        timesteps_gathered=torch.empty(
            timesteps.numel() * world_size,
            device=timesteps.device,
            dtype=timesteps.dtype,
        ),
    )


def _validate_sequence_parallel_execution_signature(
    signature: _PreparedSequenceParallelExecutionSignature,
    coordinator: GroupCoordinator,
) -> None:
    """Reject rank-divergent schedules and payload contracts before scatter."""

    coordinator.all_gather_into_tensor(
        signature.static_gathered,
        signature.static,
    )
    gathered = signature.static_gathered.reshape(
        coordinator.world_size,
        signature.static.numel(),
    )
    reference = gathered[0]
    mismatch_slots = [
        slot
        for slot in range(1, coordinator.world_size)
        if not torch.equal(gathered[slot], reference)
    ]
    if mismatch_slots:
        ranks = coordinator.group.spec.ranks
        mismatch_ranks = ", ".join(str(ranks[slot]) for slot in mismatch_slots)
        mismatch_details: list[str] = []
        for peer_slot in mismatch_slots:
            different_fields = torch.nonzero(
                gathered[peer_slot] != reference,
                as_tuple=False,
            ).flatten()
            for field_slot in different_fields[:8].tolist():
                mismatch_details.append(
                    f"slot {field_slot}: rank {ranks[0]}="
                    f"{int(reference[field_slot].item())}, "
                    f"rank {ranks[peer_slot]}="
                    f"{int(gathered[peer_slot, field_slot].item())}"
                )
        details = "; ".join(mismatch_details)
        raise RuntimeError(
            "LTX095 SP execution signature mismatch across ranks: "
            f"reference rank {ranks[0]}, mismatched peer rank(s): "
            f"{mismatch_ranks}; {details}"
        )

    coordinator.all_gather_into_tensor(
        signature.timesteps_gathered,
        signature.timesteps,
    )
    gathered_timesteps = signature.timesteps_gathered.reshape(
        coordinator.world_size,
        signature.timesteps.numel(),
    )
    reference_timesteps = gathered_timesteps[0]
    mismatch_slots = [
        slot
        for slot in range(1, coordinator.world_size)
        if not torch.equal(gathered_timesteps[slot], reference_timesteps)
    ]
    if mismatch_slots:
        ranks = coordinator.group.spec.ranks
        mismatch_ranks = ", ".join(str(ranks[slot]) for slot in mismatch_slots)
        raise RuntimeError(
            "LTX095 SP timestep content mismatch across ranks: "
            f"reference rank {ranks[0]}, mismatched peer rank(s): {mismatch_ranks}"
        )


def _synchronize_sequence_parallel_phase_error(
    local_error: BaseException | None,
    *,
    coordinator: GroupCoordinator,
    device: torch.device,
    phase: str,
) -> None:
    """Propagate a completed stage-boundary failure without losing its cause."""

    local = torch.tensor(
        [int(local_error is not None)],
        device=device,
        dtype=torch.int64,
    )
    gathered = _all_gather_sequence_parallel_control(local, coordinator)
    failed_slots = [
        slot
        for slot in range(coordinator.world_size)
        if int(gathered[slot, 0].item()) != 0
    ]
    if local_error is not None:
        raise local_error
    if failed_slots:
        ranks = coordinator.group.spec.ranks
        failed_ranks = ", ".join(str(ranks[slot]) for slot in failed_slots)
        raise RuntimeError(f"LTX095 SP {phase} failed on peer rank {failed_ranks}")


class LTX095EraseDenoisingStage(DenoisingStage):
    def __init__(
        self,
        transformer,
        scheduler,
        pipeline=None,
        server_args: ServerArgs | None = None,
    ):
        super().__init__(
            transformer,
            server_args if server_args is not None else get_global_server_args(),
        )
        self._scheduler = scheduler
        self._pipeline = pipeline

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        with ltx095_memory_phase_scope(
            batch,
            MemoryPhase.DENOISE,
            component_name="transformer",
        ):
            return self._forward_impl(batch, server_args)

    def _forward_impl(self, batch: Req, server_args: ServerArgs) -> Req:
        if _should_skip_writer_only_stage(server_args, self.__class__.__name__):
            distributed_context = _distributed_context(server_args)
            writer_rank = int(getattr(distributed_context, "writer_rank", 0) or 0)
            _record_official_parallel_event(
                batch,
                "writer_latents_broadcast_wait",
                stage=self.__class__.__name__,
                src_rank=writer_rank,
            )
            batch.latents = broadcast_tensor_from_rank(
                None,
                src_rank=writer_rank,
                device=(
                    module_device(batch.modules["vae"])
                    if batch.modules.get("vae") is not None
                    else server_args.device
                ),
            )
            batch.noise_pred = None
            _record_official_parallel_event(
                batch,
                "writer_latents_broadcast_recv",
                stage=self.__class__.__name__,
                src_rank=writer_rank,
                latents_shape=(
                    tuple(batch.latents.shape)
                    if isinstance(batch.latents, torch.Tensor)
                    else None
                ),
            )
            return batch
        sequence_parallel_contract = _active_sequence_parallel_contract(server_args)
        transformer, policy = _onload_module(batch, server_args, "transformer")
        if _official_vae_parallel_active(server_args):
            _record_official_parallel_event(
                batch,
                "stage_enter",
                stage=self.__class__.__name__,
                rank=int(getattr(_distributed_context(server_args), "rank", 0) or 0),
            )
        try:
            with _component_residency_scope(
                self._pipeline,
                server_args,
                "transformer",
                reason="denoising_stage",
            ):
                if sequence_parallel_contract is not None:
                    if sequence_parallel_contract.cfg_parallel_active:
                        return forward_ltx095_cfg_parallel_denoising(
                            batch=batch,
                            server_args=server_args,
                            transformer=transformer,
                            scheduler=self._scheduler,
                            contract=sequence_parallel_contract,
                            log_info=self.log_info,
                        )
                    return self._forward_sequence_parallel(
                        batch,
                        server_args,
                        transformer,
                        sequence_parallel_contract,
                    )
                transformer_dtype = module_dtype(transformer)
                transformer_device = module_device(transformer)
                latents = batch.latents.to(
                    device=transformer_device, dtype=transformer_dtype
                )
                cond_latents = batch.cond_latents.to(
                    device=transformer_device,
                    dtype=transformer_dtype,
                )
                mask_values = batch.mask_values.to(
                    device=transformer_device,
                    dtype=transformer_dtype,
                )
                prompt_embeds = batch.prompt_embeds.to(
                    device=transformer_device,
                    dtype=transformer_dtype,
                )
                prompt_attention_mask = batch.prompt_attention_mask.to(
                    device=transformer_device
                )
                negative_prompt_embeds = batch.negative_prompt_embeds.to(
                    device=transformer_device,
                    dtype=transformer_dtype,
                )
                negative_attention_mask = batch.negative_attention_mask.to(
                    device=transformer_device
                )
                progress_state = batch.extra.get("runtime_progress_state")
                if progress_state is not None and not isinstance(
                    progress_state, RuntimeProgressState
                ):
                    progress_state = None
                total_steps = len(batch.timesteps) if batch.timesteps is not None else 0
                dynamic_cfg = bool(batch.dynamic_cfg)
                cfg_step = int(batch.cfg_step or 0)
                enable_dynamic_cfg_space = bool(
                    getattr(batch, "enable_dynamic_cfg_space", True)
                )
                guidance_scale_max = float(batch.guidance_scale)
                guidance_scale_min = 1.0
                latent_num_frames = int(latents.shape[2])
                latent_height = int(latents.shape[3])
                latent_width = int(latents.shape[4])
                guss_cond_value = None
                if dynamic_cfg and enable_dynamic_cfg_space:
                    guss_cond_value = (
                        concrete_mask(
                            mask_values.movedim(1, 2),
                            ksize=3,
                            sigma=0.8,
                        )
                        .movedim(1, 2)
                        .to(device=latents.device, dtype=latents.dtype)
                    )
                latents = transformer.sequence_latent(latents)
                cond_latents = transformer.sequence_latent(cond_latents)
                mask_values = transformer.sequence_latent(mask_values)
                transformer_for_forward = self.select_transformer_for_forward(
                    transformer,
                    batch=batch,
                    local_shape=tuple(latents.shape),
                    dynamic_cfg=dynamic_cfg,
                )
                if guss_cond_value is not None:
                    guss_cond_value = transformer.sequence_latent(guss_cond_value)
                image_rotary_emb = transformer.rope(
                    latents,
                    latent_num_frames,
                    latent_height,
                    latent_width,
                    batch.rope_interpolation_scale,
                    None,
                )
                transformer_cache_controller = build_ltx095_transformer_cache_controller(
                    batch=batch,
                    total_steps=total_steps,
                )
                negative_teacache, negative_cache_dit = _branch_cache_adapters(
                    transformer_cache_controller,
                    CacheBranch.NEGATIVE,
                )
                positive_teacache, positive_cache_dit = _branch_cache_adapters(
                    transformer_cache_controller,
                    CacheBranch.POSITIVE,
                )

                self.log_info(
                    "dynamic_cfg=%s cfg_step=%s enable_dynamic_cfg_space=%s "
                    "transformer_cache=%s",
                    dynamic_cfg,
                    cfg_step,
                    enable_dynamic_cfg_space,
                    (
                        transformer_cache_controller.mode.value
                        if transformer_cache_controller is not None
                        else "off"
                    ),
                )

                with ltx095_transformer_cache_window_scope(
                    batch, transformer_cache_controller
                ), torch.no_grad():
                    noise_pred = None
                    for step_index, timestep in enumerate(batch.timesteps):
                        batch.step_index = step_index
                        current_step_do_cfg, current_step_guidance_scale = calc_current_cfg(
                            max_cfg=guidance_scale_max,
                            current_step=step_index,
                            max_step=cfg_step if cfg_step > 0 else total_steps,
                            min_cfg=guidance_scale_min,
                            dynamic_cfg=dynamic_cfg,
                            do_space=enable_dynamic_cfg_space,
                            guss_tensor=guss_cond_value,
                        )
                        batch.extra["current_step_do_cfg"] = current_step_do_cfg
                        if isinstance(current_step_guidance_scale, torch.Tensor):
                            batch.extra["current_step_guidance_scale"] = {
                                "mean": float(current_step_guidance_scale.mean().item()),
                                "min": float(current_step_guidance_scale.min().item()),
                                "max": float(current_step_guidance_scale.max().item()),
                            }
                            progress_guidance = float(
                                current_step_guidance_scale.mean().item()
                            )
                        else:
                            batch.extra["current_step_guidance_scale"] = float(
                                current_step_guidance_scale
                            )
                            progress_guidance = float(current_step_guidance_scale)
                        if progress_state is not None:
                            progress_state.update_denoise(
                                step_index=step_index,
                                total_steps=total_steps,
                                timestep_value=float(timestep.item()),
                                cfg_enabled=current_step_do_cfg,
                                guidance_scale=progress_guidance,
                            )
                        timestep_tensor = timestep.expand(latents.shape[0]).to(
                            device=latents.device
                        )
                        if current_step_do_cfg:
                            noise_pred_uncond = transformer_for_forward(
                                hidden_states=latents,
                                encoder_hidden_states=negative_prompt_embeds,
                                timestep=timestep_tensor,
                                encoder_attention_mask=negative_attention_mask,
                                num_frames=latent_num_frames,
                                height=latent_height,
                                width=latent_width,
                                rope_interpolation_scale=batch.rope_interpolation_scale,
                                return_dict=False,
                                cond_latents=cond_latents,
                                mask_values=mask_values,
                                image_rotary_emb=image_rotary_emb,
                                time_stemp_index=step_index,
                                teacache=negative_teacache,
                                cache_dit=negative_cache_dit,
                            )[0].float()
                            noise_pred_text = transformer_for_forward(
                                hidden_states=latents,
                                encoder_hidden_states=prompt_embeds,
                                timestep=timestep_tensor,
                                encoder_attention_mask=prompt_attention_mask,
                                num_frames=latent_num_frames,
                                height=latent_height,
                                width=latent_width,
                                rope_interpolation_scale=batch.rope_interpolation_scale,
                                return_dict=False,
                                cond_latents=cond_latents,
                                mask_values=mask_values,
                                image_rotary_emb=image_rotary_emb,
                                time_stemp_index=step_index,
                                teacache=positive_teacache,
                                cache_dit=positive_cache_dit,
                            )[0].float()
                            if isinstance(current_step_guidance_scale, torch.Tensor):
                                guidance_tensor = current_step_guidance_scale.to(
                                    device=noise_pred_text.device,
                                    dtype=noise_pred_text.dtype,
                                )
                                noise_pred = noise_pred_uncond + guidance_tensor * (
                                    noise_pred_text - noise_pred_uncond
                                )
                            else:
                                noise_pred = noise_pred_uncond + float(
                                    current_step_guidance_scale
                                ) * (noise_pred_text - noise_pred_uncond)
                        else:
                            noise_pred = transformer_for_forward(
                                hidden_states=latents,
                                encoder_hidden_states=prompt_embeds,
                                timestep=timestep_tensor,
                                encoder_attention_mask=prompt_attention_mask,
                                num_frames=latent_num_frames,
                                height=latent_height,
                                width=latent_width,
                                rope_interpolation_scale=batch.rope_interpolation_scale,
                                return_dict=False,
                                cond_latents=cond_latents,
                                mask_values=mask_values,
                                image_rotary_emb=image_rotary_emb,
                                time_stemp_index=step_index,
                                teacache=positive_teacache,
                                cache_dit=positive_cache_dit,
                            )[0].float()
                        latents = self._scheduler.step(
                            noise_pred,
                            timestep,
                            latents.float(),
                            return_dict=False,
                        )[0].to(device=transformer_device, dtype=transformer_dtype)
                        service_checkpoint(
                            batch,
                            server_args,
                            phase=f"denoise_step_{step_index}",
                        )

                if noise_pred is None:
                    raise RuntimeError("denoising stage produced no noise prediction")

                batch.noise_pred = transformer.unsequence_latent(
                    noise_pred.to(device=transformer_device, dtype=transformer_dtype),
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                )
                batch.latents = transformer.unsequence_latent(
                    latents,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                )
                if _official_vae_parallel_active(server_args):
                    distributed_context = _distributed_context(server_args)
                    writer_rank = int(getattr(distributed_context, "writer_rank", 0) or 0)
                    batch.latents = broadcast_tensor_from_rank(
                        batch.latents.contiguous(),
                        src_rank=writer_rank,
                        device=batch.latents.device,
                    )
                    _record_official_parallel_event(
                        batch,
                        "writer_latents_broadcast_send",
                        stage=self.__class__.__name__,
                        src_rank=writer_rank,
                        latents_shape=tuple(batch.latents.shape),
                    )
                    _record_official_parallel_event(
                        batch,
                        "stage_complete",
                        stage=self.__class__.__name__,
                        rank=int(
                            getattr(_distributed_context(server_args), "rank", 0) or 0
                        ),
                        latents_shape=tuple(batch.latents.shape),
                    )
                self.log_info(
                    "%s | %s",
                    _field_summary("noise_pred", batch.noise_pred),
                    _field_summary("latents", batch.latents),
                )
                return batch
        finally:
            _offload_module(
                batch,
                server_args,
                "transformer",
                policy,
                reason="denoising_stage",
            )

    def _forward_sequence_parallel(
        self,
        batch: Req,
        server_args: ServerArgs,
        transformer,
        contract: LTX095SequenceParallelContract,
    ) -> Req:
        parallel_context = getattr(server_args, "parallel_context", None)
        prepared = None
        execution_signature = None
        local_error: BaseException | None = None
        try:
            prepared = _prepare_sequence_parallel_inputs(
                batch=batch,
                server_args=server_args,
                transformer=transformer,
                contract=contract,
            )
            execution_signature = _prepare_sequence_parallel_execution_signature(
                batch,
                prepared,
                self._scheduler,
            )
        except BaseException as error:
            local_error = error

        synchronize_stage_error(
            local_error,
            parallel_context if isinstance(parallel_context, ParallelContext) else None,
        )
        if local_error is not None:
            raise local_error
        if prepared is None or execution_signature is None:
            raise RuntimeError(
                "sequence-parallel denoising preparation returned no inputs or "
                "execution signature"
            )

        runtime = prepared.runtime
        plan = runtime.shard_plan
        coordinator = runtime.data_coordinator
        _validate_sequence_parallel_execution_signature(
            execution_signature,
            coordinator,
        )
        del execution_signature
        prompt_embeds = prepared.prompt_embeds
        prompt_attention_mask = prepared.prompt_attention_mask
        negative_prompt_embeds = prepared.negative_prompt_embeds
        negative_attention_mask = prepared.negative_attention_mask
        transformer_cache_controller = prepared.transformer_cache_controller
        latents = shard_sequence_tensor(
            prepared.latents,
            plan,
            coordinator,
            sequence_dim=1,
        )
        cond_latents = shard_sequence_tensor(
            prepared.cond_latents,
            plan,
            coordinator,
            sequence_dim=1,
        )
        mask_values = shard_sequence_tensor(
            prepared.mask_values,
            plan,
            coordinator,
            sequence_dim=1,
        )
        local_noise = shard_sequence_tensor(
            prepared.noise,
            plan,
            coordinator,
            sequence_dim=1,
        )
        rope_cos = shard_sequence_tensor(
            prepared.rope_cos,
            plan,
            coordinator,
            sequence_dim=1,
        )
        rope_sin = shard_sequence_tensor(
            prepared.rope_sin,
            plan,
            coordinator,
            sequence_dim=1,
        )
        guss_cond_value = None
        if prepared.guss_cond_value is not None:
            guss_cond_value = shard_sequence_tensor(
                prepared.guss_cond_value,
                plan,
                coordinator,
                sequence_dim=1,
            )

        batch.extra[_P3_RUNTIME_KEY] = runtime
        batch.extra.pop(_P3_GLOBAL_NOISE_KEY, None)
        batch.latents = None
        batch.noisy_latents = None
        batch.noise_pred = None
        batch.cond_latents = None
        batch.cond_masks = None
        batch.mask_values = None

        # LatentPreparationStage has already applied this exact global noise once
        # through scheduler.scale_noise.  Its local shard is retained only as the
        # explicit same-plan initialization contract and diagnostic evidence; it
        # must not be applied again and change the frozen P2 numerical path.
        del local_noise
        del prepared

        progress_state = batch.extra.get("runtime_progress_state")
        if progress_state is not None and not isinstance(
            progress_state, RuntimeProgressState
        ):
            progress_state = None
        total_steps = len(batch.timesteps) if batch.timesteps is not None else 0
        dynamic_cfg = bool(batch.dynamic_cfg)
        cfg_step = int(batch.cfg_step or 0)
        enable_dynamic_cfg_space = bool(
            getattr(batch, "enable_dynamic_cfg_space", True)
        )
        guidance_scale_max = float(batch.guidance_scale)
        guidance_scale_min = 1.0
        transformer_dtype = latents.dtype
        transformer_device = latents.device
        image_rotary_emb = (rope_cos, rope_sin)
        transformer_for_forward = self.select_transformer_for_forward(
            transformer,
            batch=batch,
            local_shape=tuple(latents.shape),
            dynamic_cfg=dynamic_cfg,
        )
        negative_teacache, negative_cache_dit = _branch_cache_adapters(
            transformer_cache_controller,
            CacheBranch.NEGATIVE,
        )
        positive_teacache, positive_cache_dit = _branch_cache_adapters(
            transformer_cache_controller,
            CacheBranch.POSITIVE,
        )

        self.log_info(
            "dynamic_cfg=%s cfg_step=%s enable_dynamic_cfg_space=%s "
            "transformer_cache=%s sequence_parallel_rank=%s/%s",
            dynamic_cfg,
            cfg_step,
            enable_dynamic_cfg_space,
            (
                transformer_cache_controller.mode.value
                if transformer_cache_controller is not None
                else "off"
            ),
            plan.rank,
            plan.sp_degree,
        )

        with ltx095_transformer_cache_window_scope(
            batch, transformer_cache_controller
        ), torch.no_grad():
            noise_pred = None
            for step_index, timestep in enumerate(batch.timesteps):
                phase_error: BaseException | None = None
                try:
                    batch.step_index = step_index
                    (
                        current_step_do_cfg,
                        current_step_guidance_scale,
                    ) = calc_current_cfg(
                        max_cfg=guidance_scale_max,
                        current_step=step_index,
                        max_step=cfg_step if cfg_step > 0 else total_steps,
                        min_cfg=guidance_scale_min,
                        dynamic_cfg=dynamic_cfg,
                        do_space=enable_dynamic_cfg_space,
                        guss_tensor=guss_cond_value,
                    )
                    batch.extra["current_step_do_cfg"] = current_step_do_cfg
                    if isinstance(current_step_guidance_scale, torch.Tensor):
                        batch.extra["current_step_guidance_scale"] = {
                            "mean": float(current_step_guidance_scale.mean().item()),
                            "min": float(current_step_guidance_scale.min().item()),
                            "max": float(current_step_guidance_scale.max().item()),
                        }
                        progress_guidance = float(
                            current_step_guidance_scale.mean().item()
                        )
                    else:
                        batch.extra["current_step_guidance_scale"] = float(
                            current_step_guidance_scale
                        )
                        progress_guidance = float(current_step_guidance_scale)
                    if progress_state is not None:
                        progress_state.update_denoise(
                            step_index=step_index,
                            total_steps=total_steps,
                            timestep_value=float(timestep.item()),
                            cfg_enabled=current_step_do_cfg,
                            guidance_scale=progress_guidance,
                        )
                    timestep_tensor = timestep.expand(latents.shape[0]).to(
                        device=latents.device
                    )
                except BaseException as error:
                    phase_error = error
                _synchronize_sequence_parallel_phase_error(
                    phase_error,
                    coordinator=coordinator,
                    device=latents.device,
                    phase=f"timestep {step_index} preparation",
                )

                if current_step_do_cfg:
                    # This stage checkpoint begins only after a transformer call
                    # returns or raises back into the stage. Rank-asymmetric Task5
                    # block failures before an internal Ulysses collective require
                    # Task5-local synchronization and are outside this boundary.
                    phase_error = None
                    try:
                        noise_pred_uncond = transformer_for_forward(
                            hidden_states=latents,
                            encoder_hidden_states=negative_prompt_embeds,
                            timestep=timestep_tensor,
                            encoder_attention_mask=negative_attention_mask,
                            num_frames=runtime.latent_num_frames,
                            height=runtime.latent_height,
                            width=runtime.latent_width,
                            rope_interpolation_scale=batch.rope_interpolation_scale,
                            return_dict=False,
                            cond_latents=cond_latents,
                            mask_values=mask_values,
                            image_rotary_emb=image_rotary_emb,
                            time_stemp_index=step_index,
                            teacache=negative_teacache,
                            cache_dit=negative_cache_dit,
                            sequence_parallel_metadata=runtime.attention_metadata,
                            sequence_parallel_coordinator=coordinator,
                        )[0].float()
                    except BaseException as error:
                        phase_error = error
                    _synchronize_sequence_parallel_phase_error(
                        phase_error,
                        coordinator=coordinator,
                        device=latents.device,
                        phase=f"timestep {step_index} negative transformer",
                    )

                    phase_error = None
                    try:
                        noise_pred_text = transformer_for_forward(
                            hidden_states=latents,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timestep_tensor,
                            encoder_attention_mask=prompt_attention_mask,
                            num_frames=runtime.latent_num_frames,
                            height=runtime.latent_height,
                            width=runtime.latent_width,
                            rope_interpolation_scale=batch.rope_interpolation_scale,
                            return_dict=False,
                            cond_latents=cond_latents,
                            mask_values=mask_values,
                            image_rotary_emb=image_rotary_emb,
                            time_stemp_index=step_index,
                            teacache=positive_teacache,
                            cache_dit=positive_cache_dit,
                            sequence_parallel_metadata=runtime.attention_metadata,
                            sequence_parallel_coordinator=coordinator,
                        )[0].float()
                    except BaseException as error:
                        phase_error = error
                    _synchronize_sequence_parallel_phase_error(
                        phase_error,
                        coordinator=coordinator,
                        device=latents.device,
                        phase=f"timestep {step_index} positive transformer",
                    )
                else:
                    phase_error = None
                    try:
                        noise_pred = transformer_for_forward(
                            hidden_states=latents,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timestep_tensor,
                            encoder_attention_mask=prompt_attention_mask,
                            num_frames=runtime.latent_num_frames,
                            height=runtime.latent_height,
                            width=runtime.latent_width,
                            rope_interpolation_scale=batch.rope_interpolation_scale,
                            return_dict=False,
                            cond_latents=cond_latents,
                            mask_values=mask_values,
                            image_rotary_emb=image_rotary_emb,
                            time_stemp_index=step_index,
                            teacache=positive_teacache,
                            cache_dit=positive_cache_dit,
                            sequence_parallel_metadata=runtime.attention_metadata,
                            sequence_parallel_coordinator=coordinator,
                        )[0].float()
                    except BaseException as error:
                        phase_error = error
                    _synchronize_sequence_parallel_phase_error(
                        phase_error,
                        coordinator=coordinator,
                        device=latents.device,
                        phase=f"timestep {step_index} transformer",
                    )

                phase_error = None
                try:
                    if current_step_do_cfg:
                        if isinstance(current_step_guidance_scale, torch.Tensor):
                            guidance_tensor = current_step_guidance_scale.to(
                                device=noise_pred_text.device,
                                dtype=noise_pred_text.dtype,
                            )
                            noise_pred = noise_pred_uncond + guidance_tensor * (
                                noise_pred_text - noise_pred_uncond
                            )
                        else:
                            noise_pred = noise_pred_uncond + float(
                                current_step_guidance_scale
                            ) * (noise_pred_text - noise_pred_uncond)
                    latents = self._scheduler.step(
                        noise_pred,
                        timestep,
                        latents.float(),
                        return_dict=False,
                    )[0].to(device=transformer_device, dtype=transformer_dtype)
                except BaseException as error:
                    phase_error = error
                _synchronize_sequence_parallel_phase_error(
                    phase_error,
                    coordinator=coordinator,
                    device=latents.device,
                    phase=f"timestep {step_index} guidance and scheduler",
                )
                service_checkpoint(
                    batch,
                    server_args,
                    phase=f"denoise_step_{step_index}",
                )

        phase_error = None
        try:
            if noise_pred is None:
                raise RuntimeError("denoising stage produced no noise prediction")
        except BaseException as error:
            phase_error = error
        _synchronize_sequence_parallel_phase_error(
            phase_error,
            coordinator=coordinator,
            device=latents.device,
            phase="final noise prediction",
        )

        gathered_latents = gather_sequence_to_owner(
            latents,
            plan,
            coordinator,
            sequence_dim=1,
        )
        batch.noise_pred = None
        if gathered_latents is None:
            batch.latents = None
        else:
            batch.latents = transformer.unsequence_latent(
                gathered_latents,
                num_frames=runtime.latent_num_frames,
                height=runtime.latent_height,
                width=runtime.latent_width,
            )
        del latents, cond_latents, mask_values, rope_cos, rope_sin
        del image_rotary_emb, noise_pred, gathered_latents
        if guss_cond_value is not None:
            del guss_cond_value
        self.log_info(
            "%s | %s",
            _field_summary("noise_pred", batch.noise_pred),
            _field_summary("latents", batch.latents),
        )
        return batch


def _active_sequence_parallel_contract(
    server_args: ServerArgs,
) -> LTX095SequenceParallelContract | None:
    contract = getattr(server_args, "ltx095_sequence_parallel_contract", None)
    if contract is None:
        return None
    if not isinstance(contract, LTX095SequenceParallelContract):
        raise TypeError(
            "ltx095_sequence_parallel_contract must be a frozen "
            "LTX095SequenceParallelContract"
        )
    return contract if contract.active else None


def _prepare_sequence_parallel_inputs(
    *,
    batch: Req,
    server_args: ServerArgs,
    transformer,
    contract: LTX095SequenceParallelContract,
) -> _PreparedSequenceParallelInputs:
    context = _validate_sequence_parallel_context(server_args, contract)
    group = context.current_group("sp_")
    _validate_sequence_parallel_group(group, context, contract)
    coordinator = _bind_sequence_parallel_control(transformer, group)

    transformer_dtype = module_dtype(transformer)
    transformer_device = module_device(transformer)
    latent_shape = _validate_latent_shape(batch.latent_shape)
    batch_size, latent_channels, latent_num_frames, latent_height, latent_width = (
        latent_shape
    )
    patch_size = _positive_config_int(transformer, "patch_size")
    patch_size_t = _positive_config_int(transformer, "patch_size_t")
    if (
        latent_num_frames % patch_size_t != 0
        or latent_height % patch_size != 0
        or latent_width % patch_size != 0
    ):
        raise ValueError(
            "latent_shape must be divisible by transformer patch sizes, "
            f"got latent_shape={latent_shape}, patch_size={patch_size}, "
            f"patch_size_t={patch_size_t}"
        )
    global_length = (
        latent_num_frames
        // patch_size_t
        * (latent_height // patch_size)
        * (latent_width // patch_size)
    )
    plans = plan_sequence_shards(
        global_length,
        contract.sp_degree,
        owner_rank=0,
    )
    plan = plans[group.group_rank]
    backend = _sequence_parallel_backend(contract.attention_backend)
    if backend in {
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.SAGE_ATTN,
    } and plan.padded_length != plan.global_length:
        raise ValueError(
            f"LTX095 P3 {backend.value} does not support padded sequence shards: "
            f"global_length={plan.global_length}, "
            f"padded_length={plan.padded_length}"
        )
    attention_metadata = SequenceParallelMetadata.from_shard_plan(
        plan,
        backend=backend,
    )
    latent_feature_size = latent_channels * patch_size_t * patch_size * patch_size
    sequence_layout = TensorLayout(
        kind=TensorLayoutKind.SEQUENCE_SHARDED,
        global_shape=(batch_size, plan.global_length, latent_feature_size),
        local_shape=(batch_size, plan.local_length, latent_feature_size),
        shard=ShardMetadata(
            shard_dim=1,
            shard_index=plan.rank,
            shard_count=plan.sp_degree,
            offset=min(plan.local_start, plan.global_length),
            valid_length=plan.valid_local_length,
            padded_length=plan.local_length,
        ),
    )
    runtime = LTX095SequenceParallelDenoisingRuntime(
        shard_plan=plan,
        sequence_layout=sequence_layout,
        attention_metadata=attention_metadata,
        data_coordinator=coordinator,
        latent_num_frames=latent_num_frames,
        latent_height=latent_height,
        latent_width=latent_width,
    )

    is_owner = group.group_rank == plan.owner_rank
    writer_owned = bool(batch.extra.get("ltx095_sp_writer_owned_runtime"))
    if is_owner:
        _validate_global_video_tensor("cond_latents", batch.cond_latents, latent_shape)
        mask_shape = _validate_mask_tensor(batch.mask_values, latent_shape)
        if writer_owned:
            synchronized_mask_shape = _validate_synchronized_mask_shape(
                batch.extra.get("ltx095_sp_mask_shape"),
                latent_shape,
            )
            if mask_shape != synchronized_mask_shape:
                raise ValueError(
                    "writer mask_values shape must match synchronized "
                    "ltx095_sp_mask_shape, "
                    f"got mask_values={mask_shape}, "
                    f"ltx095_sp_mask_shape={synchronized_mask_shape}"
                )
        _validate_global_video_tensor("latents", batch.latents, latent_shape)
        noise = batch.extra.get(_P3_GLOBAL_NOISE_KEY)
        _validate_global_video_tensor("noise", noise, latent_shape)
        latents = transformer.sequence_latent(
            batch.latents.to(device=transformer_device, dtype=transformer_dtype)
        )
        cond_latents = transformer.sequence_latent(
            batch.cond_latents.to(
                device=transformer_device,
                dtype=transformer_dtype,
            )
        )
        mask_values = transformer.sequence_latent(
            batch.mask_values.to(
                device=transformer_device,
                dtype=transformer_dtype,
            )
        )
        noise = transformer.sequence_latent(
            noise.to(device=transformer_device, dtype=transformer_dtype)
        )
        rope_cos, rope_sin = transformer.rope(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            batch.rope_interpolation_scale,
            None,
        )
        guss_cond_value = None
        if bool(batch.dynamic_cfg) and bool(
            getattr(batch, "enable_dynamic_cfg_space", True)
        ):
            global_mask = batch.mask_values.to(
                device=transformer_device,
                dtype=transformer_dtype,
            )
            guss_cond_value = (
                concrete_mask(
                    global_mask.movedim(1, 2),
                    ksize=3,
                    sigma=0.8,
                )
                .movedim(1, 2)
                .to(
                    device=transformer_device,
                    dtype=transformer_dtype,
                )
            )
            guss_cond_value = transformer.sequence_latent(guss_cond_value)
    else:
        if writer_owned:
            mask_shape = _validate_synchronized_mask_shape(
                batch.extra.get("ltx095_sp_mask_shape"),
                latent_shape,
            )
        else:
            _validate_global_video_tensor(
                "cond_latents",
                batch.cond_latents,
                latent_shape,
            )
            mask_shape = _validate_mask_tensor(batch.mask_values, latent_shape)
        mask_feature_size = mask_shape[1] * patch_size_t * patch_size * patch_size
        latents = _local_sequence_template(
            batch_size,
            plan.local_length,
            latent_feature_size,
            device=transformer_device,
            dtype=transformer_dtype,
        )
        cond_latents = _local_sequence_template(
            batch_size,
            plan.local_length,
            latent_feature_size,
            device=transformer_device,
            dtype=transformer_dtype,
        )
        mask_values = _local_sequence_template(
            batch_size,
            plan.local_length,
            mask_feature_size,
            device=transformer_device,
            dtype=transformer_dtype,
        )
        noise = _local_sequence_template(
            batch_size,
            plan.local_length,
            latent_feature_size,
            device=transformer_device,
            dtype=transformer_dtype,
        )
        rope_feature_size = contract.hidden_size
        if type(rope_feature_size) is not int or rope_feature_size <= 0:
            raise ValueError("active LTX095 P3 requires a frozen positive hidden_size")
        rope_cos = _local_sequence_template(
            batch_size,
            plan.local_length,
            rope_feature_size,
            device=transformer_device,
            dtype=torch.float32,
        )
        rope_sin = _local_sequence_template(
            batch_size,
            plan.local_length,
            rope_feature_size,
            device=transformer_device,
            dtype=torch.float32,
        )
        guss_cond_value = (
            _local_sequence_template(
                batch_size,
                plan.local_length,
                mask_feature_size,
                device=transformer_device,
                dtype=transformer_dtype,
            )
            if bool(batch.dynamic_cfg)
            and bool(getattr(batch, "enable_dynamic_cfg_space", True))
            else None
        )

    _validate_packed_sequence_tensor("latents", latents, plan, is_owner)
    _validate_packed_sequence_tensor("cond_latents", cond_latents, plan, is_owner)
    _validate_packed_sequence_tensor("mask_values", mask_values, plan, is_owner)
    _validate_packed_sequence_tensor("noise", noise, plan, is_owner)
    _validate_packed_sequence_tensor("rope_cos", rope_cos, plan, is_owner)
    _validate_packed_sequence_tensor("rope_sin", rope_sin, plan, is_owner)
    if guss_cond_value is not None:
        _validate_packed_sequence_tensor(
            "guss_cond_value",
            guss_cond_value,
            plan,
            is_owner,
        )

    prompt_embeds = _move_required_tensor(
        "prompt_embeds",
        batch.prompt_embeds,
        device=transformer_device,
        dtype=transformer_dtype,
    )
    prompt_attention_mask = _move_required_tensor(
        "prompt_attention_mask",
        batch.prompt_attention_mask,
        device=transformer_device,
    )
    negative_prompt_embeds = _move_required_tensor(
        "negative_prompt_embeds",
        batch.negative_prompt_embeds,
        device=transformer_device,
        dtype=transformer_dtype,
    )
    negative_attention_mask = _move_required_tensor(
        "negative_attention_mask",
        batch.negative_attention_mask,
        device=transformer_device,
    )
    total_steps = len(batch.timesteps) if batch.timesteps is not None else 0
    transformer_cache_controller = build_ltx095_transformer_cache_controller(
        batch=batch,
        total_steps=total_steps,
        sp_degree=runtime.attention_metadata.sp_degree,
        sp_rank=runtime.attention_metadata.rank,
        cfg_degree=1,
        cfg_rank=0,
        coordinator=runtime.data_coordinator,
    )
    return _PreparedSequenceParallelInputs(
        runtime=runtime,
        mask_shape=mask_shape,
        latents=latents,
        cond_latents=cond_latents,
        mask_values=mask_values,
        noise=noise,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        guss_cond_value=guss_cond_value,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_attention_mask=negative_attention_mask,
        transformer_cache_controller=transformer_cache_controller,
    )


def _validate_sequence_parallel_context(
    server_args: ServerArgs,
    contract: LTX095SequenceParallelContract,
) -> ParallelContext:
    context = getattr(server_args, "parallel_context", None)
    if not isinstance(context, ParallelContext) or not context.enabled:
        raise RuntimeError(
            "active LTX095 sequence parallel requires an enabled ParallelContext"
        )
    plan = context.plan
    frozen_values = (
        ("world_size", plan.world_size, contract.world_size),
        ("sp_degree", plan.sp_degree, contract.sp_degree),
        ("cfg_degree", plan.cfg_degree, contract.cfg_degree),
        ("vae_degree", plan.vae_degree, contract.vae_degree),
        ("writer_rank", plan.writer_rank, contract.writer_rank),
    )
    for name, actual, expected in frozen_values:
        if actual != expected:
            raise RuntimeError(
                f"parallel context {name} changed after LTX095 P3 capability "
                f"freeze: expected {expected}, got {actual}"
            )
    if contract.distributed_compute_mode != "entry_only":
        raise RuntimeError(
            "active LTX095 P3 contract requires distributed_compute_mode='entry_only'"
        )
    if server_args.attention_backend != contract.attention_backend:
        raise RuntimeError(
            "attention_backend changed after LTX095 P3 capability freeze: "
            f"expected {contract.attention_backend!r}, "
            f"got {server_args.attention_backend!r}"
        )
    return context


def _validate_sequence_parallel_group(
    group: RuntimeGroup,
    context: ParallelContext,
    contract: LTX095SequenceParallelContract,
) -> None:
    if not isinstance(group, RuntimeGroup) or not group.is_member:
        raise RuntimeError("current SP group must be a member RuntimeGroup")
    if group.spec.ranks != tuple(range(contract.world_size)):
        raise RuntimeError(
            "LTX095 P3 SP group must contain every global rank in order, "
            f"got ranks={group.spec.ranks!r}"
        )
    if group.world_size != contract.sp_degree:
        raise RuntimeError("SP group world_size must match frozen sp_degree")
    if group.global_rank != context.global_rank:
        raise RuntimeError("SP group slot must match parallel context global_rank")
    if group.spec.ranks[0] != contract.writer_rank:
        raise RuntimeError("LTX095 P3 writer must occupy SP group slot 0")


def _bind_sequence_parallel_control(transformer, group: RuntimeGroup):
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
    coordinator_group = getattr(coordinator, "group", None)
    if getattr(getattr(coordinator_group, "spec", None), "ranks", None) != (
        group.spec.ranks
    ):
        raise RuntimeError("SP data coordinator group ranks changed")
    if getattr(coordinator_group, "group_rank", None) != group.group_rank:
        raise RuntimeError("SP data coordinator rank changed")
    if getattr(coordinator_group, "process_group", None) is not group.process_group:
        raise RuntimeError("SP data coordinator process group changed")
    if getattr(coordinator, "rank", None) != group.group_rank:
        raise RuntimeError("SP data coordinator rank must match group slot")
    if getattr(coordinator, "world_size", None) != group.world_size:
        raise RuntimeError("SP data coordinator world_size must match SP group")
    return coordinator


def _validate_latent_shape(shape) -> tuple[int, int, int, int, int]:
    if type(shape) is not tuple or len(shape) != 5:
        raise TypeError("batch.latent_shape must be an explicit rank-5 tuple")
    if any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("batch.latent_shape dimensions must be positive ints")
    return shape


def _positive_config_int(transformer, name: str) -> int:
    value = getattr(getattr(transformer, "config", None), name, None)
    if type(value) is not int or value <= 0:
        raise ValueError(f"transformer config {name} must be a positive int")
    return value


def _validate_global_video_tensor(
    name: str,
    tensor,
    expected_shape: tuple[int, int, int, int, int],
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor on the writer")
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name} must match batch.latent_shape {expected_shape}, "
            f"got {tuple(tensor.shape)}"
        )


def _validate_mask_tensor(
    tensor,
    latent_shape: tuple[int, int, int, int, int],
) -> tuple[int, int, int, int, int]:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 5:
        raise TypeError("mask_values must be a rank-5 torch.Tensor before scatter")
    mask_shape = tuple(tensor.shape)
    if (
        mask_shape[0] != latent_shape[0]
        or mask_shape[2:] != latent_shape[2:]
        or mask_shape[1] <= 0
    ):
        raise ValueError(
            "mask_values batch/frame/spatial dimensions must match latent_shape, "
            f"got mask_values={mask_shape}, latent_shape={latent_shape}"
        )
    return mask_shape


def _validate_synchronized_mask_shape(
    shape,
    latent_shape: tuple[int, int, int, int, int],
) -> tuple[int, int, int, int, int]:
    if type(shape) is not tuple or len(shape) != 5:
        raise TypeError(
            "batch.extra['ltx095_sp_mask_shape'] must be an explicit rank-5 tuple"
        )
    if any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError(
            "batch.extra['ltx095_sp_mask_shape'] dimensions must be positive ints"
        )
    if shape[0] != latent_shape[0] or shape[2:] != latent_shape[2:]:
        raise ValueError(
            "synchronized mask shape batch/frame/spatial dimensions must match "
            f"latent_shape, got mask_shape={shape}, latent_shape={latent_shape}"
        )
    return shape


def _local_sequence_template(
    batch_size: int,
    local_length: int,
    feature_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(
        (batch_size, local_length, feature_size),
        device=device,
        dtype=dtype,
    )


def _validate_packed_sequence_tensor(
    name: str,
    tensor: torch.Tensor,
    plan: SequenceShardPlan,
    is_owner: bool,
) -> None:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        raise ValueError(f"packed {name} must use rank-3 BSC layout")
    expected_length = plan.global_length if is_owner else plan.local_length
    if tensor.shape[1] != expected_length:
        raise ValueError(
            f"packed {name} sequence length must be {expected_length}, "
            f"got {tensor.shape[1]}"
        )


def _move_required_tensor(
    name: str,
    tensor,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return tensor.to(device=device, dtype=dtype)


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
            "active LTX095 P3 attention backend must be 'sdpa', 'flash_attn', "
            f"'sage_attn', or 'sage_fp8', got {name!r}"
        ) from error
