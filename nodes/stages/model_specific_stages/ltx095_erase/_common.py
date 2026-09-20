"""Shared helpers for LTX095 erase stages."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from config.server_args import ServerArgs
from models.vaes.ltx095_parallel import (
    LTX095VAEParallelAdapter,
    VAEExecutionDecision,
)
from nodes.schedule_batch import Req
from distributed import GroupCoordinator
from distributed.parallel_state import (
    ParallelContext,
    resolve_group_control_device,
)
from pipelines.runtime.windowing.sp_dispatch import (
    resolve_active_ltx095_window_commit_context,
)
from memory.policies.memory_phase_controller import MemoryPhase
from parallel.stage_policy import (
    synchronize_stage_bool,
    synchronize_stage_error,
)
from parallel.vae_parallel import (
    VAEOperation,
    VAEParallelBinding,
    VAEParallelMetrics,
)
from utils.distributed_runtime import (
    broadcast_tensor_from_rank,
    get_runtime_official_parallel_context,
)
from memory.backends.flexible_module_extent_base import (
    FlexibleModuleExtentBase,
)
from utils.resource_policy import (
    maybe_pin_tensor,
)


def _field_summary(name: str, value) -> str:
    if isinstance(value, torch.Tensor):
        return f"{name}=shape{tuple(value.shape)},dtype={value.dtype},device={value.device}"
    return f"{name}={value}"


def _distributed_context(server_args: ServerArgs) -> Any:
    return getattr(server_args, "distributed_context", None)


def _official_parallel_context(server_args: ServerArgs) -> Any:
    return getattr(server_args, "official_parallel_context", None) or get_runtime_official_parallel_context()


def _official_vae_parallel_active(server_args: ServerArgs) -> bool:
    context = _official_parallel_context(server_args)
    return bool(
        context is not None
        and getattr(context, "enabled", False)
        and str(getattr(context, "distributed_compute_mode", "entry_only")) == "official_vae_parallel"
        and bool(getattr(context, "vae_parallel_supported", True))
        and bool(getattr(context, "vae_parallel_enabled", False))
    )


@contextmanager
def _component_residency_scope(
    pipeline,
    server_args,
    component_name: str,
    *,
    reason: str,
    synchronize: bool = True,
):
    adapter = getattr(pipeline, "_memory_adapter", None)
    if adapter is None:
        yield False
        return
    snapshot_fn = getattr(adapter, "snapshot", None)
    if (
        callable(snapshot_fn)
        and snapshot_fn().get("active_component_name") == component_name
    ):
        yield True
        return

    decision = adapter.plan_component_residency(component_name)
    if (
        not decision.enabled
        and decision.fallback_reason == "dynamic_offload_disabled"
    ):
        yield False
        return

    writer_only_vae = _official_vae_parallel_active(server_args)
    rank_owned_vae = resolve_ltx095_vae_binding(server_args) is not None
    parallel_context = getattr(server_args, "parallel_context", None)
    if (
        not synchronize
        or writer_only_vae
        or rank_owned_vae
        or not isinstance(parallel_context, ParallelContext)
    ):
        parallel_context = None
    enabled = (
        bool(decision.enabled)
        if parallel_context is None
        else synchronize_stage_bool(
            bool(decision.enabled),
            parallel_context,
            field_name=f"{component_name}_residency",
        )
    )
    if not enabled:
        adapter.record_component_residency_fallback(decision)
        yield False
        return

    acquired = False
    acquire_error: BaseException | None = None
    try:
        adapter.acquire_component_residency(component_name, reason=reason)
        acquired = True
    except BaseException as error:
        acquire_error = error

    synchronized_acquire_error: BaseException | None = acquire_error
    if parallel_context is not None:
        try:
            synchronize_stage_error(acquire_error, parallel_context)
        except BaseException as error:
            synchronized_acquire_error = error

    if synchronized_acquire_error is not None:
        cleanup_error: BaseException | None = None
        if acquired:
            try:
                adapter.release_component_residency(
                    component_name,
                    reason=f"{reason}:acquire_cleanup",
                )
            except BaseException as error:
                cleanup_error = error
        if parallel_context is not None:
            synchronize_stage_error(cleanup_error, parallel_context)
        elif cleanup_error is not None:
            raise cleanup_error
        raise synchronized_acquire_error

    try:
        yield True
    finally:
        release_error: BaseException | None = None
        try:
            adapter.release_component_residency(
                component_name,
                reason=reason,
            )
        except BaseException as error:
            release_error = error
        if parallel_context is not None:
            synchronize_stage_error(release_error, parallel_context)
        elif release_error is not None:
            raise release_error


@contextmanager
def ltx095_memory_phase_scope(
    batch: Req,
    phase: MemoryPhase,
    *,
    component_name: str | None,
    skipped: bool = False,
):
    """Enter a rank-local phase when the window runtime installed a controller."""
    controller = batch.extra.get("memory_phase_controller")
    if controller is None:
        yield False
        return
    window_key = (
        int(batch.extra.get("object_index", -1)),
        int(batch.extra.get("window_index", -1)),
    )
    controller.enter(
        phase,
        component_name=component_name,
        window_key=window_key,
    )
    if skipped:
        controller.record_point("skipped")
    try:
        yield True
    finally:
        controller.exit(phase)


def _is_writer_rank(server_args: ServerArgs) -> bool:
    distributed_context = _distributed_context(server_args)
    if distributed_context is None:
        return True
    return bool(getattr(distributed_context, "is_writer_rank", True))


def _should_skip_writer_only_stage(server_args: ServerArgs, stage_name: str) -> bool:
    if not _official_vae_parallel_active(server_args):
        return False
    if _is_writer_rank(server_args):
        return False
    context = _official_parallel_context(server_args)
    writer_only_stage_names = tuple(getattr(context, "writer_only_stage_names", ()))
    return stage_name in writer_only_stage_names


def is_active_ltx095_sp_peer(server_args: ServerArgs) -> bool:
    active = resolve_active_ltx095_window_commit_context(server_args)
    return bool(active is not None and not active.is_writer)


def should_skip_ltx095_sp_writer_stage(
    batch: Req,
    server_args: ServerArgs,
) -> bool:
    return bool(batch.extra.get("ltx095_sp_writer_owned_runtime")) and (
        is_active_ltx095_sp_peer(server_args)
    )


def resolve_ltx095_vae_binding(
    server_args: ServerArgs,
) -> VAEParallelBinding | None:
    context = getattr(server_args, "parallel_context", None)
    if context is None or not bool(getattr(context, "enabled", False)):
        return None
    plan = getattr(context, "plan", None)
    if plan is None or int(getattr(plan, "vae_degree", 1)) <= 1:
        return None
    if int(getattr(plan, "vae_degree", 0)) != int(
        getattr(plan, "world_size", 0)
    ):
        raise ValueError("active LTX VAE parallelism must cover the world")
    vae_group = context.current_group("vae")
    world_control_group = context.world_control_group()
    owner_rank = int(getattr(plan, "writer_rank"))
    return VAEParallelBinding(
        global_rank=int(getattr(context, "global_rank")),
        owner_rank=owner_rank,
        vae_group=vae_group,
        world_control_group=world_control_group,
        data_coordinator=GroupCoordinator(vae_group),
        control_coordinator=GroupCoordinator(world_control_group),
    )


def resolve_ltx095_vae_control_device(
    server_args: ServerArgs,
    binding: VAEParallelBinding,
) -> torch.device:
    context = getattr(server_args, "parallel_context", None)
    if context is None:
        raise RuntimeError("VAE parallel control device requires parallel context")
    return resolve_group_control_device(
        binding.world_control_group,
        local_rank=int(getattr(context, "local_rank")),
        fallback_device=getattr(server_args, "device", "cpu"),
    )


def resolve_ltx095_vae_data_device(
    server_args: ServerArgs,
    binding: VAEParallelBinding,
) -> torch.device:
    context = getattr(server_args, "parallel_context", None)
    if context is None:
        raise RuntimeError("VAE parallel data device requires parallel context")
    return resolve_group_control_device(
        binding.vae_group,
        local_rank=int(getattr(context, "local_rank")),
        fallback_device=getattr(server_args, "device", "cpu"),
    )


_VAE_OPERATION_TO_CODE = {
    VAEOperation.ENCODE: 1,
    VAEOperation.DECODE: 2,
}
_VAE_CODE_TO_OPERATION = {
    value: key for key, value in _VAE_OPERATION_TO_CODE.items()
}
_VAE_FALLBACK_TO_CODE = {
    None: 0,
    "resolved_degree_one": 1,
    "spatial_threshold": 2,
}
_VAE_CODE_TO_FALLBACK = {
    value: key for key, value in _VAE_FALLBACK_TO_CODE.items()
}


@dataclass(frozen=True)
class LTX095VAEExecution:
    decision: VAEExecutionDecision
    binding: VAEParallelBinding

    @property
    def uses_parallel_tiles(self) -> bool:
        return self.decision.effective_degree > 1


def resolve_ltx095_vae_execution(
    *,
    batch: Req,
    server_args: ServerArgs,
    operation: VAEOperation,
    owner_input: torch.Tensor | None,
    adapter: LTX095VAEParallelAdapter,
) -> LTX095VAEExecution:
    """Broadcast and validate the operation-level LTX VAE execution decision."""
    del batch
    if not isinstance(operation, VAEOperation):
        raise TypeError("operation must be a VAEOperation")
    if not isinstance(adapter, LTX095VAEParallelAdapter):
        raise TypeError("adapter must be LTX095VAEParallelAdapter")
    binding = resolve_ltx095_vae_binding(server_args)
    if binding is None:
        raise RuntimeError("LTX VAE execution requires an active VAE binding")
    resolved_degree = len(binding.vae_group.spec.ranks)
    device = resolve_ltx095_vae_data_device(server_args, binding)
    payload = torch.zeros(6, dtype=torch.int64, device=device)
    if binding.is_owner:
        if owner_input is None:
            raise ValueError("owner_input is required on the owner rank")
        decision = adapter.resolve_execution_decision(
            operation=operation,
            input_shape=tuple(owner_input.shape),
            resolved_degree=resolved_degree,
        )
        payload.copy_(
            torch.tensor(
                (
                    _VAE_OPERATION_TO_CODE[decision.operation],
                    decision.resolved_degree,
                    decision.effective_degree,
                    decision.sample_height,
                    decision.sample_width,
                    _VAE_FALLBACK_TO_CODE[decision.fallback_reason],
                ),
                dtype=torch.int64,
                device=device,
            )
        )
    binding.data_coordinator.broadcast(payload, src=binding.owner_rank)
    values = tuple(int(value) for value in payload.tolist())
    operation_code, received_resolved, effective, sample_height, sample_width, reason_code = values
    received_operation = _VAE_CODE_TO_OPERATION.get(operation_code)
    fallback_reason = _VAE_CODE_TO_FALLBACK.get(reason_code, "__invalid__")
    if received_operation is not operation:
        raise RuntimeError("broadcast VAE operation does not match local operation")
    if received_resolved != resolved_degree:
        raise RuntimeError("broadcast VAE resolved degree does not match VAE group")
    if effective not in (1, resolved_degree):
        raise RuntimeError("broadcast VAE effective degree is invalid")
    if sample_height <= 0 or sample_width <= 0:
        raise RuntimeError("broadcast VAE sample dimensions are invalid")
    if fallback_reason == "__invalid__":
        raise RuntimeError("broadcast VAE fallback reason is invalid")
    if (effective == 1) != (fallback_reason is not None):
        raise RuntimeError("broadcast VAE fallback reason is inconsistent")
    decision = VAEExecutionDecision(
        operation=received_operation,
        resolved_degree=received_resolved,
        effective_degree=effective,
        sample_height=sample_height,
        sample_width=sample_width,
        fallback_reason=fallback_reason,
    )
    expected = adapter.resolve_execution_decision(
        operation=operation,
        input_shape=(
            tuple(owner_input.shape)
            if owner_input is not None
            else (
                1,
                1,
                1,
                sample_height // adapter.spatial_ratio
                if operation is VAEOperation.DECODE
                else sample_height,
                sample_width // adapter.spatial_ratio
                if operation is VAEOperation.DECODE
                else sample_width,
            )
        ),
        resolved_degree=resolved_degree,
    )
    if decision != expected:
        raise RuntimeError("broadcast VAE execution decision failed validation")
    return LTX095VAEExecution(decision=decision, binding=binding)


def record_ltx095_vae_execution_decision(
    batch: Req,
    decision: VAEExecutionDecision,
) -> dict[str, Any]:
    history = batch.extra.get("ltx095_vae_execution_history")
    if not isinstance(history, list):
        history = []
        batch.extra["ltx095_vae_execution_history"] = history
    item = {
        "operation": decision.operation.value,
        "resolved_degree": decision.resolved_degree,
        "effective_degree": decision.effective_degree,
        "sample_height": decision.sample_height,
        "sample_width": decision.sample_width,
        "fallback_reason": decision.fallback_reason,
    }
    history.append(item)
    return item


def record_ltx095_vae_parallel_metrics(
    batch: Req,
    metrics: VAEParallelMetrics,
) -> dict[str, Any]:
    if not isinstance(metrics, VAEParallelMetrics):
        raise TypeError("metrics must be VAEParallelMetrics")
    history = batch.extra.get("ltx095_vae_parallel_history")
    if not isinstance(history, list):
        history = []
        batch.extra["ltx095_vae_parallel_history"] = history
    item = asdict(metrics)
    history.append(item)
    return item


def _ltx095_sp_writer_stage_history(batch: Req) -> list[dict[str, Any]]:
    history = batch.extra.get("ltx095_sp_writer_stage_history")
    if not isinstance(history, list):
        history = []
        batch.extra["ltx095_sp_writer_stage_history"] = history
    return history


def record_ltx095_sp_writer_stage_skip(batch: Req, *, stage: str) -> None:
    _ltx095_sp_writer_stage_history(batch).append(
        {
            "event": "stage_skip_sp_peer",
            "stage": stage,
            "reason": "writer_owned_window_preparation",
        }
    )


def record_ltx095_sp_writer_stage_operation(
    batch: Req,
    server_args: ServerArgs,
    *,
    operation: str,
) -> None:
    if not bool(batch.extra.get("ltx095_sp_writer_owned_runtime")):
        return
    active = resolve_active_ltx095_window_commit_context(server_args)
    if active is None or not active.is_writer or batch.metrics is None:
        return
    batch.metrics.record_operation(operation)


def _official_parallel_event_history(batch: Req) -> list[dict[str, Any]]:
    history = batch.extra.get("runtime_official_parallel_history")
    if not isinstance(history, list):
        history = []
        batch.extra["runtime_official_parallel_history"] = history
    return history


def _record_official_parallel_event(batch: Req, event: str, **payload: Any) -> dict[str, Any]:
    item = {"event": str(event), **payload}
    _official_parallel_event_history(batch).append(item)
    return item


def _resource_event_history(batch: Req) -> list[dict[str, Any]]:
    history = batch.extra.get("runtime_resource_event_history")
    if not isinstance(history, list):
        history = []
        batch.extra["runtime_resource_event_history"] = history
    return history


def _record_resource_event(batch: Req, event: str, **payload: Any) -> dict[str, Any]:
    item = {"event": str(event), **payload}
    _resource_event_history(batch).append(item)
    return item


def _select_resource_policy(batch: Req, server_args: ServerArgs) -> Any:
    policy = server_args.resolve_resource_policy()
    batch.extra.setdefault("runtime_resource_policy", policy.as_dict())
    if not batch.extra.get("_runtime_resource_policy_selected_logged", False):
        _record_resource_event(batch, "resource_policy_selected", **policy.as_dict())
        for reason in policy.fallback_reasons:
            _record_resource_event(
                batch,
                "resource_policy_fallback",
                requested_policy=policy.requested_policy,
                selected_policy=policy.selected_policy,
                reason=reason,
            )
        batch.extra["_runtime_resource_policy_selected_logged"] = True
    return policy


def _maybe_register_pin_memory(
    batch: Req,
    name: str,
    kind: str,
    changed: bool,
) -> None:
    if changed:
        _record_resource_event(
            batch,
            "pin_memory_register",
            name=name,
            kind=kind,
        )


def _onload_module(
    batch: Req,
    server_args: ServerArgs,
    module_name: str,
) -> tuple[Any, Any]:
    policy = _select_resource_policy(batch, server_args)
    module = batch.modules[module_name]
    if not isinstance(module, nn.Module):
        return module, policy
    if policy.dynamic_offload:
        target_device = torch.device(server_args.device)
        has_flexible_extent = FlexibleModuleExtentBase.check_flexible(
            module,
            contiain_sub=True,
        )
        from memory.backends.offload_tools import cuda_module as oth_cuda
        oth_cuda(
            module=module,
            device=target_device,
            contain_sub=True,
            skip_flexible=True,
            check_device=False,
        )
        if has_flexible_extent:
            module._mgerase_execution_device = target_device
            _record_resource_event(
                batch,
                "extent_lifecycle_enter",
                module_name=module_name,
                target_device=str(target_device),
                policy=policy.selected_policy,
            )
        else:
            _record_resource_event(
                batch,
                "module_onload",
                module_name=module_name,
                target_device=str(target_device),
                dtype=str(server_args.resolve_component_dtype(module_name)),
                policy=policy.selected_policy,
            )
    return module, policy


def _offload_module(
    batch: Req,
    server_args: ServerArgs,
    module_name: str,
    policy: Any,
    *,
    reason: str,
) -> None:
    module = batch.modules[module_name]
    if not isinstance(module, nn.Module):
        return
    if policy.dynamic_offload:
        has_flexible_extent = FlexibleModuleExtentBase.check_flexible(
            module,
            contiain_sub=True,
        )
        from memory.backends.offload_tools import offload_module as oth_offload
        oth_offload(
            module=module,
            contain_sub=True,
            skip_flexible=True,
            check_device=False,
        )
        if has_flexible_extent:
            _record_resource_event(
                batch,
                "extent_lifecycle_exit",
                module_name=module_name,
                reason=reason,
                policy=policy.selected_policy,
            )
        else:
            _record_resource_event(
                batch,
                "module_offload",
                module_name=module_name,
                target_device="cpu",
                reason=reason,
                policy=policy.selected_policy,
            )


def _linear_quadratic_schedule(
    num_steps: int,
    threshold_noise: float = 0.025,
    linear_steps: int | None = None,
) -> torch.Tensor:
    if linear_steps is None:
        linear_steps = num_steps // 2
    if num_steps < 2:
        return torch.tensor([1.0], dtype=torch.float32)
    linear_sigma_schedule = [
        i * threshold_noise / linear_steps for i in range(linear_steps)
    ]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (
        quadratic_steps**2
    )
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const
        for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.tensor(sigma_schedule[:-1], dtype=torch.float32)


def _retrieve_timesteps(scheduler, num_inference_steps: int, device: torch.device) -> torch.Tensor:
    timesteps = _linear_quadratic_schedule(num_inference_steps) * 1000
    scheduler.set_timesteps(timesteps=timesteps.tolist(), device=device)
    return scheduler.timesteps


def _trim_timesteps_for_strength(
    scheduler,
    timesteps: torch.Tensor,
    num_inference_steps: int,
    strength: float,
) -> tuple[torch.Tensor, int]:
    init_timestep = min(num_inference_steps * strength, num_inference_steps)
    t_start = int(max(num_inference_steps - init_timestep, 0))
    trimmed = timesteps[t_start * scheduler.order :]
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(t_start * scheduler.order)
    return trimmed, num_inference_steps - t_start
