"""Rank-local model memory registration for the LTX095 pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Mapping

import torch
from torch import nn

from memory.backends.federated_storage import FederatedStorage
from memory.backends.flexible_memory_states import FlexibleMemoryState
from memory.backends.flexible_module_extent_base import FlexibleModuleExtentBase
from memory.backends.flexible_module_extent_cuda_async import (
    FlexibleModuleExtentCudaAsync,
)
from memory.backends.offload_tools import (
    cuda_module,
    malloc_pin_memory,
    offload_module,
    release_module_pin_memory,
)


@dataclass(frozen=True)
class MemoryRegistrationSummary:
    rank: int
    device: str
    dynamic_offload: bool
    pin_memory: bool
    max_weight_usage: int
    registered_extent_count: int
    registered_extent_bytes: int
    pinned_non_extent_bytes: int
    largest_extent_bytes: int
    candidate_module_names: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["candidate_module_names"] = list(self.candidate_module_names)
        return result


@dataclass(frozen=True)
class ComponentResidencyDecision:
    component_name: str
    enabled: bool
    hotset_bytes: int
    max_weight_usage: int
    fallback_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _candidate_modules(modules: Mapping[str, object]) -> list[tuple[str, nn.Module]]:
    result: list[tuple[str, nn.Module]] = []
    text_encoder = modules.get("text_encoder")
    transformer = modules.get("transformer")
    vae = modules.get("vae")
    if isinstance(text_encoder, nn.Module):
        result.append(("text_encoder", text_encoder))
    if isinstance(transformer, nn.Module):
        result.append(("transformer", transformer))
    if isinstance(vae, nn.Module):
        encoder = getattr(vae, "encoder", None)
        decoder = getattr(vae, "decoder", None)
        if isinstance(encoder, nn.Module):
            result.append(("vae.encoder", encoder))
        if isinstance(decoder, nn.Module):
            result.append(("vae.decoder", decoder))
    return result


def _named_tensors(module: nn.Module) -> dict[str, torch.Tensor]:
    result = dict(module.named_parameters())
    result.update(dict(module.named_buffers()))
    return result


def _unique_module_bytes(module: nn.Module, *, skip_flexible: bool) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0

    def visit(current: nn.Module) -> None:
        nonlocal total
        if (
            skip_flexible
            and hasattr(current, "is_flexible")
            and current.is_flexible()
        ):
            return
        for tensor in (
            list(current.parameters(recurse=False))
            + list(current.buffers(recurse=False))
        ):
            storage = tensor.untyped_storage()
            identity = (storage.data_ptr(), storage.nbytes())
            if identity not in seen:
                seen.add(identity)
                total += int(storage.nbytes())
        for child in current.children():
            visit(child)

    visit(module)
    return total


class ModelMemoryAdapter:
    """Own flexible extents and component-level pinned mirrors for one rank."""

    def __init__(
        self,
        *,
        extent_cls=FlexibleModuleExtentCudaAsync,
        pin_module_fn: Callable[..., object] = malloc_pin_memory,
        offload_module_fn: Callable[..., object] = offload_module,
        onload_module_fn: Callable[..., object] = cuda_module,
        release_pin_fn: Callable[[nn.Module], None] = release_module_pin_memory,
        configure_device_fn: Callable[[torch.device, int], object] | None = None,
        reset_device_fn: Callable[[torch.device], None] | None = None,
        module_size_fn: Callable[[nn.Module], int] | None = None,
    ) -> None:
        self.extent_cls = extent_cls
        self.pin_module_fn = pin_module_fn
        self.offload_module_fn = offload_module_fn
        self.onload_module_fn = onload_module_fn
        self.release_pin_fn = release_pin_fn
        self.configure_device_fn = configure_device_fn or (
            lambda device, budget: FlexibleMemoryState.configure_device(
                device,
                max_weight_usage=budget,
            )
        )
        self.reset_device_fn = reset_device_fn or (
            lambda device: FlexibleMemoryState.reset_device(
                device,
                release_worker=True,
            )
        )
        self.module_size_fn = module_size_fn
        self._candidate_modules: list[nn.Module] = []
        self._execution_roots: list[nn.Module] = []
        self._extents: list[object] = []
        self._device: torch.device | None = None
        self._summary: MemoryRegistrationSummary | None = None
        self._component_extents: dict[str, tuple[object, ...]] = {}
        self._component_modules: dict[str, nn.Module] = {}
        self._component_hotset_bytes: dict[str, int] = {}
        self._component_acquire_count: dict[str, int] = {}
        self._component_release_count: dict[str, int] = {}
        self._component_fallback_reasons: dict[str, dict[str, int]] = {}
        self._active_component_name: str | None = None
        self._last_residency_reason: str | None = None
        self._dynamic_offload = False
        self._max_weight_usage = 0
        self._close_had_active_lease = False
        self._shutdown_mode = "open"
        self._closed_snapshot: dict[str, object] | None = None
        self._closed = False

    def register(
        self,
        *,
        modules: Mapping[str, object],
        device: torch.device,
        dynamic_offload: bool,
        pin_memory: bool,
        max_weight_usage: int,
        rank: int,
    ) -> MemoryRegistrationSummary:
        candidates = _candidate_modules(modules)
        names = tuple(name for name, _ in candidates)
        if int(max_weight_usage) <= 0:
            raise ValueError("max_weight_usage must be positive")
        for name, module in candidates:
            if module.training:
                raise ValueError(
                    f"memory candidate {name} must be in eval mode before registration"
                )

        FederatedStorage.validate_no_cross_extent_aliases(
            {
                name: {
                    f"{name}.{tensor_name}": tensor
                    for tensor_name, tensor in _named_tensors(module).items()
                }
                for name, module in candidates
            }
        )

        self._candidate_modules = [module for _, module in candidates]
        self._device = torch.device(device)
        self._dynamic_offload = bool(dynamic_offload)
        self._max_weight_usage = int(max_weight_usage)
        registered_extent_bytes = 0
        pinned_non_extent_bytes = 0
        largest_extent_bytes = 0

        if dynamic_offload:
            if self._device.type != "cuda":
                raise ValueError("dynamic_offload requires a CUDA device")
            self.configure_device_fn(self._device, int(max_weight_usage))
            seen_extent_ids: set[int] = set()
            for name, module in candidates:
                self._component_modules[name] = module
                before_ids = {
                    id(getattr(submodule, "flexible_extent"))
                    for submodule in module.modules()
                    if hasattr(submodule, "flexible_extent")
                }
                self.extent_cls.register_module_extent_by_size(
                    module,
                    self._device,
                    min_size=15 * 1024**2,
                    max_size=500 * 1024**2,
                    init_offload=True,
                    module_size_fn=self.module_size_fn,
                )
                component_extents: list[object] = []
                for submodule in module.modules():
                    extent = getattr(submodule, "flexible_extent", None)
                    if (
                        extent is None
                        or id(extent) in before_ids
                        or id(extent) in seen_extent_ids
                    ):
                        continue
                    seen_extent_ids.add(id(extent))
                    component_extents.append(extent)
                self._component_extents[name] = tuple(component_extents)
                self._component_hotset_bytes[name] = sum(
                    int(extent.estimated_memory_requirement())
                    for extent in component_extents
                )
                self._extents.extend(component_extents)
                module._mgerase_execution_device = self._device
            vae = modules.get("vae")
            if isinstance(vae, nn.Module):
                vae._mgerase_execution_device = self._device
                self._execution_roots.append(vae)
            extent_sizes = [
                int(extent.estimated_memory_requirement())
                for extent in self._extents
                if hasattr(extent, "estimated_memory_requirement")
            ]
            registered_extent_bytes = sum(extent_sizes)
            largest_extent_bytes = max(extent_sizes, default=0)

            for _, module in candidates:
                if pin_memory:
                    self.pin_module_fn(
                        module,
                        ref_device=self._device,
                        skip_flexible=True,
                        contigous=True,
                    )
                    pinned_non_extent_bytes += _unique_module_bytes(
                        module,
                        skip_flexible=True,
                    )
                self.offload_module_fn(
                    module,
                    contain_sub=True,
                    skip_flexible=True,
                    check_device=False,
                )
        elif pin_memory:
            for _, module in candidates:
                self.pin_module_fn(
                    module,
                    ref_device=self._device,
                    skip_flexible=False,
                    contigous=True,
                )
                pinned_non_extent_bytes += _unique_module_bytes(
                    module,
                    skip_flexible=False,
                )

        self._summary = MemoryRegistrationSummary(
            rank=int(rank),
            device=str(self._device),
            dynamic_offload=bool(dynamic_offload),
            pin_memory=bool(pin_memory),
            max_weight_usage=int(max_weight_usage),
            registered_extent_count=len(self._extents),
            registered_extent_bytes=registered_extent_bytes,
            pinned_non_extent_bytes=pinned_non_extent_bytes,
            largest_extent_bytes=largest_extent_bytes,
            candidate_module_names=names,
        )
        return self._summary

    def plan_component_residency(
        self,
        component_name: str,
    ) -> ComponentResidencyDecision:
        if not self._dynamic_offload:
            reason = "dynamic_offload_disabled"
        elif component_name not in self._component_extents:
            reason = "component_unregistered"
        elif not self._component_extents[component_name]:
            reason = "component_has_no_extents"
        elif (
            self._component_hotset_bytes[component_name]
            > self._max_weight_usage
        ):
            reason = "budget_insufficient"
        else:
            reason = None
        return ComponentResidencyDecision(
            component_name=component_name,
            enabled=reason is None,
            hotset_bytes=self._component_hotset_bytes.get(
                component_name,
                0,
            ),
            max_weight_usage=self._max_weight_usage,
            fallback_reason=reason,
        )

    def record_component_residency_fallback(
        self,
        decision: ComponentResidencyDecision,
    ) -> None:
        reason = decision.fallback_reason
        if decision.enabled or reason is None:
            raise ValueError("enabled residency decision is not a fallback")
        reasons = self._component_fallback_reasons.setdefault(
            decision.component_name,
            {},
        )
        reasons[reason] = reasons.get(reason, 0) + 1
        self._last_residency_reason = reason

    def acquire_component_residency(
        self,
        component_name: str,
        *,
        reason: str,
    ) -> None:
        decision = self.plan_component_residency(component_name)
        if not decision.enabled:
            raise RuntimeError(
                f"component residency is unavailable: {decision.as_dict()}"
            )
        if self._active_component_name is not None:
            raise RuntimeError(
                "component residency overlap is forbidden: "
                f"active={self._active_component_name}, "
                f"requested={component_name}"
            )

        self._active_component_name = component_name
        self._last_residency_reason = reason
        acquired: list[object] = []
        try:
            for extent in self._component_extents[component_name]:
                extent.acquire_residency(label=reason)
                acquired.append(extent)
            if self._device is None:
                raise RuntimeError("component device is unavailable")
            self.onload_module_fn(
                self._component_modules[component_name],
                self._device,
                contain_sub=True,
                skip_flexible=True,
                check_device=False,
            )
        except BaseException as acquire_error:
            rollback_errors: list[BaseException] = []
            for extent in reversed(acquired):
                try:
                    extent.release_residency(label=f"{reason}:rollback")
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
            if not rollback_errors:
                self._active_component_name = None
            if rollback_errors:
                raise RuntimeError(
                    "component residency acquire rollback failed"
                ) from acquire_error
            raise

        self._component_acquire_count[component_name] = (
            self._component_acquire_count.get(component_name, 0) + 1
        )

    def release_component_residency(
        self,
        component_name: str,
        *,
        reason: str,
    ) -> None:
        if self._active_component_name != component_name:
            raise RuntimeError(
                "component residency release does not match active lease: "
                f"active={self._active_component_name}, "
                f"requested={component_name}"
            )

        self._last_residency_reason = reason
        release_errors: list[BaseException] = []
        for extent in reversed(self._component_extents[component_name]):
            if not bool(getattr(extent, "is_resident", False)):
                continue
            try:
                extent.release_residency(label=reason)
            except BaseException as error:
                release_errors.append(error)
        try:
            self.offload_module_fn(
                self._component_modules[component_name],
                contain_sub=True,
                skip_flexible=True,
                check_device=False,
            )
        except BaseException as error:
            release_errors.append(error)
        if release_errors:
            raise release_errors[0]

        self._active_component_name = None
        self._component_release_count[component_name] = (
            self._component_release_count.get(component_name, 0) + 1
        )

    @property
    def active_component_name(self) -> str | None:
        return self._active_component_name

    def settle_component_transfers(self) -> None:
        if (
            self._device is not None
            and self._device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.current_stream(self._device).synchronize()

    def _component_residency_snapshot(self) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for component_name, extents in self._component_extents.items():
            extent_snapshots = [
                extent.residency_snapshot()
                for extent in extents
            ]
            fallback_reasons = dict(
                self._component_fallback_reasons.get(component_name, {})
            )
            result[component_name] = {
                "hotset_bytes": self._component_hotset_bytes.get(
                    component_name,
                    0,
                ),
                "extent_count": len(extents),
                "component_acquire_count": self._component_acquire_count.get(
                    component_name,
                    0,
                ),
                "component_release_count": self._component_release_count.get(
                    component_name,
                    0,
                ),
                "extent_onload_count": sum(
                    int(snapshot["resident_onload_count"])
                    for snapshot in extent_snapshots
                ),
                "extent_onload_bytes": sum(
                    int(snapshot["resident_onload_bytes"])
                    for snapshot in extent_snapshots
                ),
                "extent_release_count": sum(
                    int(snapshot["resident_release_count"])
                    for snapshot in extent_snapshots
                ),
                "extent_release_bytes": sum(
                    int(snapshot["resident_release_bytes"])
                    for snapshot in extent_snapshots
                ),
                "resident_fast_path_forward_count": sum(
                    int(snapshot["resident_fast_path_forward_count"])
                    for snapshot in extent_snapshots
                ),
                "fallback_count": sum(fallback_reasons.values()),
                "fallback_reasons": fallback_reasons,
                "active": self._active_component_name == component_name,
                "last_reason": self._last_residency_reason,
            }
        return result

    def _live_snapshot(self) -> dict[str, object]:
        result = self._summary.as_dict() if self._summary is not None else {}
        if (
            self._device is not None
            and self._device.type == "cuda"
            and self._extents
        ):
            result["flexible_state"] = FlexibleMemoryState.snapshot(
                self._device
            )
        else:
            result["flexible_state"] = {}
        result["component_residency"] = (
            self._component_residency_snapshot()
        )
        result["closed"] = self._closed
        result["shutdown_mode"] = self._shutdown_mode
        result["close_had_active_lease"] = self._close_had_active_lease
        result["active_component_name"] = self._active_component_name
        result["resident_bytes"] = (
            self._component_hotset_bytes.get(self._active_component_name, 0)
            if self._active_component_name is not None
            else 0
        )
        result["terminal_restore_bytes"] = 0
        return result

    def snapshot(self) -> dict[str, object]:
        if self._closed_snapshot is not None:
            return self._closed_snapshot
        return self._live_snapshot()

    def shutdown(self, *, terminal: bool = False) -> dict[str, object]:
        if self._closed:
            return self.snapshot()
        self._close_had_active_lease = self._active_component_name is not None
        if self._active_component_name is not None:
            self.release_component_residency(
                self._active_component_name,
                reason="adapter_shutdown",
            )
        for extent in reversed(self._extents):
            if hasattr(extent, "close"):
                extent.close(
                    restore_forward=not terminal,
                    restore_module_storage=not terminal,
                )
        for module in self._candidate_modules:
            self.release_pin_fn(module)
            if hasattr(module, "_mgerase_execution_device"):
                delattr(module, "_mgerase_execution_device")
        for module in self._execution_roots:
            if hasattr(module, "_mgerase_execution_device"):
                delattr(module, "_mgerase_execution_device")
        self._closed = True
        self._shutdown_mode = "terminal" if terminal else "restored"
        self._closed_snapshot = self._live_snapshot()
        if (
            self._device is not None
            and self._device.type == "cuda"
            and self._extents
        ):
            self.reset_device_fn(self._device)
        return self._closed_snapshot

    def close(self) -> None:
        self.shutdown(terminal=False)

    def close_and_snapshot(self) -> dict[str, object]:
        return self.shutdown(terminal=False)
