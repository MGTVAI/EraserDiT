"""Synchronous module extents for bounded weight residency."""

from __future__ import annotations

import weakref
from typing import Any, Callable

import torch
from torch import nn

from memory.backends import offload_tools
from memory.backends.federated_storage import FederatedStorage
from memory.backends.flexible_memory_states import FlexibleMemoryState
from memory.backends.flexible_module_extent_base import (
    FlexibleMemoryTypeEnum,
    FlexibleModuleExtentBase,
    FlexibleSupportStagesEnum,
)


class FlexibleModuleExtent(FlexibleModuleExtentBase):
    def __init__(
        self,
        module: nn.Module,
        calc_device: torch.device,
        *,
        use_weak_ref: bool = True,
        init_offload: bool = True,
        memory_contigous: bool = True,
        storage_type: FlexibleMemoryTypeEnum = (
            FlexibleMemoryTypeEnum.FederatedPinMemory
        ),
    ) -> None:
        super().__init__()
        self._module_ref = weakref.ref(module) if use_weak_ref else module
        self._use_weak_ref = use_weak_ref
        self._original_forward = module.forward
        self._calc_device = torch.device(calc_device)
        self._storage_type = storage_type
        self._closed = False
        self._estimated_memory_requirement = self.get_module_memory_require(
            module,
            contain_grad=False,
            contain_sub=True,
        )
        self._federated_storage: FederatedStorage | None = None

        if self._calc_device.type == "cuda":
            FlexibleMemoryState.add_flexible_register_bytes(
                self._estimated_memory_requirement,
                device=self._calc_device,
            )

        if storage_type == FlexibleMemoryTypeEnum.FederatedPinMemory:
            self._federated_storage = FederatedStorage.register_federate_module(
                module,
                target_device=None,
                skip_flexible=False,
                use_pin_memory=True,
                use_contigous_memory=memory_contigous,
            )
        elif storage_type == FlexibleMemoryTypeEnum.PinMemory:
            offload_tools.malloc_pin_memory(
                module,
                skip_flexible=False,
                contigous=memory_contigous,
            )

        if init_offload:
            self._weight_offload(check_device=False)

    @property
    def module_ref(self) -> nn.Module:
        module = self._module_ref() if self._use_weak_ref else self._module_ref
        if not isinstance(module, nn.Module):
            raise RuntimeError("flexible extent module has already been released")
        return module

    @property
    def calc_device(self) -> torch.device:
        return self._calc_device

    def estimated_memory_requirement(self) -> int:
        return max(0, int(self._estimated_memory_requirement))

    def support_train(self) -> FlexibleSupportStagesEnum:
        return FlexibleSupportStagesEnum.OnlyFunctionWithoutMemOpt

    def support_inference(self) -> FlexibleSupportStagesEnum:
        return FlexibleSupportStagesEnum.OnlyFunctionWithoutMemOpt

    def _weight_onload(self, *, check_device: bool) -> None:
        if self._storage_type == FlexibleMemoryTypeEnum.FederatedPinMemory:
            FederatedStorage.cuda_federate_module(
                self.module_ref,
                device=self._calc_device,
                check_device=check_device,
            )
        else:
            offload_tools.cuda_module(
                self.module_ref,
                self._calc_device,
                contain_sub=True,
                skip_flexible=False,
                check_device=check_device,
            )

    def _weight_offload(self, *, check_device: bool) -> None:
        if self._storage_type == FlexibleMemoryTypeEnum.FederatedPinMemory:
            FederatedStorage.offload_federate_module(
                self.module_ref,
                check_device=check_device,
            )
        else:
            offload_tools.offload_module(
                self.module_ref,
                contain_sub=True,
                skip_flexible=False,
                check_device=check_device,
            )

    def weight_onload(self, label: str = "") -> None:
        self._weight_onload(check_device=False)

    @torch.compiler.disable(recursive=False)
    def inference(self, label: str = "", *args, **kwargs) -> Any:
        return self._original_forward(*args, **kwargs)

    def weight_offload(self, label: str = "") -> None:
        self._weight_offload(check_device=False)

    def close(
        self,
        *,
        restore_forward: bool = False,
        restore_module_storage: bool = True,
    ) -> None:
        if self._closed:
            return
        module = self.module_ref
        if self._federated_storage is not None:
            self._federated_storage.release(
                restore_module_storage=restore_module_storage
            )
        elif self._storage_type == FlexibleMemoryTypeEnum.PinMemory:
            offload_tools.release_module_pin_memory(module)
        if self._calc_device.type == "cuda":
            FlexibleMemoryState.remove_flexible_register_bytes(
                self._estimated_memory_requirement,
                device=self._calc_device,
            )
        if restore_forward:
            module.forward = self._original_forward
        self._closed = True

    @classmethod
    def register_module_flexible_extent(
        cls,
        module: nn.Module,
        calc_device: torch.device,
        *,
        use_weak_ref: bool = True,
        init_offload: bool = True,
        memory_contigous: bool = True,
        storage_type: FlexibleMemoryTypeEnum = (
            FlexibleMemoryTypeEnum.FederatedPinMemory
        ),
    ) -> tuple[nn.Module, bool]:
        existing = getattr(module, "flexible_extent", None)
        if isinstance(existing, cls):
            return module, True
        cls.clean_module_flexible_extent(module)
        extent = cls(
            module,
            calc_device,
            use_weak_ref=use_weak_ref,
            init_offload=init_offload,
            memory_contigous=memory_contigous,
            storage_type=storage_type,
        )
        module.flexible_extent = extent
        module.is_flexible = extent.is_flexible
        module.forward = extent.forward
        return module, True

    @classmethod
    def register_module_extent_by_size(
        cls,
        module: nn.Module,
        calc_device: torch.device,
        *,
        min_size: int = 15 * 1024**2,
        max_size: int = 500 * 1024**2,
        use_weak_ref: bool = True,
        init_offload: bool = True,
        memory_contigous: bool = True,
        storage_type: FlexibleMemoryTypeEnum = (
            FlexibleMemoryTypeEnum.FederatedPinMemory
        ),
        module_size_fn: Callable[[nn.Module], int] | None = None,
    ) -> tuple[nn.Module, bool]:
        if cls.check_flexible(module, contiain_sub=False):
            return module, True
        size_fn = module_size_fn or (
            lambda value: cls.get_module_memory_require(
                value,
                contain_grad=False,
                contain_sub=True,
            )
        )
        required = int(size_fn(module))
        if not isinstance(module, nn.ModuleList) and min_size <= required <= max_size:
            return cls.register_module_flexible_extent(
                module,
                calc_device,
                use_weak_ref=use_weak_ref,
                init_offload=init_offload,
                memory_contigous=memory_contigous,
                storage_type=storage_type,
            )
        if required < min_size and not isinstance(module, nn.ModuleList):
            return module, False

        registered = False
        for child in module.children():
            _, child_registered = cls.register_module_extent_by_size(
                child,
                calc_device,
                min_size=min_size,
                max_size=max_size,
                use_weak_ref=use_weak_ref,
                init_offload=init_offload,
                memory_contigous=memory_contigous,
                storage_type=storage_type,
                module_size_fn=size_fn,
            )
            registered = registered or child_registered
        return module, registered

    @classmethod
    def clean_module_flexible_extent(cls, module: nn.Module) -> nn.Module:
        extents: list[FlexibleModuleExtent] = []
        for submodule in module.modules():
            extent = getattr(submodule, "flexible_extent", None)
            if isinstance(extent, cls):
                extents.append(extent)
        for extent in extents:
            owner = extent.module_ref
            extent.close(restore_forward=True)
            if hasattr(owner, "flexible_extent"):
                delattr(owner, "flexible_extent")
            if hasattr(owner, "is_flexible"):
                delattr(owner, "is_flexible")
        return module
