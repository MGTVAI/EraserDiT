#
# Copyright 2025 shanhai team of MGTV. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
from torch import nn

from memory.backends.offload_tools import cuda_module, offload_module


def tensor_storage_identity(tensor: torch.Tensor) -> tuple[int, int, int, torch.dtype]:
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tensor.numel(),
        tensor.dtype,
    )


@dataclass(frozen=True)
class StorageInfo:
    start: int
    length: int
    shape: torch.Size
    dtype: torch.dtype
    is_parameter: bool
    requires_grad: bool
    identity: tuple[int, int, int, torch.dtype]

    STORAGE_DTYPE = torch.uint8

    @classmethod
    def from_tensor(
        cls,
        raw_tensor: torch.Tensor,
        *,
        start: int,
        length: int,
        identity: tuple[int, int, int, torch.dtype],
    ) -> "StorageInfo":
        return cls(
            start=start,
            length=length,
            shape=raw_tensor.shape,
            dtype=raw_tensor.dtype,
            is_parameter=isinstance(raw_tensor, nn.Parameter),
            requires_grad=raw_tensor.requires_grad,
            identity=identity,
        )

    def recover_tensor(self, storage_tensor: torch.Tensor) -> torch.Tensor:
        """
        no data copy, return slice of storage
        """
        if storage_tensor is None:
            return None

        assert storage_tensor.dtype == StorageInfo.STORAGE_DTYPE
        tensor = (
            storage_tensor[self.start : self.start + self.length]
            .view(self.dtype)
            .view(self.shape)
        )

        if self.is_parameter:
            return nn.Parameter(tensor, requires_grad=self.requires_grad)
        tensor.requires_grad_(self.requires_grad)
        return tensor

    @staticmethod
    def standardizin_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """
        no data copy, only view change
        """
        return tensor.view((-1,)).view(StorageInfo.STORAGE_DTYPE)


class FederatedStorage:
    def __init__(
        self,
        tensors: Mapping[str, torch.Tensor],
        device: torch.device = None,
        use_pin_memory=True,
        use_contigous_memory=True,
        module: nn.Module | None = None,
    ):
        """
        read-only once init finish
        """
        self.member_info: Dict[str, StorageInfo] = {}
        self._module_ref = weakref.ref(module) if module is not None else None
        self._released = False

        self.use_pin_memory = use_pin_memory
        self.use_contigous_memory = use_contigous_memory

        self.storage: torch.Tensor | None = None
        self._cpu_storage: torch.Tensor | None = None
        self.__pin_storage__: torch.Tensor | None = None

        self.__federated_tensors__(tensors=tensors, device=device)

    def __federated_tensors__(
        self, tensors: Dict[str, torch.Tensor], device: torch.device = None
    ):
        """
        federated tensors, call repeate will override previous op

        :param tensors: tensor need to federate, must in same device, or give aim device
        :type tensors: Dict[str, torch.Tensor]
        :param device: aim device
        :type device: torch.device
        """
        self.member_info = {}
        count = 0
        standard_tensors = []
        unique_info: dict[
            tuple[int, int, int, torch.dtype],
            StorageInfo,
        ] = {}
        if len(tensors) < 1:
            self.__pin_storage__ = None
            self.storage = None
            return

        for name, tensor in tensors.items():
            if device is not None:
                tensor = tensor.to(device)

            identity = tensor_storage_identity(tensor)
            existing = unique_info.get(identity)
            if existing is not None:
                self.member_info[name] = existing
                continue
            standard_tensor = StorageInfo.standardizin_tensor(tensor)
            info = StorageInfo.from_tensor(
                tensor,
                start=count,
                length=standard_tensor.size(0),
                identity=identity,
            )
            unique_info[identity] = info
            self.member_info[name] = info
            standard_tensors.append(standard_tensor)
            count += standard_tensor.size(0)
        host_storage = torch.empty(
            count,
            dtype=StorageInfo.STORAGE_DTYPE,
            device=torch.device("cpu"),
            pin_memory=self.use_pin_memory,
        )
        offset = 0
        for standard_tensor in standard_tensors:
            length = standard_tensor.numel()
            host_storage[offset : offset + length].copy_(
                standard_tensor,
                non_blocking=False,
            )
            offset += length

        self._cpu_storage = host_storage
        self.__pin_storage__ = host_storage if self.use_pin_memory else None
        self.storage = host_storage

    @property
    def pin_storage(self) -> torch.Tensor | None:
        return self.__pin_storage__

    @property
    def nbytes(self) -> int:
        if self.storage is None:
            return 0
        return int(self.storage.numel() * self.storage.element_size())

    def htod(self, device: torch.device = None, non_blocking: bool = True):
        if self.storage is None:
            return

        if self.__pin_storage__ is not None:
            self.storage = self.__pin_storage__.to(
                device=device, non_blocking=non_blocking
            )
        else:
            self.storage = self.storage.to(device=device, non_blocking=non_blocking)

    def dtoh(self):
        if self.storage is None:
            return

        if self.__pin_storage__ is not None:
            self.storage = self.__pin_storage__
        else:
            self.storage = self.storage.cpu()

    def get_tensors(self) -> Dict[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {}
        recovered: dict[
            tuple[int, int, int, torch.dtype],
            torch.Tensor,
        ] = {}
        for name, storage_info in self.member_info.items():
            tensor = recovered.get(storage_info.identity)
            if tensor is None:
                tensor = storage_info.recover_tensor(self.storage)
                recovered[storage_info.identity] = tensor
            tensors[name] = tensor
        return tensors

    def __len__(self):
        return len(self.member_info)

    def __getitem__(self, key: str) -> torch.Tensor:
        identity = self.member_info[key].identity
        for name, tensor in self.get_tensors().items():
            if self.member_info[name].identity == identity:
                return tensor
        raise KeyError(key)

    def __setitem__(self, key, value):
        raise SyntaxError("not allow")

    def __delitem__(self, key):
        raise SyntaxError("not allow")

    @staticmethod
    def register_federate_module(
        module: torch.nn.Module,
        target_device=None,
        skip_flexible=True,
        use_pin_memory=True,
        use_contigous_memory=True,
    ) -> "FederatedStorage | None":
        if hasattr(module, "is_flexible") and module.is_flexible() and skip_flexible:
            # ignore Flexible Module
            return None
        existing = getattr(module, "__weights_federated__", None)
        if isinstance(existing, FederatedStorage) and not existing._released:
            return existing

        tensors = {}
        for sub_module_name, sub_module in module.named_modules():
            for name, param in sub_module.named_parameters(recurse=False, remove_duplicate=False):
                tensors[f"p_{sub_module_name}.{name}"] = param

            for name, buffer in sub_module.named_buffers(recurse=False, remove_duplicate=False):
                tensors[f"b_{sub_module_name}.{name}"] = buffer

        federated_storage = FederatedStorage(
            tensors,
            device=target_device,
            use_pin_memory=use_pin_memory,
            use_contigous_memory=use_contigous_memory,
            module=module,
        )
        setattr(module, "__weights_federated__", federated_storage)
        FederatedStorage.flush_federate_module(module)
        return federated_storage

    @staticmethod
    def flush_federate_module(
        module: torch.nn.Module, check_device_cpu: Optional[bool] = None
    ):
        federated_storage = getattr(
            module, "__weights_federated__", None
        )  # type:FederatedStorage
        if federated_storage is None:
            return

        tensors = federated_storage.get_tensors()
        for sub_module_name, sub_module in module.named_modules():
            for name, param in sub_module.named_parameters(recurse=False, remove_duplicate=False):
                if check_device_cpu is not None:
                    if (param.device.type == "cpu") != check_device_cpu:
                        print(
                            f"warning federate storage, param not in expected device(cpu={check_device_cpu}), get({param.device.type}) before do any operate"
                        )

                sub_module._parameters[name] = tensors[f"p_{sub_module_name}.{name}"]

            for name, buffer in sub_module.named_buffers(recurse=False, remove_duplicate=False):
                if check_device_cpu is not None:
                    if (buffer.device.type == "cpu") != check_device_cpu:
                        print(
                            f"warning federate storage, buffer not on expected device(cpu={check_device_cpu}), get({buffer.device.type}) before do any operate"
                        )

                sub_module._buffers[name] = tensors[f"b_{sub_module_name}.{name}"]

    @staticmethod
    def cuda_federate_module(
        module: torch.nn.Module, device: torch.device, check_device: bool = True
    ):
        if not hasattr(module, "__weights_federated__"):
            return cuda_module(
                module=module, device=device, contain_sub=True, skip_flexible=False
            )
        else:
            federated_storage = getattr(
                module, "__weights_federated__"
            )  # type:FederatedStorage

            federated_storage.htod(device=device, non_blocking=True)
            FederatedStorage.flush_federate_module(
                module=module, check_device_cpu=True if check_device else None
            )

    @staticmethod
    def offload_federate_module(module: torch.nn.Module, check_device: bool = True):
        if not hasattr(module, "__weights_federated__"):
            return offload_module(module=module, contain_sub=True, skip_flexible=False)
        else:
            federated_storage = getattr(
                module, "__weights_federated__"
            )  # type:FederatedStorage
            federated_storage.dtoh()
            FederatedStorage.flush_federate_module(
                module=module, check_device_cpu=False if check_device else None
            )

    @staticmethod
    def validate_no_cross_extent_aliases(
        extent_tensors: Mapping[str, Mapping[str, torch.Tensor]],
    ) -> None:
        owners: dict[
            tuple[int, int, int, torch.dtype],
            tuple[str, str],
        ] = {}
        for extent_name, tensors in extent_tensors.items():
            for full_name, tensor in tensors.items():
                identity = tensor_storage_identity(tensor)
                previous = owners.get(identity)
                if previous is not None and previous[0] != extent_name:
                    raise ValueError(
                        "shared tensor crosses flexible extents: "
                        f"{previous[1]} ({previous[0]}) and "
                        f"{full_name} ({extent_name})"
                    )
                owners[identity] = (extent_name, full_name)

    def release(self, *, restore_module_storage: bool = True) -> None:
        if self._released:
            return
        module = self._module_ref() if self._module_ref is not None else None
        if module is not None:
            replacement_by_identity: dict[
                tuple[int, int, int, torch.dtype],
                torch.Tensor,
            ] = {}
            if restore_module_storage:
                self.dtoh()
                tensors = self.get_tensors()
                for name, info in self.member_info.items():
                    if info.identity in replacement_by_identity:
                        continue
                    value = tensors[name].detach().cpu().clone()
                    if info.is_parameter:
                        value = nn.Parameter(
                            value,
                            requires_grad=info.requires_grad,
                        )
                    else:
                        value.requires_grad_(info.requires_grad)
                    replacement_by_identity[info.identity] = value
            else:
                for info in self.member_info.values():
                    if info.identity in replacement_by_identity:
                        continue
                    value = torch.empty(
                        0,
                        dtype=info.dtype,
                        device=torch.device("cpu"),
                    )
                    if info.is_parameter:
                        value = nn.Parameter(
                            value,
                            requires_grad=info.requires_grad,
                        )
                    else:
                        value.requires_grad_(info.requires_grad)
                    replacement_by_identity[info.identity] = value
            for submodule_name, submodule in module.named_modules():
                for name in tuple(submodule._parameters):
                    key = f"p_{submodule_name}.{name}"
                    info = self.member_info.get(key)
                    if info is not None:
                        submodule._parameters[name] = replacement_by_identity[
                            info.identity
                        ]
                for name in tuple(submodule._buffers):
                    key = f"b_{submodule_name}.{name}"
                    info = self.member_info.get(key)
                    if info is not None:
                        submodule._buffers[name] = replacement_by_identity[
                            info.identity
                        ]
            if getattr(module, "__weights_federated__", None) is self:
                delattr(module, "__weights_federated__")
        self.storage = None
        self._cpu_storage = None
        self.__pin_storage__ = None
        self.member_info.clear()
        self._released = True
