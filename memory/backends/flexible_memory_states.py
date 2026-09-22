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
import torch
from memory.backends.flexible_memory_device_state import FlexibleMemoryDeviceState


class FlexibleMemoryState:
    total_flexible_register_bytes = 0
    device_states = {}

    @staticmethod
    def get_device_idx(device: torch.device = None) -> int:
        if device is None:
            return torch.cuda.current_device()
        else:
            assert device.type == "cuda", "only support cuda device"
            return (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )

    @staticmethod
    def _new_device_state(
        device: torch.device,
        max_weight_usage: int,
    ) -> FlexibleMemoryDeviceState:
        return FlexibleMemoryDeviceState(
            device=device,
            max_weight_usage=max_weight_usage,
        )

    @staticmethod
    def add_device_state(
        device: torch.device = None,
        *,
        max_weight_usage: int = 2 * 1024**3,
    ):
        device_idx = FlexibleMemoryState.get_device_idx(device=device)

        if device_idx not in FlexibleMemoryState.device_states:
            cuda_device = torch.device(f"cuda:{device_idx}")
            FlexibleMemoryState.device_states[device_idx] = (
                FlexibleMemoryState._new_device_state(
                    cuda_device,
                    int(max_weight_usage),
                )
            )
        return FlexibleMemoryState.device_states[device_idx]

    @staticmethod
    def get_device_state(device: torch.device = None) -> FlexibleMemoryDeviceState:
        device_idx = FlexibleMemoryState.get_device_idx(device=device)

        if device_idx not in FlexibleMemoryState.device_states:
            FlexibleMemoryState.add_device_state(device)

        return FlexibleMemoryState.device_states.get(device_idx, None)

    @staticmethod
    def add_flexible_register_bytes(bytes_size: int, device: torch.device = None):
        FlexibleMemoryState.get_device_state(
            device=device
        ).total_flexible_register_bytes += bytes_size
        FlexibleMemoryState.total_flexible_register_bytes += bytes_size

    @staticmethod
    def remove_flexible_register_bytes(bytes_size: int, device: torch.device = None):
        FlexibleMemoryState.get_device_state(
            device=device
        ).total_flexible_register_bytes -= bytes_size
        FlexibleMemoryState.total_flexible_register_bytes -= bytes_size

    @staticmethod
    def reset_flexible_register_bytes(device: torch.device = None):
        device_state = FlexibleMemoryState.get_device_state(device=device)
        FlexibleMemoryState.total_flexible_register_bytes -= (
            device_state.total_flexible_register_bytes
        )
        device_state.total_flexible_register_bytes = 0

    # @staticmethod
    # def set_min_distance(min_distance=0, device: torch.device = None):
    #     device_state = FlexibleMemoryState.get_device_state(device=device)
    #     device_state.min_distance = min_distance

    @staticmethod
    def set_max_memory_usage(
        max_memory_usage=5 * (1024**3), device: torch.device = None
    ):
        device_state = FlexibleMemoryState.get_device_state(device=device)
        if int(max_memory_usage) <= 0:
            raise ValueError("max_weight_usage must be positive")
        device_state.max_weight_usage = int(max_memory_usage)

    @staticmethod
    def configure_device(
        device: torch.device,
        *,
        max_weight_usage: int,
    ) -> FlexibleMemoryDeviceState:
        if int(max_weight_usage) <= 0:
            raise ValueError("max_weight_usage must be positive")
        state = FlexibleMemoryState.add_device_state(
            device,
            max_weight_usage=int(max_weight_usage),
        )
        state.max_weight_usage = int(max_weight_usage)
        return state

    @staticmethod
    def snapshot(device: torch.device) -> dict[str, int]:
        state = FlexibleMemoryState.get_device_state(device=device)
        return {
            "total_flexible_register_bytes": int(
                state.total_flexible_register_bytes
            ),
            "flexible_usage_bytes": int(state.flexible_usage_bytes),
            "flexible_wait_onload": int(state.flexible_wait_onload),
            "flexible_wait_offload": int(state.flexible_wait_offload),
            "peak_flexible_usage_bytes": int(
                state.peak_flexible_usage_bytes
            ),
            "resident_bytes": int(state.resident_bytes),
            "peak_resident_bytes": int(state.peak_resident_bytes),
            "max_weight_usage": int(state.max_weight_usage),
            "event_queue_size": len(state.event_queue),
        }

    @staticmethod
    def reset_device(
        device: torch.device,
        *,
        release_worker: bool,
    ) -> None:
        device_idx = FlexibleMemoryState.get_device_idx(device=device)
        state = FlexibleMemoryState.device_states.pop(device_idx, None)
        if state is None:
            return
        FlexibleMemoryState.total_flexible_register_bytes -= (
            state.total_flexible_register_bytes
        )
        FlexibleMemoryState.total_flexible_register_bytes = max(
            0,
            FlexibleMemoryState.total_flexible_register_bytes,
        )
        if release_worker:
            state.release(wait=True)
        state.reset_counters()

    @staticmethod
    def reset_all(*, release_workers: bool) -> None:
        for device_idx in tuple(FlexibleMemoryState.device_states):
            FlexibleMemoryState.reset_device(
                torch.device(f"cuda:{device_idx}"),
                release_worker=release_workers,
            )
        FlexibleMemoryState.total_flexible_register_bytes = 0

    @staticmethod
    def release(wait: bool = True, device: torch.device = None):
        device_state = FlexibleMemoryState.get_device_state(device=device)
        device_state.release(wait=wait)

    @staticmethod
    def release_all(wait: bool = True):
        for _, device_state in FlexibleMemoryState.device_states.items():
            device_state.release(wait=wait)
