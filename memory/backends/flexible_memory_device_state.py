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

from collections import deque
from typing import Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import Future
from memory.backends.op_event import OPEvent
from memory.backends.event_type import EventType


class FlexibleMemoryDeviceState:
    def __init__(
        self,
        device: torch.device,
        async_worker: int = 8,
        compute_with_default_stream=True,
        max_weight_usage: int = 5 * 1024**3,
        initialize_cuda: bool = True,
    ):
        self.device = device
        # states
        self.total_flexible_register_bytes = 0
        self.flexible_usage_bytes = 0
        self.flexible_wait_onload = 0
        self.flexible_wait_offload = 0
        self.peak_flexible_usage_bytes = 0
        self.resident_bytes = 0
        self.peak_resident_bytes = 0

        self.__last_offload_op_event__ = None

        self.max_weight_usage = int(max_weight_usage)
        if self.max_weight_usage <= 0:
            raise ValueError("max_weight_usage must be positive")

        # streams
        if initialize_cuda:
            self.htod_stream = torch.cuda.Stream(device)
            self.compute_stream = (
                torch.cuda.current_stream(device)
                if compute_with_default_stream
                else torch.cuda.Stream(device)
            )
            self.dtoh_stream = torch.cuda.Stream(device)
        else:
            self.htod_stream = None
            self.compute_stream = None
            self.dtoh_stream = None

        # event_queue
        self.event_queue = deque()  # type:deque[OPEvent]

        # async worker
        self.async_worker = ThreadPoolExecutor(max_workers=async_worker)
        self.closed = False
        self.background_error: BaseException | None = None

    def add_op_event(self, op_event: OPEvent, skip_calc=True):
        if skip_calc and op_event.type == EventType.CALC:
            return

        if op_event.memory_effect > 0:
            self.flexible_wait_onload += abs(op_event.memory_effect)
        else:
            self.flexible_wait_offload += abs(op_event.memory_effect)
        self.event_queue.append(op_event)

    def remove_op_event(self, count=1) -> Tuple[Optional[OPEvent], bool]:
        last_event = None
        for _ in range(count):
            if len(self.event_queue) < 1:
                return last_event, False

            last_event = self.event_queue.popleft()  # type:OPEvent
            if last_event.memory_effect > 0:
                self.flexible_wait_onload -= abs(last_event.memory_effect)
            else:
                self.flexible_wait_offload -= abs(last_event.memory_effect)

            self.flexible_usage_bytes += last_event.memory_effect
            self.peak_flexible_usage_bytes = max(
                self.peak_flexible_usage_bytes,
                self.flexible_usage_bytes,
            )
        return last_event, True  # return last event

    def remove_op_event_to_timestemp(
        self, timestemp: int
    ) -> Tuple[Optional[OPEvent], bool]:
        """
        remove event until timestemp in last event that be removed or no event in list anymore
        """

        last_event = None
        while True:
            if len(self.event_queue) < 1:
                return last_event, False

            last_event = self.event_queue.popleft()  # type:OPEvent
            if last_event.memory_effect > 0:
                self.flexible_wait_onload -= abs(last_event.memory_effect)
            else:
                self.flexible_wait_offload -= abs(last_event.memory_effect)

            self.flexible_usage_bytes += last_event.memory_effect
            self.peak_flexible_usage_bytes = max(
                self.peak_flexible_usage_bytes,
                self.flexible_usage_bytes,
            )

            if last_event.timestamp == timestemp:
                return last_event, True  # return last event

    def find_op_rely(
        self, require_memory: int, max_memory: int = -1
    ) -> Optional[OPEvent]:
        """
        Just for calculation, it does not actually affect the data. Return Rely Event

        :param require_memory: cuda memoery need to use
        :type require_memory: int
        :param max_memory: max cuda memoery allow to use for flexible offload
        :type max_memory: int
        :return: op_event need to wait finish, none means no reliance.
        :rtype: OPEvent | None
        """
        self.validate_extent_budget("<unregistered>", require_memory)
        max_memory = self.max_weight_usage if max_memory < 0 else max_memory
        usage_memory = self.flexible_usage_bytes
        wait_onload_memory = self.flexible_wait_onload
        wait_offload_memory = self.flexible_wait_offload

        last_op_event = None
        for op_event in self.event_queue:
            if usage_memory + wait_onload_memory + require_memory <= max_memory:
                return last_op_event

            # memory is not enough
            if op_event.memory_effect > 0:
                wait_onload_memory -= abs(op_event.memory_effect)
            else:
                wait_offload_memory -= abs(op_event.memory_effect)
                last_op_event = op_event  # can only rely on op_event in other stream different HTODs

            usage_memory += op_event.memory_effect

        return last_op_event

    def validate_extent_budget(self, extent_name: str, required_bytes: int) -> None:
        required_bytes = int(required_bytes)
        if required_bytes > self.max_weight_usage:
            raise ValueError(
                f"extent {extent_name} requires {required_bytes} bytes, "
                f"exceeding max_weight_usage={self.max_weight_usage} "
                f"on {self.device}"
            )

    def mark_resident_acquired(self, bytes_size: int) -> None:
        bytes_size = int(bytes_size)
        if bytes_size < 0:
            raise ValueError("resident acquire bytes must be non-negative")
        candidate = self.resident_bytes + bytes_size
        if candidate > self.flexible_usage_bytes:
            raise RuntimeError(
                "resident bytes exceed flexible usage: "
                f"resident={candidate}, "
                f"flexible_usage={self.flexible_usage_bytes}"
            )
        if candidate > self.max_weight_usage:
            raise RuntimeError(
                "resident bytes exceed max_weight_usage: "
                f"resident={candidate}, "
                f"max_weight_usage={self.max_weight_usage}"
            )
        self.resident_bytes = candidate
        self.peak_resident_bytes = max(
            self.peak_resident_bytes,
            candidate,
        )

    def mark_resident_released(self, bytes_size: int) -> None:
        bytes_size = int(bytes_size)
        if bytes_size < 0:
            raise ValueError("resident release bytes must be non-negative")
        candidate = self.resident_bytes - bytes_size
        if candidate < 0:
            raise RuntimeError(
                "resident bytes cannot be negative: "
                f"resident={self.resident_bytes}, release={bytes_size}"
            )
        self.resident_bytes = candidate

    def add_async_job(self, func: Callable, *args, **kwargs) -> Future:
        if self.closed:
            raise RuntimeError(f"flexible memory state for {self.device} is closed")
        if self.background_error is not None:
            raise RuntimeError(
                f"flexible memory background task failed on {self.device}"
            ) from self.background_error
        return self.async_worker.submit(func, *args, **kwargs)

    def release(self, wait: bool = True):
        if self.closed:
            return
        self.closed = True
        self.async_worker.shutdown(wait=wait)

    def reset_counters(self) -> None:
        self.total_flexible_register_bytes = 0
        self.flexible_usage_bytes = 0
        self.flexible_wait_onload = 0
        self.flexible_wait_offload = 0
        self.peak_flexible_usage_bytes = 0
        self.resident_bytes = 0
        self.peak_resident_bytes = 0
        self.event_queue.clear()
