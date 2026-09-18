"""CUDA-streamed flexible extent with per-device budget backpressure."""

from __future__ import annotations

from concurrent.futures import Future
from contextlib import AbstractContextManager
from typing import Any, Callable

import torch
from torch import nn

from memory.backends.event_type import EventType
from memory.backends.flexible_memory_device_state import (
    FlexibleMemoryDeviceState,
)
from memory.backends.flexible_memory_states import FlexibleMemoryState
from memory.backends.flexible_module_extent import FlexibleModuleExtent
from memory.backends.flexible_module_extent_base import (
    FlexibleMemoryTypeEnum,
    FlexibleSupportStagesEnum,
)
from memory.backends.op_event import OPEvent


class FlexibleModuleExtentCudaAsync(FlexibleModuleExtent):
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
        device_state: FlexibleMemoryDeviceState | None = None,
        event_factory: Callable[[], Any] | None = None,
        stream_context: Callable[[Any], AbstractContextManager] | None = None,
        current_stream_fn: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__(
            module,
            calc_device,
            use_weak_ref=use_weak_ref,
            init_offload=init_offload,
            memory_contigous=memory_contigous,
            storage_type=storage_type,
        )
        self._device_state = device_state
        self._event_factory = event_factory or torch.cuda.Event
        self._stream_context = stream_context or torch.cuda.stream
        self._current_stream_fn = current_stream_fn or (
            lambda: torch.cuda.current_stream(self.calc_device)
        )
        self._offload_cpu_job: Future | None = None
        self._last_offload_event: OPEvent | None = None
        self._active_onload_event: OPEvent | None = None
        self._active_calc_event: OPEvent | None = None
        self._resident = False
        self._resident_onload_count = 0
        self._resident_onload_bytes = 0
        self._resident_release_count = 0
        self._resident_release_bytes = 0
        self._resident_fast_path_forward_count = 0
        self._accept_tasks = True

    def _state(self) -> FlexibleMemoryDeviceState:
        if self._device_state is not None:
            return self._device_state
        return FlexibleMemoryState.get_device_state(self.calc_device)

    @torch.compiler.disable
    def _new_op_event(self, event_type: EventType) -> OPEvent:
        return OPEvent(
            event_type,
            self.estimated_memory_requirement(),
            start_cuda_event=self._event_factory(),
            end_cuda_event=self._event_factory(),
            object_id=id(self.module_ref),
            object_tag=self.module_ref.__class__.__name__,
        )

    def _finish_previous_offload(self, state: FlexibleMemoryDeviceState) -> None:
        if self._offload_cpu_job is None:
            return
        try:
            self._offload_cpu_job.result()
        except BaseException as exc:
            state.background_error = exc
            raise
        finally:
            self._offload_cpu_job = None
        if self._last_offload_event is not None:
            state.remove_op_event_to_timestemp(
                self._last_offload_event.timestamp
            )
            self._last_offload_event = None

    def support_train(self) -> FlexibleSupportStagesEnum:
        return FlexibleSupportStagesEnum.NoneSupport

    def support_inference(self) -> FlexibleSupportStagesEnum:
        return FlexibleSupportStagesEnum.FullSupport

    @property
    def is_resident(self) -> bool:
        return self._resident

    def weight_onload(self, label: str = "") -> None:
        if not self._accept_tasks:
            raise RuntimeError("flexible extent is closed")
        state = self._state()
        if state.background_error is not None:
            raise RuntimeError(
                f"flexible memory background task failed on {state.device}"
            ) from state.background_error
        self._finish_previous_offload(state)

        required = self.estimated_memory_requirement()
        state.validate_extent_budget(self.module_ref.__class__.__name__, required)
        onload_event = self._new_op_event(EventType.HTOD)
        rely = state.find_op_rely(required)
        if rely is not None:
            state.remove_op_event_to_timestemp(rely.timestamp)
            rely.sync_cpu()
            rely.end_cuda_event.synchronize()

        with self._stream_context(state.htod_stream):
            current = self._current_stream_fn()
            onload_event.record_start(current)
            super().weight_onload(label)
            onload_event.record_end(current)
        state.add_op_event(onload_event)
        self._active_onload_event = onload_event

    @torch.compiler.disable(recursive=False)
    def inference(self, label: str = "", *args, **kwargs) -> Any:
        state = self._state()
        calc_event = self._new_op_event(EventType.CALC)
        caller_stream = self._current_stream_fn()
        with self._stream_context(state.compute_stream):
            state.compute_stream.wait_event(
                self._active_onload_event.end_cuda_event
            )
            current = self._current_stream_fn()
            calc_event.record_start(current)
            output = super().inference(label, *args, **kwargs)
            calc_event.record_end(current)
        if caller_stream is not state.compute_stream:
            caller_stream.wait_event(calc_event.end_cuda_event)
        self._active_calc_event = calc_event
        return output

    def _settle_active_onload(
        self,
        state: FlexibleMemoryDeviceState,
    ) -> None:
        event = self._active_onload_event
        if event is None:
            raise RuntimeError("resident acquire produced no onload event")
        event.sync_cpu()
        event.end_cuda_event.synchronize()
        state.remove_op_event_to_timestemp(event.timestamp)

    def acquire_residency(self, *, label: str) -> None:
        if self._resident:
            raise RuntimeError("flexible extent is already resident")
        state = self._state()
        self._active_calc_event = None
        required = self.estimated_memory_requirement()
        try:
            self.weight_onload(label)
            self._settle_active_onload(state)
            state.mark_resident_acquired(required)
        except BaseException:
            if self._active_onload_event is not None:
                self.weight_offload(f"{label}:acquire_rollback")
                self._finish_previous_offload(state)
            self._active_onload_event = None
            self._active_calc_event = None
            raise
        self._resident = True
        self._resident_onload_count += 1
        self._resident_onload_bytes += required

    @torch.compiler.disable(recursive=False)
    def forward(self, *args, **kwargs) -> Any:
        if not self._resident:
            return super().forward(*args, **kwargs)
        label = self.generate_label()
        self._resident_fast_path_forward_count += 1
        return self.inference(label, *args, **kwargs)

    def _offload_after_compute(
        self,
        state: FlexibleMemoryDeviceState,
        offload_event: OPEvent,
        label: str,
    ) -> None:
        with self._stream_context(state.dtoh_stream):
            if self._active_calc_event is not None:
                state.dtoh_stream.wait_event(
                    self._active_calc_event.end_cuda_event
                )
            current = self._current_stream_fn()
            offload_event.record_start(current)
            offload_event.record_end(current)
        offload_event.end_cuda_event.synchronize()
        super().weight_offload(label)

    def weight_offload(self, label: str = "") -> None:
        state = self._state()
        offload_event = self._new_op_event(EventType.DTOH)
        self._offload_cpu_job = state.add_async_job(
            self._offload_after_compute,
            state,
            offload_event,
            label,
        )
        offload_event.cpu_event = self._offload_cpu_job
        state.add_op_event(offload_event)
        self._last_offload_event = offload_event

    def release_residency(self, *, label: str) -> None:
        if not self._resident:
            raise RuntimeError("flexible extent is not resident")
        state = self._state()
        required = self.estimated_memory_requirement()
        self.weight_offload(label)
        self._finish_previous_offload(state)
        state.mark_resident_released(required)
        self._resident = False
        self._resident_release_count += 1
        self._resident_release_bytes += required
        self._active_onload_event = None
        self._active_calc_event = None

    def residency_snapshot(self) -> dict[str, object]:
        return {
            "is_resident": self._resident,
            "resident_onload_count": self._resident_onload_count,
            "resident_onload_bytes": self._resident_onload_bytes,
            "resident_release_count": self._resident_release_count,
            "resident_release_bytes": self._resident_release_bytes,
            "resident_fast_path_forward_count": (
                self._resident_fast_path_forward_count
            ),
        }

    def close(
        self,
        *,
        restore_forward: bool = False,
        restore_module_storage: bool = True,
    ) -> None:
        if not self._accept_tasks:
            return
        if self._resident:
            self.release_residency(label="extent_close")
        self._accept_tasks = False
        state = self._state()
        self._finish_previous_offload(state)
        super().close(
            restore_forward=restore_forward,
            restore_module_storage=restore_module_storage,
        )
