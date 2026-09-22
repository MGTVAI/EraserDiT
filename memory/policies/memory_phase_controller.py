"""Explicit large-component residency phases for the EraserDiT videoerase runtime."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import torch


class MemoryPhase(str, Enum):
    TEXT_ENCODE = "text_encode"
    VAE_ENCODE = "vae_encode"
    DENOISE = "denoise"
    VAE_DECODE = "vae_decode"
    COMMIT = "commit"


_PHASE_ORDER = {phase: index for index, phase in enumerate(MemoryPhase)}


@dataclass(frozen=True)
class MemoryPhaseEvent:
    monotonic_ns: int
    timestamp_ns: int
    rank: int
    window_key: tuple[int, int] | None
    phase: str | None
    component_name: str | None
    point: str
    allocated_bytes: int
    reserved_bytes: int
    adapter_resident_bytes: int
    active_component_name: str | None
    active_component_count: int


def _cuda_memory_snapshot(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return (0, 0)
    return (
        int(torch.cuda.memory_allocated(device)),
        int(torch.cuda.memory_reserved(device)),
    )


class MemoryPhaseController:
    def __init__(
        self,
        adapter: Any,
        *,
        rank: int,
        device: torch.device | str,
        cuda_memory_snapshot: Callable[[torch.device], tuple[int, int]]
        | None = None,
    ) -> None:
        self.adapter = adapter
        self.rank = int(rank)
        self.device = torch.device(device)
        self._cuda_memory_snapshot = cuda_memory_snapshot or _cuda_memory_snapshot
        self._active_phase: MemoryPhase | None = None
        self._active_component: str | None = None
        self._window_key: tuple[int, int] | None = None
        self._active_component_acquired = False
        self._last_phase_by_window: dict[tuple[int, int], MemoryPhase] = {}
        self._events: list[MemoryPhaseEvent] = []
        self._closed_events: tuple[MemoryPhaseEvent, ...] | None = None

    @property
    def events(self) -> tuple[MemoryPhaseEvent, ...]:
        if self._closed_events is not None:
            return self._closed_events
        return tuple(self._events)

    @property
    def active_phase(self) -> MemoryPhase | None:
        return self._active_phase

    def enter(
        self,
        phase: MemoryPhase,
        *,
        component_name: str | None,
        window_key: tuple[int, int],
    ) -> None:
        if self._closed_events is not None:
            raise RuntimeError("memory phase controller is closed")
        if self._active_phase is not None:
            raise RuntimeError("an active component phase already exists")
        if not isinstance(phase, MemoryPhase):
            raise TypeError("phase must be a MemoryPhase")
        if phase is MemoryPhase.COMMIT and component_name is not None:
            raise ValueError("commit phase cannot own a component")
        if phase is not MemoryPhase.COMMIT and not component_name:
            raise ValueError("component_name is required for compute phases")
        previous = self._last_phase_by_window.get(window_key)
        if previous is not None and _PHASE_ORDER[phase] < _PHASE_ORDER[previous]:
            raise RuntimeError("memory phase order cannot regress within a window")
        if component_name is not None:
            decision_fn = getattr(self.adapter, "plan_component_residency", None)
            decision = decision_fn(component_name) if callable(decision_fn) else None
            if decision is None or bool(getattr(decision, "enabled", True)):
                self.adapter.acquire_component_residency(
                    component_name,
                    reason=f"memory_phase:{phase.value}:enter",
                )
                self._active_component_acquired = True
            else:
                record_fallback = getattr(
                    self.adapter,
                    "record_component_residency_fallback",
                    None,
                )
                if callable(record_fallback):
                    record_fallback(decision)
        self._active_phase = phase
        self._active_component = component_name
        self._window_key = window_key
        self._last_phase_by_window[window_key] = phase
        self.record_point("enter")

    def exit(self, phase: MemoryPhase) -> None:
        if self._active_phase is not phase:
            raise RuntimeError("memory phase exit does not match active phase")
        self.record_point("exit")
        if self._active_component is not None and self._active_component_acquired:
            self.adapter.release_component_residency(
                self._active_component,
                reason=f"memory_phase:{phase.value}:exit",
            )
            self.adapter.settle_component_transfers()
        self._active_phase = None
        self._active_component = None
        self._active_component_acquired = False
        self._window_key = None

    def record_point(self, point: str) -> MemoryPhaseEvent:
        snapshot = self.adapter.snapshot()
        allocated, reserved = self._cuda_memory_snapshot(self.device)
        resident_bytes = int(snapshot.get("resident_bytes", 0))
        event = MemoryPhaseEvent(
            monotonic_ns=time.monotonic_ns(),
            timestamp_ns=time.time_ns(),
            rank=self.rank,
            window_key=self._window_key,
            phase=self._active_phase.value
            if self._active_phase is not None
            else None,
            component_name=self._active_component,
            point=str(point),
            allocated_bytes=int(allocated),
            reserved_bytes=int(reserved),
            adapter_resident_bytes=resident_bytes,
            active_component_name=snapshot.get("active_component_name"),
            active_component_count=int(
                snapshot.get("active_component_name") is not None
            ),
        )
        self._events.append(event)
        return event

    def close(self) -> tuple[MemoryPhaseEvent, ...]:
        if self._closed_events is not None:
            return self._closed_events
        if self._active_phase is not None:
            self.exit(self._active_phase)
        self._closed_events = tuple(self._events)
        return self._closed_events


__all__ = ("MemoryPhase", "MemoryPhaseController", "MemoryPhaseEvent")
