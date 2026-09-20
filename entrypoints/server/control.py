"""Cooperative request cancellation and service progress adapters."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from entrypoints.server.task import TaskPhase


class RequestCancelled(RuntimeError):
    """Raised by every participating rank at the same safe boundary."""


@dataclass
class CancellationToken:
    request_id: str
    _event: threading.Event = field(default_factory=threading.Event, repr=False)
    requested_at: float | None = None
    last_checkpoint_at: float = field(default_factory=time.time)

    def request(self) -> None:
        if not self._event.is_set():
            self.requested_at = time.time()
            self._event.set()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def touch(self) -> None:
        self.last_checkpoint_at = time.time()


class ServiceProgressState:
    """RuntimeProgressState-compatible owner sink backed by TaskStore."""

    def __init__(self, task_store: Any, task_id: str) -> None:
        self.task_store = task_store
        self.task_id = task_id
        self._pipeline_completed = 0
        self._pipeline_total = 1

    def stop(self) -> None:
        return None

    def add_pipeline_task(self, total: int) -> None:
        self._pipeline_total = max(int(total), 1)

    def update_pipeline(
        self,
        completed: int,
        total: int,
        *,
        object_index: int,
        object_count: int,
        window_index: int,
        window_count: int,
    ) -> None:
        self._pipeline_completed = max(int(completed), 0)
        self._pipeline_total = max(int(total), 1)
        progress = min(94, 5 + int(89 * self._pipeline_completed / self._pipeline_total))
        self.task_store.update_progress(
            self.task_id,
            phase=TaskPhase.PROCESSING,
            progress=progress,
            object_index=object_index,
            object_count=object_count,
            window_index=window_index,
            window_count=window_count,
        )

    def reset_denoise_task(self, **_: object) -> None:
        return None

    def update_denoise(self, *_: object, **__: object) -> None:
        return None

    def hide_denoise(self) -> None:
        return None

    def set_pipeline_meta(self, meta: str) -> None:
        return None


def service_checkpoint(batch: Any, server_args: Any, *, phase: str) -> None:
    """Synchronize owner cancellation at a predefined all-rank safe point."""

    extra = getattr(batch, "extra", None)
    token = extra.get("service_cancellation_token") if isinstance(extra, dict) else None
    if token is None and server_args is not None:
        token = getattr(server_args, "_service_cancellation_token", None)
    if token is None:
        return
    cancelled = bool(getattr(token, "requested", False))
    context = getattr(server_args, "parallel_context", None)
    if context is not None and bool(getattr(context, "enabled", False)):
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("service cancellation requires torch.distributed")
        group = getattr(context, "control_process_group", None)
        backend = str(
            dist.get_backend(group) if group is not None else dist.get_backend()
        ).lower()
        device = (
            torch.device("cuda", int(getattr(context, "local_rank", 0) or 0))
            if backend == "nccl"
            else torch.device("cpu")
        )
        value = torch.tensor([int(cancelled)], dtype=torch.int64, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX, group=group)
        cancelled = bool(value.item())
    token.touch()
    if cancelled:
        raise RequestCancelled(f"request cancelled at {phase}")


__all__ = (
    "CancellationToken",
    "RequestCancelled",
    "ServiceProgressState",
    "service_checkpoint",
)
