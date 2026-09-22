"""Service progress adapter and compatibility exports for request cancellation."""

from __future__ import annotations

from typing import Any

from entrypoints.server.task import TaskPhase
from nodes.control import CancellationToken, RequestCancelled, service_checkpoint


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


__all__ = (
    "CancellationToken",
    "RequestCancelled",
    "ServiceProgressState",
    "service_checkpoint",
)
