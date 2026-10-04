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
        self._location: dict[str, int] = {}

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

    def reset_denoise_task(self, *, object_index: int, object_count: int,
                           window_index: int, window_count: int, total_steps: int) -> None:
        self._location = dict(object_index=object_index, object_count=object_count,
                              window_index=window_index, window_count=window_count)
        self._update_fraction(0.)

    def _update_fraction(self, fraction: float) -> None:
        # Reserve the final tenth of each window for decoding and commit.
        completed = self._pipeline_completed + .9 * max(0., min(fraction, 1.))
        progress = min(94, 5 + int(89 * completed / self._pipeline_total))
        self.task_store.update_progress(self.task_id, phase=TaskPhase.PROCESSING,
                                        progress=progress, **self._location)

    def update_denoise(self, step_index: int, total_steps: int, **_: object) -> None:
        self._update_fraction((step_index + 1) / max(total_steps, 1))

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
