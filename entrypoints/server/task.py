"""Task lifecycle contracts for the EraserDiT service."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPhase(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    PROCESSING = "processing"
    FINALIZING = "finalizing"
    TERMINAL = "terminal"


TERMINAL_STATUSES = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)


class ServiceError(RuntimeError):
    """Stable service-layer failure suitable for API error mapping."""

    def __init__(self, code: str, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.status_code = int(status_code)


@dataclass(frozen=True)
class TaskError:
    code: str
    message: str
    phase: str | None = None

    def as_dict(self) -> dict[str, str]:
        payload = {"code": self.code, "message": self.message}
        if self.phase is not None:
            payload["phase"] = self.phase
        return payload


_ALLOWED_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.QUEUED: frozenset({TaskStatus.RUNNING, TaskStatus.CANCELLED}),
    TaskStatus.RUNNING: frozenset(
        {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}


@dataclass
class TaskRecord:
    """Mutable task state; callers must mutate it through ``TaskStore``."""

    task_id: str
    request_payload: dict[str, Any]
    task_dir: Path
    video_input_path: Path
    mask_input_path: Path
    bbox_input_path: Path | None = None
    status: TaskStatus = TaskStatus.QUEUED
    phase: TaskPhase = TaskPhase.QUEUED
    progress: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    expires_at: float | None = None
    output_path: Path | None = None
    result_url: str | None = None
    storage_mode: str = "local"
    storage_fallback: bool = False
    storage_committed: bool = False
    error: TaskError | None = None
    queue_position: int | None = None
    object_index: int | None = None
    object_count: int | None = None
    window_index: int | None = None
    window_count: int | None = None
    cancellation_requested: bool = False
    cancellation_requested_at: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def transition(
        self,
        status: TaskStatus,
        *,
        now: float | None = None,
        ttl_seconds: int | None = None,
        error: TaskError | None = None,
    ) -> None:
        status = TaskStatus(status)
        if status not in _ALLOWED_TRANSITIONS[self.status]:
            raise ServiceError(
                "invalid_task_transition",
                f"cannot transition task from {self.status} to {status}",
                status_code=409,
            )
        timestamp = time.time() if now is None else float(now)
        self.status = status
        self.error = error
        if status is TaskStatus.RUNNING:
            self.started_at = timestamp
            self.phase = TaskPhase.PREPARING
            self.queue_position = None
        elif status in TERMINAL_STATUSES:
            self.completed_at = timestamp
            self.expires_at = (
                timestamp + int(ttl_seconds) if ttl_seconds is not None else None
            )
            self.phase = TaskPhase.TERMINAL
            self.queue_position = None
            if status is TaskStatus.COMPLETED:
                self.progress = 100

    def update_progress(
        self,
        *,
        phase: TaskPhase,
        progress: int,
        object_index: int | None = None,
        object_count: int | None = None,
        window_index: int | None = None,
        window_count: int | None = None,
    ) -> None:
        if self.status is not TaskStatus.RUNNING:
            return
        normalized = max(self.progress, min(max(int(progress), 0), 99))
        self.phase = TaskPhase(phase)
        self.progress = normalized
        if object_index is not None:
            self.object_index = object_index
        if object_count is not None:
            self.object_count = object_count
        if window_index is not None:
            self.window_index = window_index
        if window_count is not None:
            self.window_count = window_count

    def public_dict(self) -> dict[str, Any]:
        content_url = (
            f"/v1/videos/{self.task_id}/content"
            if (
                self.status is TaskStatus.COMPLETED
                and self.output_path is not None
                and self.result_url is None
            )
            else None
        )
        return {
            "id": self.task_id,
            "object": "video",
            "status": self.status.value,
            "phase": self.phase.value,
            "progress": int(self.progress),
            "created_at": int(self.created_at),
            "started_at": (
                int(self.started_at) if self.started_at is not None else None
            ),
            "completed_at": (
                int(self.completed_at) if self.completed_at is not None else None
            ),
            "expires_at": (
                int(self.expires_at) if self.expires_at is not None else None
            ),
            "queue_position": self.queue_position,
            "object_index": self.object_index,
            "object_count": self.object_count,
            "window_index": self.window_index,
            "window_count": self.window_count,
            "url": self.result_url,
            "content_url": content_url,
            "storage_mode": self.storage_mode,
            "storage_fallback": self.storage_fallback,
            "error": self.error.as_dict() if self.error is not None else None,
            "metrics": dict(self.metrics),
        }


__all__ = (
    "ServiceError",
    "TaskError",
    "TaskPhase",
    "TaskRecord",
    "TaskStatus",
    "TERMINAL_STATUSES",
)
