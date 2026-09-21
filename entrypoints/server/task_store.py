"""Thread-safe in-memory task registry with owned-directory cleanup."""

from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Iterable
from pathlib import Path

from entrypoints.server.task import (
    ServiceError,
    TaskError,
    TaskPhase,
    TaskRecord,
    TaskStatus,
    TERMINAL_STATUSES,
)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class TaskStore:
    def __init__(
        self,
        task_root: str | Path,
        *,
        terminal_ttl_seconds: int,
        max_terminal_tasks: int,
    ) -> None:
        self.root = Path(task_root).expanduser().resolve()
        self.tasks_root = self.root / "tasks"
        self.staging_root = self.root / "staging"
        self.terminal_ttl_seconds = int(terminal_ttl_seconds)
        self.max_terminal_tasks = int(max_terminal_tasks)
        self._records: dict[str, TaskRecord] = {}
        self._lock = threading.RLock()
        self.tasks_root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)

    def create(self, record: TaskRecord) -> TaskRecord:
        with self._lock:
            if record.task_id in self._records:
                raise ServiceError(
                    "task_already_exists",
                    f"task {record.task_id} already exists",
                    status_code=409,
                )
            task_dir = record.task_dir.expanduser().resolve()
            if not _is_relative_to(task_dir, self.tasks_root):
                raise ServiceError(
                    "invalid_task_directory",
                    "task directory is outside the service task root",
                    status_code=500,
                )
            record.task_dir = task_dir
            self._records[record.task_id] = record
            return record

    def get(self, task_id: str) -> TaskRecord:
        with self._lock:
            try:
                return self._records[task_id]
            except KeyError as error:
                raise ServiceError(
                    "task_not_found",
                    f"task {task_id} was not found",
                    status_code=404,
                ) from error

    def snapshot(self, task_id: str) -> dict[str, object]:
        with self._lock:
            return dict(self.get(task_id).public_dict())

    def list_records(
        self,
        *,
        after: str | None = None,
        limit: int = 20,
        order: str = "desc",
    ) -> list[TaskRecord]:
        records, _ = self.list_page(after=after, limit=limit, order=order)
        return records

    def list_page(
        self, *, after: str | None = None, limit: int = 20, order: str = "desc"
    ) -> tuple[list[TaskRecord], bool]:
        """Select a page and determine whether another row exists under one lock."""
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ServiceError(
                "invalid_limit", "limit must be in [1, 100]", status_code=422
            )
        if order not in {"asc", "desc"}:
            raise ServiceError(
                "invalid_order", "order must be 'asc' or 'desc'", status_code=422
            )
        with self._lock:
            records = sorted(
                self._records.values(),
                key=lambda record: (record.created_at, record.task_id),
                reverse=order == "desc",
            )
            if after is not None:
                indexes = [
                    index
                    for index, record in enumerate(records)
                    if record.task_id == after
                ]
                if not indexes:
                    raise ServiceError(
                        "invalid_cursor",
                        f"after cursor {after} was not found",
                        status_code=422,
                    )
                records = records[indexes[0] + 1 :]
            return list(records[:limit]), len(records) > limit

    def transition(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        error: TaskError | None = None,
        now: float | None = None,
    ) -> TaskRecord:
        with self._lock:
            record = self.get(task_id)
            record.transition(
                status,
                error=error,
                now=now,
                ttl_seconds=(
                    self.terminal_ttl_seconds if status in TERMINAL_STATUSES else None
                ),
            )
            return record

    def update_progress(
        self,
        task_id: str,
        *,
        phase: TaskPhase,
        progress: int,
        object_index: int | None = None,
        object_count: int | None = None,
        window_index: int | None = None,
        window_count: int | None = None,
    ) -> TaskRecord:
        with self._lock:
            record = self.get(task_id)
            record.update_progress(
                phase=phase,
                progress=progress,
                object_index=object_index,
                object_count=object_count,
                window_index=window_index,
                window_count=window_count,
            )
            return record

    def request_cancellation(self, task_id: str) -> TaskRecord:
        with self._lock:
            record = self.get(task_id)
            if record.is_terminal:
                raise ServiceError(
                    "task_already_terminal",
                    "terminal tasks must be deleted instead of cancelled",
                    status_code=409,
                )
            if record.storage_committed:
                raise ServiceError(
                    "task_finalizing",
                    "task result publication has started and can no longer be cancelled",
                    status_code=409,
                )
            record.cancellation_requested = True
            if record.cancellation_requested_at is None:
                record.cancellation_requested_at = time.time()
            return record

    def begin_result_publication(self, task_id: str) -> bool:
        """Atomically cross the non-cancellable result-publication boundary."""

        with self._lock:
            record = self.get(task_id)
            if record.status is not TaskStatus.RUNNING:
                raise ServiceError(
                    "task_not_running",
                    "only a running task can publish a result",
                    status_code=409,
                )
            if record.cancellation_requested:
                return False
            record.storage_committed = True
            record.update_progress(phase=TaskPhase.FINALIZING, progress=99)
            return True

    def set_result_storage(
        self,
        task_id: str,
        *,
        mode: str,
        local_path: Path | None,
        url: str | None,
        fallback: bool,
    ) -> TaskRecord:
        with self._lock:
            record = self.get(task_id)
            if not record.storage_committed or record.status is not TaskStatus.RUNNING:
                raise ServiceError(
                    "result_publication_not_started",
                    "result storage can only be set during publication",
                    status_code=409,
                )
            if mode not in {"local", "s3"}:
                raise ValueError("result storage mode must be local or s3")
            if (local_path is None) == (url is None):
                raise ValueError("result storage must expose exactly one location")
            record.storage_mode = mode
            record.storage_fallback = bool(fallback)
            record.output_path = (
                local_path.resolve() if local_path is not None else None
            )
            record.result_url = url
            return record

    def update_queue_positions(self, task_ids: Iterable[str]) -> None:
        with self._lock:
            for position, task_id in enumerate(task_ids):
                record = self._records.get(task_id)
                if record is not None and record.status is TaskStatus.QUEUED:
                    record.queue_position = position

    def purge(self, task_id: str) -> TaskRecord:
        with self._lock:
            record = self.get(task_id)
            if not record.is_terminal:
                raise ServiceError(
                    "task_not_terminal",
                    "only terminal tasks can be purged",
                    status_code=409,
                )
            task_dir = record.task_dir.resolve()
            if not _is_relative_to(task_dir, self.tasks_root):
                raise ServiceError(
                    "invalid_task_directory",
                    "refusing to delete a directory outside the service task root",
                    status_code=500,
                )
            del self._records[task_id]
        if task_dir.exists():
            shutil.rmtree(task_dir)
        return record

    def cleanup_expired(self, *, now: float | None = None) -> list[str]:
        timestamp = time.time() if now is None else float(now)
        with self._lock:
            terminal = [
                record for record in self._records.values() if record.is_terminal
            ]
            terminal.sort(key=lambda record: record.completed_at or record.created_at)
            expired_ids = {
                record.task_id
                for record in terminal
                if record.expires_at is not None and record.expires_at <= timestamp
            }
            retained = [
                record for record in terminal if record.task_id not in expired_ids
            ]
            overflow = max(0, len(retained) - self.max_terminal_tasks)
            expired_ids.update(record.task_id for record in retained[:overflow])
        purged: list[str] = []
        for task_id in sorted(expired_ids):
            self.purge(task_id)
            purged.append(task_id)
        return purged

    def status_counts(self) -> dict[str, int]:
        with self._lock:
            return {
                status.value: sum(
                    record.status is status for record in self._records.values()
                )
                for status in TaskStatus
            }

    def latest_completed_metrics(self) -> dict[str, object] | None:
        with self._lock:
            completed = [
                record
                for record in self._records.values()
                if record.status is TaskStatus.COMPLETED
            ]
            if not completed:
                return None
            record = max(
                completed,
                key=lambda item: item.completed_at or item.created_at,
            )
            return {
                "request_id": record.task_id,
                "completed_at": record.completed_at,
                "metrics": dict(record.metrics),
            }


__all__ = ("TaskStore",)
