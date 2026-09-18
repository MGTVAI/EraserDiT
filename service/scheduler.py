"""Strict FIFO scheduler with one active LTX095 GPU request."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from service.control import RequestCancelled
from service.task import (
    ServiceError,
    TaskError,
    TaskPhase,
    TaskRecord,
    TaskStatus,
)
from utils.inference_timing import build_ltx095_pure_timing_payload
from utils.logging_utils import init_logger

logger = init_logger(__name__)


def _log_event(event: str, **fields: object) -> None:
    logger.info(
        "%s",
        json.dumps(
            {"event": event, "timestamp": time.time(), **fields},
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


class ServiceScheduler:
    def __init__(
        self,
        task_store: Any,
        worker_group: Any,
        result_storage: Any,
        *,
        max_queued_tasks: int,
        warmup_steps: int | None = None,
        cancel_timeout_seconds: float = 120.0,
    ) -> None:
        self.task_store = task_store
        self.worker_group = worker_group
        self.result_storage = result_storage
        self.max_queued_tasks = int(max_queued_tasks)
        self._queue: deque[str] = deque()
        self._condition = threading.Condition()
        self._accepting = True
        self._active_task_id: str | None = None
        self._warmup_steps = warmup_steps
        self._cancel_timeout_seconds = float(cancel_timeout_seconds)
        self._cancel_watchdogs: set[str] = set()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="mgerase-service-scheduler",
            daemon=True,
        )
        self._thread.start()

    @property
    def active_task_id(self) -> str | None:
        with self._condition:
            return self._active_task_id

    def submit(self, record: TaskRecord) -> TaskRecord:
        with self._condition:
            if not self._accepting:
                raise ServiceError(
                    "service_not_ready", "service is draining", status_code=503
                )
            if len(self._queue) >= self.max_queued_tasks:
                raise ServiceError(
                    "queue_full", "service queue is full", status_code=429
                )
            self.task_store.create(record)
            self._queue.append(record.task_id)
            self._refresh_positions_locked()
            self._condition.notify()
            _log_event(
                "request_queued",
                request_id=record.task_id,
                queue_depth=len(self._queue),
            )
            return record

    def cancel_or_purge(self, task_id: str) -> tuple[str, dict[str, object]]:
        with self._condition:
            record = self.task_store.get(task_id)
            if record.is_terminal:
                self.task_store.purge(task_id)
                _log_event("result_deleted", request_id=task_id)
                return "purged", {
                    "id": task_id,
                    "deleted": True,
                    "remote_result_deleted": False,
                }
            if record.status is TaskStatus.QUEUED:
                try:
                    self._queue.remove(task_id)
                except ValueError:
                    pass
                self.task_store.request_cancellation(task_id)
                self.task_store.transition(task_id, TaskStatus.CANCELLED)
                self._remove_incomplete_artifacts(record)
                self._refresh_positions_locked()
                _log_event(
                    "request_cancelled",
                    request_id=task_id,
                    phase=TaskPhase.QUEUED.value,
                    cancellation_latency_seconds=0.0,
                )
                return "cancelled", self.task_store.snapshot(task_id)
            self.task_store.request_cancellation(task_id)
            self.worker_group.request_cancellation(task_id)
            _log_event("cancellation_requested", request_id=task_id)
            self._start_cancel_watchdog_locked(task_id)
            return "cancellation_requested", self.task_store.snapshot(task_id)

    def _start_cancel_watchdog_locked(self, task_id: str) -> None:
        if task_id in self._cancel_watchdogs:
            return
        self._cancel_watchdogs.add(task_id)
        threading.Thread(
            target=self._cancel_watchdog,
            args=(task_id,),
            name=f"mgerase-cancel-watchdog-{task_id[:8]}",
            daemon=True,
        ).start()

    def _cancel_watchdog(self, task_id: str) -> None:
        time.sleep(self._cancel_timeout_seconds)
        with self._condition:
            self._cancel_watchdogs.discard(task_id)
            record = self.task_store.get(task_id)
            if (
                record.status is not TaskStatus.RUNNING
                or not record.cancellation_requested
            ):
                return
            message = (
                f"request {task_id} did not reach an all-rank cancellation boundary "
                f"within {self._cancel_timeout_seconds:.3f}s"
            )
            self.worker_group.mark_fatal(message)
            self._accepting = False
            _log_event(
                "cancellation_timeout",
                request_id=task_id,
                timeout_seconds=self._cancel_timeout_seconds,
            )
            self._condition.notify_all()

    def _refresh_positions_locked(self) -> None:
        self.task_store.update_queue_positions(tuple(self._queue))

    @staticmethod
    def _remove_incomplete_artifacts(record: TaskRecord) -> None:
        for name in ("inputs", "outputs", "runtime"):
            target = record.task_dir / name
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()

    def _run_loop(self) -> None:
        while True:
            with self._condition:
                while self._accepting and not self._queue:
                    self._condition.wait(timeout=1.0)
                    self.task_store.cleanup_expired()
                    if self._accepting and not self._queue:
                        self.worker_group.heartbeat()
                if not self._accepting and not self._queue:
                    return
                task_id = self._queue.popleft()
                self._active_task_id = task_id
                self._refresh_positions_locked()
            record = self.task_store.transition(task_id, TaskStatus.RUNNING)
            _log_event("request_started", request_id=task_id)
            self.task_store.update_progress(
                task_id, phase=TaskPhase.PREPARING, progress=1
            )
            payload = {
                "sampling": dict(record.request_payload),
                "video_input_path": str(record.video_input_path),
                "mask_input_path": str(record.mask_input_path),
                "bbox_input_path": (
                    str(record.bbox_input_path) if record.bbox_input_path else None
                ),
                "output_dir": str(record.task_dir / "outputs"),
                "output_file_name": "result.partial.mp4",
                "runtime_workdir": str(record.task_dir / "runtime"),
                "warmup_steps": self._warmup_steps,
            }
            self._warmup_steps = None
            started = time.perf_counter()
            try:
                if record.cancellation_requested:
                    raise RequestCancelled("request cancelled before execution")
                result = self.worker_group.execute(task_id, payload)
                output_path = Path(result.extra["output_file_path"]).resolve()
                partial = (record.task_dir / "outputs" / "result.partial.mp4").resolve()
                published = (record.task_dir / "outputs" / "result.mp4").resolve()
                if output_path != partial or not output_path.is_file():
                    raise RuntimeError(
                        "worker did not publish the expected service output"
                    )
                if not self.task_store.begin_result_publication(task_id):
                    raise RequestCancelled(
                        "request cancelled before output publication"
                    )
                os.replace(partial, published)
                storage_started = time.perf_counter()
                storage_outcome = self.result_storage.publish(task_id, published)
                storage_seconds = time.perf_counter() - storage_started
                self.task_store.set_result_storage(
                    task_id,
                    mode=storage_outcome.mode,
                    local_path=storage_outcome.local_path,
                    url=storage_outcome.url,
                    fallback=storage_outcome.fallback,
                )
                _log_event(
                    "result_published",
                    request_id=task_id,
                    storage_mode=storage_outcome.mode,
                    storage_fallback=storage_outcome.fallback,
                )
                pure_timing = build_ltx095_pure_timing_payload(result.metrics)
                record.metrics = {
                    "process_seconds": time.perf_counter() - started,
                    "queue_wait_seconds": (
                        float(record.started_at - record.created_at)
                        if record.started_at is not None
                        else None
                    ),
                    "pipeline_seconds": (
                        float(result.metrics.total_duration_ms) / 1000.0
                        if result.metrics is not None
                        else None
                    ),
                    "pure_inference_seconds": pure_timing["pure_inference_seconds"],
                    "pure_inference_stage_breakdown_ms": pure_timing[
                        "pure_inference_stage_breakdown_ms"
                    ],
                    "stage_breakdown_ms": (
                        dict(result.metrics.stages)
                        if result.metrics is not None
                        else {}
                    ),
                    "stage_counts": (
                        dict(result.metrics.stage_counts)
                        if result.metrics is not None
                        else {}
                    ),
                    "operation_counts": (
                        dict(result.metrics.operation_counts)
                        if result.metrics is not None
                        else {}
                    ),
                    "transformer_cache_history": list(
                        result.extra.get("transformer_cache_history", [])
                    ),
                    "torch_compile": result.extra.get("torch_compile"),
                    "peak_memory": result.extra.get("service_peak_memory"),
                    "result_storage_seconds": storage_seconds,
                    "result_storage_mode": storage_outcome.mode,
                    "result_storage_fallback": storage_outcome.fallback,
                    "result_storage_error_type": storage_outcome.error_type,
                }
                self.task_store.transition(task_id, TaskStatus.COMPLETED)
                _log_event(
                    "request_completed",
                    request_id=task_id,
                    process_seconds=record.metrics["process_seconds"],
                    pipeline_seconds=record.metrics["pipeline_seconds"],
                    pure_inference_seconds=record.metrics["pure_inference_seconds"],
                    peak_memory=record.metrics["peak_memory"],
                )
            except RequestCancelled:
                cancelled_phase = record.phase.value
                self._remove_incomplete_artifacts(record)
                cancellation_latency = (
                    time.time() - record.cancellation_requested_at
                    if record.cancellation_requested_at is not None
                    else None
                )
                record.metrics = {
                    "cancellation_latency_seconds": cancellation_latency,
                }
                self.task_store.transition(task_id, TaskStatus.CANCELLED)
                _log_event(
                    "request_cancelled",
                    request_id=task_id,
                    phase=cancelled_phase,
                    cancellation_latency_seconds=cancellation_latency,
                )
            except BaseException as error:
                logger.exception("service request %s failed", task_id)
                self._remove_incomplete_artifacts(record)
                fatal_error = self.worker_group.fatal_error
                self.task_store.transition(
                    task_id,
                    TaskStatus.FAILED,
                    error=TaskError(
                        code=(
                            "worker_group_fatal" if fatal_error else "request_failed"
                        ),
                        message=(
                            "worker group is unavailable"
                            if fatal_error
                            else "request execution failed"
                        ),
                        phase=record.phase.value,
                    ),
                )
                _log_event(
                    "request_failed",
                    request_id=task_id,
                    fatal=bool(fatal_error),
                    error_type=type(error).__name__,
                )
                if fatal_error:
                    with self._condition:
                        self._accepting = False
                        queued = tuple(self._queue)
                        self._queue.clear()
                        for queued_task_id in queued:
                            queued_record = self.task_store.get(queued_task_id)
                            self.task_store.request_cancellation(queued_task_id)
                            self.task_store.transition(
                                queued_task_id, TaskStatus.CANCELLED
                            )
                            self._remove_incomplete_artifacts(queued_record)
                        self._refresh_positions_locked()
            finally:
                with self._condition:
                    self._active_task_id = None
                    self._condition.notify_all()

    def stats(self) -> dict[str, object]:
        counts = self.task_store.status_counts()
        with self._condition:
            return {
                "counts": counts,
                "queue_depth": len(self._queue),
                "active_task_id": self._active_task_id,
                "accepting": self._accepting,
                "result_storage": self.result_storage.summary(),
                "latest_completed": self.task_store.latest_completed_metrics(),
            }

    def health_snapshot(self) -> dict[str, object]:
        return self.worker_group.health_snapshot()

    def shutdown(self, *, timeout: float) -> None:
        with self._condition:
            self._accepting = False
            queued = tuple(self._queue)
            self._queue.clear()
            for task_id in queued:
                record = self.task_store.get(task_id)
                self.task_store.request_cancellation(task_id)
                self.task_store.transition(task_id, TaskStatus.CANCELLED)
                self._remove_incomplete_artifacts(record)
            if self._active_task_id is not None:
                task_id = self._active_task_id
                try:
                    self.task_store.request_cancellation(task_id)
                except ServiceError as error:
                    if error.code != "task_finalizing":
                        raise
                    _log_event(
                        "shutdown_waiting_for_result_publication",
                        request_id=task_id,
                    )
                else:
                    self.worker_group.request_cancellation(task_id)
                    _log_event(
                        "cancellation_requested", request_id=task_id, source="shutdown"
                    )
            self._condition.notify_all()
        self._thread.join(timeout=float(timeout))
        if self._thread.is_alive():
            raise RuntimeError("service scheduler did not stop before timeout")


__all__ = ("ServiceScheduler",)
