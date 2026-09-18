"""Small performance logging helpers for the minimal runtime."""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class RequestMetrics:
    """Minimal per-request timings used by the stage/executor skeleton."""

    request_id: str
    stages: dict[str, float] = field(default_factory=dict)
    stage_counts: dict[str, int] = field(default_factory=dict)
    operation_counts: dict[str, int] = field(default_factory=dict)
    steps: list[float] = field(default_factory=list)
    total_duration_ms: float = 0.0

    def record_stage(self, stage_name: str, duration_s: float) -> None:
        duration_ms = duration_s * 1000.0
        self.stages[stage_name] = self.stages.get(stage_name, 0.0) + duration_ms
        self.stage_counts[stage_name] = self.stage_counts.get(stage_name, 0) + 1

    def record_step(self, duration_s: float) -> None:
        self.steps.append(duration_s * 1000.0)

    def record_operation(self, operation_name: str) -> None:
        self.operation_counts[operation_name] = (
            self.operation_counts.get(operation_name, 0) + 1
        )

    def ensure_operation(self, operation_name: str) -> None:
        """Expose an audited operation even when its expected count is zero."""
        self.operation_counts.setdefault(operation_name, 0)


class StageProfiler(contextlib.AbstractContextManager):
    """A no-frills profiler that records stage timing into RequestMetrics."""

    def __init__(
        self,
        stage_name: str,
        logger,
        metrics: RequestMetrics | None,
        log_stage_start_end: bool = False,
        perf_dump_path_provided: bool = False,
    ) -> None:
        self.stage_name = stage_name
        self.logger = logger
        self.metrics = metrics
        self.log_stage_start_end = log_stage_start_end
        self.perf_dump_path_provided = perf_dump_path_provided
        self._start = 0.0

    def __enter__(self):
        if self.log_stage_start_end:
            self.logger.info("[%s] started", self.stage_name)
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        duration = time.perf_counter() - self._start
        if self.metrics is not None:
            self.metrics.record_stage(self.stage_name, duration)
        if self.log_stage_start_end or self.perf_dump_path_provided:
            self.logger.info("[%s] finished in %.3fs", self.stage_name, duration)
        return False


class PerformanceLogger:
    """Tiny benchmark dump helper kept for CLI compatibility."""

    @staticmethod
    def dump_benchmark_report(
        file_path: str,
        metrics: RequestMetrics,
        meta: dict[str, Any] | None = None,
        tag: str = "benchmark_dump",
    ) -> None:
        report = {
            "request_id": metrics.request_id,
            "tag": tag,
            "total_duration_ms": metrics.total_duration_ms,
            "stages": metrics.stages,
            "stage_counts": metrics.stage_counts,
            "operation_counts": metrics.operation_counts,
            "steps": metrics.steps,
            "meta": meta or {},
        }
        abs_path = os.path.abspath(file_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Metrics dumped to %s", abs_path)

    @staticmethod
    def log_request_summary(
        metrics: RequestMetrics,
        tag: str = "total_inference_time",
    ) -> None:
        logger.info(
            "Request %s summary [%s]: total=%.2fms, stages=%s",
            metrics.request_id,
            tag,
            metrics.total_duration_ms,
            asdict(metrics)["stages"],
        )
