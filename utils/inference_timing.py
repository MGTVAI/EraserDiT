"""Helpers for writing structured inference timing sidecars."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from utils.perf_logger import RequestMetrics

PURE_TIMING_ENV_VAR = "MGERASE_PURE_TIMING_JSON"
DIAGNOSTIC_TIMING_ENV_VAR = "MGERASE_DIAGNOSTIC_TIMING"

PURE_INFERENCE_STAGE_NAMES = (
    "LTX095EraseConditionEncodingStage",
    "LTX095EraseLatentPreparationStage",
    "LTX095EraseTimestepPreparationStage",
    "LTX095EraseDenoisingStage",
    "LTX095EraseDecodingStage",
)


def diagnostic_timing_enabled() -> bool:
    return os.environ.get(DIAGNOSTIC_TIMING_ENV_VAR) == "1"


def record_diagnostic_stage(
    metrics: RequestMetrics | None,
    stage_name: str,
    duration_s: float,
) -> None:
    if diagnostic_timing_enabled() and metrics is not None:
        metrics.record_stage(stage_name, duration_s)


def build_ltx095_pure_timing_payload(
    metrics: RequestMetrics | None,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra_payload = dict(extra or {})
    if metrics is None:
        return {
            "request_id": None,
            "pure_inference_seconds": None,
            "pure_inference_stage_names": list(PURE_INFERENCE_STAGE_NAMES),
            "pure_inference_stage_breakdown_ms": {},
            "pure_inference_stage_counts": {},
            "pure_inference_operation_counts": {},
            "pipeline_total_seconds": None,
            "extra": extra_payload,
        }

    stage_breakdown_ms = {
        stage_name: float(metrics.stages.get(stage_name, 0.0))
        for stage_name in PURE_INFERENCE_STAGE_NAMES
        if stage_name in metrics.stages
    }
    stage_counts = {
        stage_name: int(metrics.stage_counts.get(stage_name, 0))
        for stage_name in PURE_INFERENCE_STAGE_NAMES
        if stage_name in metrics.stage_counts
    }
    pure_inference_ms = sum(stage_breakdown_ms.values())
    payload = {
        "request_id": metrics.request_id,
        "pure_inference_seconds": pure_inference_ms / 1000.0,
        "pure_inference_stage_names": list(PURE_INFERENCE_STAGE_NAMES),
        "pure_inference_stage_breakdown_ms": stage_breakdown_ms,
        "pure_inference_stage_counts": stage_counts,
        "pure_inference_operation_counts": dict(metrics.operation_counts),
        "pipeline_total_seconds": (
            float(metrics.total_duration_ms) / 1000.0
            if metrics.total_duration_ms > 0.0
            else None
        ),
        "extra": extra_payload,
    }
    if diagnostic_timing_enabled():
        payload["diagnostic_stage_breakdown_ms"] = {
            name: float(duration_ms)
            for name, duration_ms in metrics.stages.items()
        }
        payload["diagnostic_stage_counts"] = {
            name: int(count) for name, count in metrics.stage_counts.items()
        }
    return payload


def maybe_write_pure_timing_payload(payload: dict[str, Any]) -> str | None:
    output_path = os.environ.get(PURE_TIMING_ENV_VAR)
    if not output_path:
        return None
    resolved = Path(output_path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=True)
    return str(resolved)
