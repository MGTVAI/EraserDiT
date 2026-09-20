"""Project-owned LTX095 TeaCache policy and request validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from config.transformer_cache import TransformerCacheMode

LTX095_TEACACHE_MODEL_IDENTITY = "ltxvideo/erase/checkpoint-206K"
LTX095_TEACACHE_COEFFICIENT_POLICY = "ltx095_checkpoint_206k"
LTX095_TEACACHE_DEFAULT_THRESHOLD = 0.03


@dataclass(frozen=True)
class TeaCacheCoefficientRecord:
    fit_length: int
    complexity: int
    calibration_size: int
    coefficients: tuple[float, ...]


@dataclass(frozen=True)
class TeaCacheCoefficientSelection:
    policy: str
    model_identity: str
    requested_global_sequence_length: int
    selected_fit_length: int
    fit_length_distance: int
    complexity: int
    calibration_size: int
    coefficients: tuple[float, ...]


_LTX095_COEFFICIENT_RECORDS = (
    TeaCacheCoefficientRecord(
        fit_length=32640,
        complexity=4,
        calibration_size=200,
        coefficients=(
            70356.53677060452,
            -7439.580188855337,
            231.59388228079706,
            -1.199310499953305,
            0.013993239220206724,
        ),
    ),
    TeaCacheCoefficientRecord(
        fit_length=14720,
        complexity=4,
        calibration_size=200,
        coefficients=(
            74483.94264897682,
            -7870.101280007631,
            245.58472780172286,
            -1.3426960354405535,
            0.013830190007859787,
        ),
    ),
)


@dataclass(frozen=True)
class TeaCacheParams:
    supported_coefficient_policies: ClassVar[tuple[str, ...]] = (LTX095_TEACACHE_COEFFICIENT_POLICY,)
    enabled: bool
    threshold: float = LTX095_TEACACHE_DEFAULT_THRESHOLD
    max_consecutive_skip: int = 1
    min_skip_step: int = 2
    end_guard_steps: int = 1
    coefficient_policy: str = LTX095_TEACACHE_COEFFICIENT_POLICY
    calibrate: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("TeaCache enabled must be a bool")
        if isinstance(self.threshold, bool) or not isinstance(
            self.threshold, (int, float)
        ):
            raise TypeError("TeaCache threshold must be a finite float")
        threshold = float(self.threshold)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("TeaCache threshold must be finite and > 0")
        object.__setattr__(self, "threshold", threshold)
        for name in ("max_consecutive_skip", "min_skip_step", "end_guard_steps"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"TeaCache {name} must be a non-bool int")
        if self.max_consecutive_skip < 1:
            raise ValueError("TeaCache max_consecutive_skip must be >= 1")
        if self.min_skip_step < 0:
            raise ValueError("TeaCache min_skip_step must be >= 0")
        if self.end_guard_steps < 0:
            raise ValueError("TeaCache end_guard_steps must be >= 0")
        if self.coefficient_policy not in self.supported_coefficient_policies:
            raise ValueError(
                "unsupported TeaCache coefficient_policy="
                f"{self.coefficient_policy!r}"
            )
        if not isinstance(self.calibrate, bool):
            raise TypeError("TeaCache calibrate must be a bool")


def resolve_ltx095_teacache_params(
    *,
    mode: TransformerCacheMode,
    threshold: object,
    max_consecutive_skip: object,
    calibrate: object,
    coefficient_policy: object,
) -> TeaCacheParams:
    if not isinstance(mode, TransformerCacheMode):
        raise TypeError("mode must be a TransformerCacheMode")
    return TeaCacheParams(
        enabled=mode is TransformerCacheMode.TEACACHE,
        threshold=threshold,
        max_consecutive_skip=max_consecutive_skip,
        min_skip_step=2,
        end_guard_steps=1,
        coefficient_policy=str(coefficient_policy),
        calibrate=calibrate,
    )


def select_ltx095_teacache_coefficients(
    global_sequence_length: int,
    *,
    policy: str = LTX095_TEACACHE_COEFFICIENT_POLICY,
) -> TeaCacheCoefficientSelection:
    if policy != LTX095_TEACACHE_COEFFICIENT_POLICY:
        raise ValueError(f"unsupported TeaCache coefficient policy: {policy!r}")
    if type(global_sequence_length) is not int or global_sequence_length <= 0:
        raise ValueError("global_sequence_length must be a positive non-bool int")
    record = min(
        _LTX095_COEFFICIENT_RECORDS,
        key=lambda item: (
            abs(item.fit_length - global_sequence_length),
            -item.fit_length,
        ),
    )
    return TeaCacheCoefficientSelection(
        policy=policy,
        model_identity=LTX095_TEACACHE_MODEL_IDENTITY,
        requested_global_sequence_length=global_sequence_length,
        selected_fit_length=record.fit_length,
        fit_length_distance=abs(record.fit_length - global_sequence_length),
        complexity=record.complexity,
        calibration_size=record.calibration_size,
        coefficients=record.coefficients,
    )


__all__ = (
    "LTX095_TEACACHE_COEFFICIENT_POLICY",
    "LTX095_TEACACHE_DEFAULT_THRESHOLD",
    "LTX095_TEACACHE_MODEL_IDENTITY",
    "TeaCacheCoefficientSelection",
    "TeaCacheParams",
    "resolve_ltx095_teacache_params",
    "select_ltx095_teacache_coefficients",
)
