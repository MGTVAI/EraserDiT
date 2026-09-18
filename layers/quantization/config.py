"""Configuration contract for the model-agnostic FP8 W8A8 Linear backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import torch


class FP8LinearConfigurationError(ValueError):
    """Raised when an FP8 Linear configuration violates the frozen contract."""


class FP8LinearBackend(str, Enum):
    DISABLED = "disabled"
    AUTO = "auto"
    NATIVE_SCALED_MM = "native_scaled_mm"
    TORCHAO = "torchao"


class FP8ActivationQuantization(str, Enum):
    EAGER = "eager"
    TRITON_PER_ROW = "triton_per_row"


class FP8LinearGranularity(str, Enum):
    PER_ROW = "per_row"


def _normalize_enum(value: str | Enum, enum_cls: type[Enum], field: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value).lower())
    except ValueError as exc:
        choices = ", ".join(item.value for item in enum_cls)
        raise FP8LinearConfigurationError(
            f"{field} must be one of: {choices}"
        ) from exc


@dataclass(frozen=True)
class FP8LinearConfig:
    backend: FP8LinearBackend = FP8LinearBackend.AUTO
    weight_dtype: torch.dtype = torch.float8_e4m3fn
    activation_dtype: torch.dtype = torch.float8_e4m3fn
    granularity: FP8LinearGranularity = FP8LinearGranularity.PER_ROW
    activation_quantization: FP8ActivationQuantization = FP8ActivationQuantization.EAGER
    output_dtype: torch.dtype = torch.bfloat16
    scale_dtype: torch.dtype = torch.float32
    use_fast_accum: bool = False
    strict: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend",
            _normalize_enum(self.backend, FP8LinearBackend, "backend"),
        )
        object.__setattr__(
            self,
            "granularity",
            _normalize_enum(
                self.granularity,
                FP8LinearGranularity,
                "granularity",
            ),
        )
        object.__setattr__(
            self,
            "activation_quantization",
            _normalize_enum(
                self.activation_quantization,
                FP8ActivationQuantization,
                "activation_quantization",
            ),
        )
        if self.weight_dtype != torch.float8_e4m3fn:
            raise FP8LinearConfigurationError(
                "weight_dtype must be torch.float8_e4m3fn"
            )
        if self.activation_dtype != torch.float8_e4m3fn:
            raise FP8LinearConfigurationError(
                "activation_dtype must be torch.float8_e4m3fn"
            )
        if self.output_dtype != torch.bfloat16:
            raise FP8LinearConfigurationError(
                "output_dtype must be torch.bfloat16"
            )
        if self.scale_dtype != torch.float32:
            raise FP8LinearConfigurationError(
                "scale_dtype must be torch.float32"
            )
        if not isinstance(self.use_fast_accum, bool):
            raise FP8LinearConfigurationError(
                "use_fast_accum must be boolean"
            )
        if not isinstance(self.strict, bool):
            raise FP8LinearConfigurationError("strict must be boolean")

    @classmethod
    def from_values(
        cls,
        *,
        backend: str = "auto",
        granularity: str = "per_row",
        activation_quantization: str = "eager",
        use_fast_accum: bool = False,
        strict: bool = True,
    ) -> "FP8LinearConfig":
        return cls(
            backend=FP8LinearBackend(backend.lower()),
            granularity=FP8LinearGranularity(granularity.lower()),
            activation_quantization=FP8ActivationQuantization(
                activation_quantization.lower()
            ),
            use_fast_accum=use_fast_accum,
            strict=strict,
        )

    @property
    def enabled(self) -> bool:
        return self.backend is not FP8LinearBackend.DISABLED

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "backend": self.backend.value,
                "weight_dtype": str(self.weight_dtype).removeprefix("torch."),
                "activation_dtype": str(
                    self.activation_dtype
                ).removeprefix("torch."),
                "granularity": self.granularity.value,
                "activation_quantization": self.activation_quantization.value,
                "output_dtype": str(self.output_dtype).removeprefix("torch."),
                "scale_dtype": str(self.scale_dtype).removeprefix("torch."),
                "zero_point": None,
            }
        )
        return payload
