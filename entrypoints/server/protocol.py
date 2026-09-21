"""Strict public request and response contracts for the MGErase video API."""

from __future__ import annotations

from typing import Annotated, Any, Literal

# Request schemas are model-provided service contracts; re-exported here for
# callers that still import them from the protocol module.
from config.service_contracts.ltx095 import (  # noqa: F401
    DEFAULT_NEGATIVE_PROMPT,
    LocalVideoCreateRequest,
    MultipartVideoParameters,
    VideoSamplingRequest,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator



class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    message: str
    phase: str | None = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    error: ErrorDetail


class VideoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["video"] = "video"
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    phase: Literal["queued", "preparing", "processing", "finalizing", "terminal"]
    progress: Annotated[int, Field(ge=0, le=100)]
    created_at: int
    started_at: int | None = None
    completed_at: int | None = None
    expires_at: int | None = None
    queue_position: int | None = None
    object_index: int | None = None
    object_count: int | None = None
    window_index: int | None = None
    window_count: int | None = None
    url: str | None = None
    content_url: str | None = None
    storage_mode: Literal["local", "s3"] = "local"
    storage_fallback: bool = False
    error: ErrorDetail | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_result_location(self) -> "VideoResponse":
        if self.url is not None and self.content_url is not None:
            raise ValueError("url and content_url must be mutually exclusive")
        return self


class VideoListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object: Literal["list"] = "list"
    data: list[VideoResponse]
    has_more: bool
    first_id: str | None = None
    last_id: str | None = None


class DeletedTaskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    deleted: Literal[True] = True
    remote_result_deleted: Literal[False] = False


class ModelCard(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: Literal["mgerase"] = "mgerase"
    capability: str


class ModelListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object: Literal["list"] = "list"
    data: list[ModelCard]


__all__ = (
    "DEFAULT_NEGATIVE_PROMPT",
    "DeletedTaskResponse",
    "ErrorResponse",
    "LocalVideoCreateRequest",
    "ModelCard",
    "ModelListResponse",
    "MultipartVideoParameters",
    "VideoListResponse",
    "VideoResponse",
    "VideoSamplingRequest",
)
