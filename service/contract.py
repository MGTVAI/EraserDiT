"""Model-provided service contract.

The service skeleton must not hard-code a model: the request schema, the
sampling-parameter builder and the capability identifier all come from the
pipeline the server was started with (``vibe/requirements.md`` 服务端).  Adding a
model therefore means adding a contract module, not editing ``video_api`` /
``worker`` / ``http_server``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

__all__ = ["PipelineServiceContract", "resolve_service_contract"]


@dataclass(frozen=True)
class PipelineServiceContract:
    """Everything the service layer needs to know about one model."""

    capability: str
    # Request schema for the local-path create call (JSON body).
    request_schema_cls: type[BaseModel]
    # Request schema for the ``parameters`` field of a multipart create call.
    multipart_schema_cls: type[BaseModel]
    # Builds the model's sampling parameters from a task payload.
    build_sampling_params: Callable[..., Any]
    request_modes: tuple[str, ...] = ("multipart_upload", "controlled_local_paths")
    # Fields on the create request that carry paths rather than sampling values.
    path_fields: tuple[str, ...] = ("video_path", "mask_path", "bbox_path")
    # Per-request validation beyond the schema (e.g. cache/compile combinations).
    validate_request: Callable[..., None] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def sampling_payload(self, parsed: BaseModel) -> dict[str, Any]:
        return parsed.model_dump(exclude=set(self.path_fields))


def resolve_service_contract(pipeline_name: str | None) -> PipelineServiceContract:
    """Resolve the contract owned by the named pipeline."""
    from pipelines.registry import PipelineRegistry

    pipeline_cls, _ = PipelineRegistry.resolve(pipeline_name)
    contract = getattr(pipeline_cls, "service_contract", None)
    if contract is None:
        raise ValueError(
            f"pipeline {pipeline_cls.__name__} does not declare a service contract"
        )
    return contract
