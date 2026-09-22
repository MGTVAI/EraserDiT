"""Model-discovery routes for the local EraserDiT service."""

from __future__ import annotations

import time
from collections.abc import Mapping

from fastapi import APIRouter

from entrypoints.server.protocol import ModelCard, ModelListResponse
from entrypoints.server.task import ServiceError


def create_common_router(model_summary: Mapping[str, object]) -> APIRouter:
    model_id = str(model_summary["id"])
    created = int(time.time())
    card = ModelCard(
        id=model_id,
        created=created,
        capability=str(model_summary["capability"]),
    )
    router = APIRouter(prefix="/v1")

    @router.get("/models", response_model=ModelListResponse)
    async def list_models() -> ModelListResponse:
        return ModelListResponse(data=[card])

    @router.get("/models/{requested_model_id}", response_model=ModelCard)
    async def get_model(requested_model_id: str) -> ModelCard:
        if requested_model_id != model_id:
            raise ServiceError(
                "model_not_found",
                f"model {requested_model_id} was not found",
                status_code=404,
            )
        return card

    return router


__all__ = ("create_common_router",)
