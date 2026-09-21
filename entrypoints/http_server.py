"""FastAPI application composition for the resident MGErase service."""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from config.service_args import ServiceArgs
from entrypoints.server.common_api import create_common_router
from entrypoints.server.video_api import create_video_router
from entrypoints.server.artifacts import TaskArtifactManager
from entrypoints.server.task import ServiceError


def _error_response(error: ServiceError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={"error": {"code": error.code, "message": error.message}},
    )


def create_http_server_app(
    *,
    service_args: ServiceArgs,
    scheduler: Any,
    task_store: Any,
    artifact_manager: TaskArtifactManager,
    server_summary: dict[str, object],
    model_summary: dict[str, object],
    service_contract: Any,
    effective_acceleration: Any = None,
) -> FastAPI:
    capability = str(model_summary.get("capability", "video_erase"))
    app = FastAPI(title=f"MGErase {capability} Service", version="1")
    app.state.ready = True
    app.state.started_at = time.time()

    @app.exception_handler(ServiceError)
    async def handle_service_error(_: Request, error: ServiceError) -> JSONResponse:
        return _error_response(error)

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(
        _: Request, error: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_request", "message": str(error)}},
        )

    @app.exception_handler(HTTPException)
    async def handle_http_error(_: Request, error: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            headers=error.headers,
            content={"error": {"code": f"http_{error.status_code}", "message": str(error.detail)}},
        )

    def readiness() -> tuple[bool, dict[str, object]]:
        worker_health = scheduler.health_snapshot()
        heartbeat_at = float(worker_health["last_all_rank_heartbeat_at"])
        heartbeat_fresh = (
            time.time() - heartbeat_at <= service_args.health_timeout_seconds
        )
        ready = bool(app.state.ready and worker_health["ready"] and heartbeat_fresh)
        return ready, worker_health

    app.state.readiness = readiness

    @app.get("/health")
    @app.get("/ready")
    async def health() -> JSONResponse:
        ready, worker_health = readiness()
        return JSONResponse(
            status_code=(200 if ready else 503),
            content={
                "status": "ready" if ready else "not_ready",
                "worker_group": worker_health,
                "uptime_seconds": time.time() - app.state.started_at,
            },
        )

    @app.get("/get_server_info")
    @app.get("/server_info")
    async def server_info() -> dict[str, object]:
        return {
            "service": capability,
            "api_version": "v1",
            "limits": {
                "max_queued_tasks": service_args.max_queued_tasks,
                "max_upload_bytes": service_args.max_upload_bytes,
            },
            "startup_config": server_summary,
            "effective_acceleration": (
                effective_acceleration() if callable(effective_acceleration) else {}
            ),
        }

    @app.get("/get_model_info")
    @app.get("/model_info")
    async def model_info() -> dict[str, object]:
        return model_summary

    @app.get("/stats")
    async def stats() -> dict[str, object]:
        return scheduler.stats()

    app.include_router(
        create_video_router(
            service_args=service_args,
            scheduler=scheduler,
            task_store=task_store,
            artifact_manager=artifact_manager,
            request_schema_cls=service_contract.request_schema_cls,
            multipart_schema_cls=service_contract.multipart_schema_cls,
            path_fields=service_contract.path_fields,
            model_id=str(model_summary["id"]),
        )
    )
    app.include_router(create_common_router(model_summary))
    return app


__all__ = ("create_http_server_app",)
