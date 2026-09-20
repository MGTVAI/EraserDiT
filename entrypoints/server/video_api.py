"""OpenAI-style asynchronous video routes for the local erase capability."""

from __future__ import annotations

import shutil
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
from pydantic import ValidationError

from config.service_args import ServiceArgs
from entrypoints.server.protocol import (
    DeletedTaskResponse,
    VideoListResponse,
    VideoResponse,
)
from entrypoints.server.artifacts import (
    TaskArtifactManager,
    resolve_allowed_input,
)
from entrypoints.server.task import ServiceError, TaskRecord, TaskStatus


def create_video_router(
    *,
    service_args: ServiceArgs,
    scheduler: Any,
    task_store: Any,
    artifact_manager: TaskArtifactManager,
    request_schema_cls: Any,
    multipart_schema_cls: Any,
    path_fields: tuple[str, ...] = ("video_path", "mask_path", "bbox_path"),
) -> APIRouter:
    router = APIRouter(prefix="/v1/videos")

    @router.post("", status_code=202, response_model=VideoResponse)
    async def create_video(request: Request) -> dict[str, object]:
        if not request.app.state.ready:
            raise ServiceError(
                "service_not_ready", "service is not ready", status_code=503
            )
        task_id = uuid.uuid4().hex
        task_dir = artifact_manager.create_task_directory(task_id)
        registered = False
        content_type = request.headers.get("content-type", "").lower()
        try:
            if content_type.startswith("application/json"):
                try:
                    parsed = request_schema_cls.model_validate_json(
                        await request.body()
                    )
                except ValidationError as error:
                    raise ServiceError(
                        "invalid_request", str(error), status_code=422
                    ) from error
                video_path = resolve_allowed_input(
                    parsed.video_path, service_args.input_allowed_roots
                )
                mask_path = resolve_allowed_input(
                    parsed.mask_path, service_args.input_allowed_roots
                )
                bbox_path = (
                    resolve_allowed_input(
                        parsed.bbox_path, service_args.input_allowed_roots
                    )
                    if parsed.bbox_path
                    else None
                )
                sampling = parsed.model_dump(exclude=set(path_fields))
            elif content_type.startswith("multipart/form-data"):
                form = await request.form()
                unknown = set(form.keys()) - {
                    "video",
                    "mask",
                    "parameters",
                    "bbox_path",
                }
                if unknown:
                    raise ServiceError(
                        "invalid_request",
                        f"unknown multipart fields: {sorted(unknown)}",
                        status_code=422,
                    )
                video = form.get("video")
                mask = form.get("mask")
                if (
                    video is None
                    or mask is None
                    or not hasattr(video, "file")
                    or not hasattr(mask, "file")
                ):
                    raise ServiceError(
                        "invalid_request",
                        "multipart requires video and mask files",
                        status_code=422,
                    )
                raw_parameters = form.get("parameters", "{}")
                if not isinstance(raw_parameters, str):
                    raise ServiceError(
                        "invalid_request",
                        "parameters must be JSON text",
                        status_code=422,
                    )
                try:
                    parsed_params = multipart_schema_cls.model_validate_json(
                        raw_parameters
                    )
                except ValidationError as error:
                    raise ServiceError(
                        "invalid_request", str(error), status_code=422
                    ) from error
                video_path = task_dir / "inputs" / "video.mp4"
                mask_path = task_dir / "inputs" / "mask.mp4"
                video_bytes = artifact_manager.copy_upload(video.file, video_path)
                mask_bytes = artifact_manager.copy_upload(mask.file, mask_path)
                if video_bytes + mask_bytes > service_args.max_upload_bytes:
                    raise ServiceError(
                        "upload_too_large",
                        "combined uploads exceed the configured byte limit",
                        status_code=429,
                    )
                raw_bbox = form.get("bbox_path")
                bbox_path = (
                    resolve_allowed_input(raw_bbox, service_args.input_allowed_roots)
                    if isinstance(raw_bbox, str) and raw_bbox
                    else None
                )
                sampling = parsed_params.model_dump()
            else:
                raise ServiceError(
                    "unsupported_content_type",
                    "content type must be application/json or multipart/form-data",
                    status_code=422,
                )
            record = TaskRecord(
                task_id=task_id,
                request_payload=sampling,
                task_dir=task_dir,
                video_input_path=video_path,
                mask_input_path=mask_path,
                bbox_input_path=bbox_path,
                storage_mode=service_args.result_storage_mode,
            )
            scheduler.submit(record)
            registered = True
            return task_store.snapshot(task_id)
        except BaseException:
            if not registered:
                shutil.rmtree(task_dir, ignore_errors=True)
            raise

    @router.get("", response_model=VideoListResponse)
    async def list_videos(
        after: str | None = None,
        limit: int = 20,
        order: str = "desc",
    ) -> dict[str, object]:
        records = task_store.list_records(after=after, limit=limit, order=order)
        return {
            "object": "list",
            "data": [record.public_dict() for record in records],
            "has_more": len(records) == limit,
        }

    @router.get("/{task_id}", response_model=VideoResponse)
    async def get_video(task_id: str) -> dict[str, object]:
        return task_store.snapshot(task_id)

    @router.delete("/{task_id}", response_model=VideoResponse | DeletedTaskResponse)
    async def delete_video(task_id: str) -> dict[str, object]:
        _, response = scheduler.cancel_or_purge(task_id)
        return response

    @router.get("/{task_id}/content")
    async def video_content(task_id: str) -> FileResponse:
        record = task_store.get(task_id)
        if record.status is not TaskStatus.COMPLETED:
            raise ServiceError(
                "result_not_ready", "task result is not ready", status_code=409
            )
        if record.result_url is not None:
            raise ServiceError(
                "result_remote",
                "task result is available through the response url",
                status_code=409,
            )
        if record.output_path is None or not record.output_path.is_file():
            raise ServiceError(
                "result_missing", "task result is missing", status_code=410
            )
        return FileResponse(
            record.output_path,
            media_type="video/mp4",
            filename=f"{task_id}.mp4",
        )

    return router


__all__ = ("create_video_router",)
