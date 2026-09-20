"""Service-owned artifact creation and deployment-controlled input paths."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

from entrypoints.server.task import ServiceError


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_allowed_input(path: str, allowed_roots: tuple[str, ...]) -> Path:
    if not allowed_roots:
        raise ServiceError(
            "local_paths_disabled",
            "local input paths are disabled for this deployment",
            status_code=422,
        )
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ServiceError("invalid_input_path", "input path is not a file", status_code=422)
    roots = tuple(Path(root).resolve(strict=True) for root in allowed_roots)
    if not any(_is_relative_to(resolved, root) for root in roots):
        raise ServiceError(
            "input_path_not_allowed",
            "input path is outside the deployment allowlist",
            status_code=422,
        )
    return resolved


class TaskArtifactManager:
    def __init__(self, task_root: str | Path, *, max_upload_bytes: int) -> None:
        self.root = Path(task_root).expanduser().resolve()
        self.tasks_root = self.root / "tasks"
        self.staging_root = self.root / "staging"
        self.max_upload_bytes = int(max_upload_bytes)
        self.tasks_root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)

    def cleanup_orphan_staging(self) -> None:
        for child in self.staging_root.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)

    def create_task_directory(self, task_id: str) -> Path:
        if not task_id or any(character not in "0123456789abcdef" for character in task_id):
            raise ServiceError("invalid_task_id", "invalid generated task id", status_code=500)
        staging = self.staging_root / f"{task_id}-{uuid.uuid4().hex}"
        destination = self.tasks_root / task_id
        staging.mkdir(mode=0o700)
        try:
            (staging / "inputs").mkdir()
            (staging / "outputs").mkdir()
            os.replace(staging, destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return destination.resolve()

    def copy_upload(self, source: BinaryIO, destination: Path) -> int:
        destination = destination.resolve()
        if not _is_relative_to(destination, self.tasks_root):
            raise ServiceError("invalid_task_directory", "upload target is not service-owned", status_code=500)
        total = 0
        try:
            with destination.open("xb") as output:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > self.max_upload_bytes:
                        raise ServiceError(
                            "upload_too_large",
                            "uploaded file exceeds the configured byte limit",
                            status_code=429,
                        )
                    output.write(chunk)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        return total


__all__ = ("TaskArtifactManager", "resolve_allowed_input")
