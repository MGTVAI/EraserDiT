"""Result publication backends for the MGErase video API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import quote

from config.service_args import ServiceArgs
from utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class ResultStorageOutcome:
    mode: Literal["local", "s3"]
    local_path: Path | None
    url: str | None
    fallback: bool = False
    error_type: str | None = None


class ResultStorage(Protocol):
    mode: str

    def publish(self, task_id: str, local_path: Path) -> ResultStorageOutcome: ...

    def summary(self) -> dict[str, object]: ...


class LocalResultStorage:
    mode = "local"

    def publish(self, task_id: str, local_path: Path) -> ResultStorageOutcome:
        del task_id
        return ResultStorageOutcome(
            mode="local",
            local_path=local_path.resolve(),
            url=None,
        )

    def summary(self) -> dict[str, object]:
        return {"mode": self.mode}


class S3ResultStorage:
    mode = "s3"

    def __init__(
        self,
        *,
        client: Any,
        bucket: str,
        key_prefix: str,
        region: str | None,
        public_base_url: str | None,
    ) -> None:
        self._client = client
        self._bucket = bucket
        self._key_prefix = key_prefix
        self._region = region
        self._public_base_url = public_base_url

    def publish(self, task_id: str, local_path: Path) -> ResultStorageOutcome:
        resolved = local_path.resolve()
        key = f"{self._key_prefix}/{task_id}.mp4"
        try:
            self._client.upload_file(
                str(resolved),
                self._bucket,
                key,
                ExtraArgs={"ContentType": "video/mp4"},
            )
        except Exception as error:
            logger.warning(
                "result upload failed for task %s; retaining local result (%s)",
                task_id,
                type(error).__name__,
            )
            return ResultStorageOutcome(
                mode="local",
                local_path=resolved,
                url=None,
                fallback=True,
                error_type=type(error).__name__,
            )

        url = self._result_url(key)
        try:
            resolved.unlink()
        except OSError as error:
            logger.warning(
                "uploaded task %s but could not remove its local result (%s)",
                task_id,
                type(error).__name__,
            )
        return ResultStorageOutcome(
            mode="s3",
            local_path=None,
            url=url,
        )

    def _result_url(self, key: str) -> str:
        encoded_key = quote(key, safe="/")
        if self._public_base_url is not None:
            return f"{self._public_base_url}/{encoded_key}"
        if self._region and self._region != "us-east-1":
            return (
                f"https://{self._bucket}.s3.{self._region}.amazonaws.com/{encoded_key}"
            )
        return f"https://{self._bucket}.s3.amazonaws.com/{encoded_key}"

    def summary(self) -> dict[str, object]:
        return {"mode": self.mode}


def create_result_storage(
    service_args: ServiceArgs,
    *,
    client: Any | None = None,
) -> LocalResultStorage | S3ResultStorage:
    if service_args.result_storage_mode == "local":
        return LocalResultStorage()
    if client is None:
        try:
            import boto3
        except ImportError as error:
            raise RuntimeError(
                "s3 result storage requires the optional 'mgerase-s3' dependency"
            ) from error
        client_kwargs: dict[str, object] = {}
        if service_args.result_storage_endpoint_url is not None:
            client_kwargs["endpoint_url"] = service_args.result_storage_endpoint_url
        if service_args.result_storage_region is not None:
            client_kwargs["region_name"] = service_args.result_storage_region
        if service_args.result_storage_access_key is not None:
            client_kwargs["aws_access_key_id"] = service_args.result_storage_access_key
            client_kwargs["aws_secret_access_key"] = (
                service_args.result_storage_secret_key
            )
        client = boto3.client("s3", **client_kwargs)
    assert service_args.result_storage_bucket is not None
    return S3ResultStorage(
        client=client,
        bucket=service_args.result_storage_bucket,
        key_prefix=service_args.result_storage_key_prefix,
        region=service_args.result_storage_region,
        public_base_url=service_args.result_storage_public_base_url,
    )


__all__ = (
    "LocalResultStorage",
    "ResultStorage",
    "ResultStorageOutcome",
    "S3ResultStorage",
    "create_result_storage",
)
