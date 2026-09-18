"""HTTP service configuration for the local MGErase runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class ServiceArgs:
    """Configuration owned by the service layer, not by model requests."""

    host: str = "127.0.0.1"
    port: int = 30000
    task_root: str = "./mgerase_service_tasks"
    input_allowed_roots: tuple[str, ...] = ()
    max_queued_tasks: int = 8
    max_upload_bytes: int = 4 * 1024**3
    terminal_task_ttl_seconds: int = 24 * 60 * 60
    max_terminal_tasks: int = 128
    health_timeout_seconds: float = 120.0
    cancel_timeout_seconds: float = 120.0
    result_storage_mode: str = "local"
    result_storage_bucket: str | None = None
    result_storage_endpoint_url: str | None = None
    result_storage_region: str | None = None
    result_storage_public_base_url: str | None = None
    result_storage_key_prefix: str = "mgerase/results"
    result_storage_access_key: str | None = field(default=None, repr=False)
    result_storage_secret_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("host must not be empty")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("port must be an integer in [1, 65535]")
        for name in (
            "max_queued_tasks",
            "max_upload_bytes",
            "terminal_task_ttl_seconds",
            "max_terminal_tasks",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive non-bool int")
        for name in ("health_timeout_seconds", "cancel_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a positive number")
            if float(value) <= 0:
                raise ValueError(f"{name} must be a positive number")

        task_root = Path(self.task_root).expanduser().resolve()
        allowed_roots = tuple(
            str(Path(path).expanduser().resolve()) for path in self.input_allowed_roots
        )
        storage_mode = self.result_storage_mode.strip().lower()
        if storage_mode not in {"local", "s3"}:
            raise ValueError("result_storage_mode must be 'local' or 's3'")
        bucket = (
            self.result_storage_bucket.strip()
            if self.result_storage_bucket is not None
            else None
        )
        endpoint_url = self._normalize_http_url(
            "result_storage_endpoint_url", self.result_storage_endpoint_url
        )
        public_base_url = self._normalize_http_url(
            "result_storage_public_base_url",
            self.result_storage_public_base_url,
        )
        key_prefix = self.result_storage_key_prefix.strip().strip("/")
        if not key_prefix:
            raise ValueError("result_storage_key_prefix must not be empty")
        if any(part in {"", ".", ".."} for part in key_prefix.split("/")):
            raise ValueError("result_storage_key_prefix contains an unsafe segment")
        if storage_mode == "s3":
            if not bucket:
                raise ValueError("result_storage_bucket is required for s3 storage")
            if endpoint_url is not None and public_base_url is None:
                raise ValueError(
                    "result_storage_public_base_url is required with a custom endpoint"
                )
        access_key = self.result_storage_access_key or None
        secret_key = self.result_storage_secret_key or None
        if bool(access_key) != bool(secret_key):
            raise ValueError(
                "result storage access and secret keys must be configured together"
            )
        object.__setattr__(self, "task_root", str(task_root))
        object.__setattr__(self, "input_allowed_roots", allowed_roots)
        object.__setattr__(self, "result_storage_mode", storage_mode)
        object.__setattr__(self, "result_storage_bucket", bucket)
        object.__setattr__(self, "result_storage_endpoint_url", endpoint_url)
        object.__setattr__(self, "result_storage_public_base_url", public_base_url)
        object.__setattr__(self, "result_storage_key_prefix", key_prefix)
        object.__setattr__(self, "result_storage_access_key", access_key)
        object.__setattr__(self, "result_storage_secret_key", secret_key)

    @staticmethod
    def _normalize_http_url(name: str, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().rstrip("/")
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"{name} must be an absolute http(s) URL")
        if parsed.query or parsed.fragment:
            raise ValueError(f"{name} must not contain a query or fragment")
        return normalized


__all__ = ("ServiceArgs",)
