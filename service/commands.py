"""Fixed-schema commands for the resident worker group."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


PROTOCOL_VERSION = 1


class CommandKind(str, Enum):
    RUN = "run"
    HEARTBEAT = "heartbeat"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True)
class WorkerCommand:
    sequence_id: int
    kind: CommandKind
    request_id: str | None = None
    payload: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if type(self.sequence_id) is not int or self.sequence_id < 1:
            raise ValueError("sequence_id must be a positive int")
        if not isinstance(self.kind, CommandKind):
            raise TypeError("kind must be CommandKind")
        if self.kind is CommandKind.RUN:
            if not self.request_id or not isinstance(self.payload, dict):
                raise ValueError("RUN requires request_id and payload")
        elif self.request_id is not None or self.payload is not None:
            raise ValueError("non-RUN commands cannot carry request data")

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "sequence_id": self.sequence_id,
            "kind": self.kind.value,
            "request_id": self.request_id,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, value: object) -> "WorkerCommand":
        if type(value) is not dict:
            raise ValueError("worker command must be a dict")
        expected = {
            "protocol_version",
            "sequence_id",
            "kind",
            "request_id",
            "payload",
        }
        if set(value) != expected:
            raise ValueError("worker command schema mismatch")
        if value["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError("worker command protocol version mismatch")
        return cls(
            sequence_id=value["sequence_id"],
            kind=CommandKind(value["kind"]),
            request_id=value["request_id"],
            payload=value["payload"],
        )


__all__ = ("CommandKind", "PROTOCOL_VERSION", "WorkerCommand")
