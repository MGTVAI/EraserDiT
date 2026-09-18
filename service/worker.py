"""Resident all-rank worker group for LTX095 erase requests."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

from config.server_args import ServerArgs
from service.commands import CommandKind, WorkerCommand
from service.contract import PipelineServiceContract, resolve_service_contract
from service.control import CancellationToken, ServiceProgressState
from videoerase.session import LTX095EraseSession


_FATAL_ERROR_MARKERS = (
    "nccl",
    "processgroup",
    "process group",
    "collective",
    "connection closed",
    "connection reset",
    "command sequence mismatch",
    "command protocol",
)


def _is_fatal_worker_error(error: BaseException) -> bool:
    """Keep request/data errors recoverable, but retire a broken worker group."""

    if isinstance(error, (SystemExit, KeyboardInterrupt)):
        return True
    message = f"{type(error).__name__}: {error}".lower()
    return any(marker in message for marker in _FATAL_ERROR_MARKERS)




class ResidentWorkerGroup:
    """Rank 0 dispatches idle commands; every rank owns one resident session."""

    def __init__(
        self,
        server_args: ServerArgs,
        *,
        runtime_mode: str,
        task_store: Any | None = None,
        service_contract: PipelineServiceContract | None = None,
    ) -> None:
        self.server_args = server_args
        self.runtime_mode = runtime_mode
        self.service_contract = service_contract or resolve_service_contract(
            getattr(server_args, "pipeline_class_name", None)
        )
        self.task_store = task_store
        self.session = LTX095EraseSession(server_args)
        self.context = self.session.distributed_context
        self._sequence = 0
        self._last_received_sequence = 0
        self._active_lock = threading.Lock()
        self._active_token: CancellationToken | None = None
        self._closed = False
        self._fatal_error: str | None = None
        self._last_heartbeat_at = time.time()

    @property
    def is_owner(self) -> bool:
        return bool(self.context.is_main_process)

    def _broadcast(self, command: WorkerCommand | None) -> WorkerCommand:
        if not self.context.distributed_enabled:
            if command is None:
                raise RuntimeError("local command cannot be None")
            return command
        objects: list[object] = [command.as_dict() if command is not None else None]
        dist.broadcast_object_list(objects, src=int(self.context.writer_rank))
        resolved = WorkerCommand.from_dict(objects[0])
        if resolved.sequence_id != self._last_received_sequence + 1:
            raise RuntimeError("worker command sequence mismatch")
        self._last_received_sequence = resolved.sequence_id
        return resolved

    def _next_command(
        self,
        kind: CommandKind,
        *,
        request_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> WorkerCommand:
        if not self.is_owner:
            raise RuntimeError("only the owner may create worker commands")
        self._sequence += 1
        return WorkerCommand(self._sequence, kind, request_id, payload)

    def execute(self, request_id: str, payload: dict[str, Any]) -> Any:
        command = self._next_command(
            CommandKind.RUN,
            request_id=request_id,
            payload=payload,
        )
        command = self._broadcast(command)
        return self._execute_run(command)

    def _execute_run(self, command: WorkerCommand) -> Any:
        assert command.request_id is not None and command.payload is not None
        token = CancellationToken(command.request_id)
        with self._active_lock:
            self._active_token = token
        # Window batches reconstructed by SP peer ranks intentionally carry only
        # model/runtime payload.  Keep the request-scoped control token on the
        # resident ServerArgs as an all-rank fallback for service checkpoints.
        # There is exactly one active request per worker group, so this binding is
        # unambiguous and is cleared at the request boundary below.
        self.server_args._service_cancellation_token = token
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            params = self.service_contract.build_sampling_params(
                command.payload,
                runtime_mode=self.runtime_mode,
            )
            validate_request = self.service_contract.validate_request
            if validate_request is not None:
                validate_request(command.payload, self.server_args)
            request_extra: dict[str, object] = {
                "service_cancellation_token": token,
                "service_server_args": self.server_args,
            }
            if self.is_owner and self.task_store is not None:
                request_extra["service_progress_state"] = ServiceProgressState(
                    self.task_store,
                    command.request_id,
                )
            warmup_steps = command.payload.get("warmup_steps")
            result = self.session.run(
                params,
                request_extra=request_extra,
                warmup_steps=(int(warmup_steps) if warmup_steps is not None else None),
            )
            allocated = (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            )
            reserved = (
                int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
            )
            peaks = torch.tensor(
                [allocated, reserved],
                dtype=torch.int64,
                device=(
                    torch.device("cuda", torch.cuda.current_device())
                    if self.context.distributed_enabled
                    and dist.get_backend() == "nccl"
                    else torch.device("cpu")
                ),
            )
            if self.context.distributed_enabled:
                dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
            if self.is_owner:
                result.extra["service_peak_memory"] = {
                    "max_allocated_bytes": int(peaks[0].item()),
                    "max_reserved_bytes": int(peaks[1].item()),
                }
            return result
        except BaseException as error:
            if _is_fatal_worker_error(error):
                self._fatal_error = f"{type(error).__name__}: {error}"
            raise
        finally:
            self.server_args._service_cancellation_token = None
            with self._active_lock:
                self._active_token = None

    def request_cancellation(self, request_id: str) -> bool:
        with self._active_lock:
            token = self._active_token
            if token is None or token.request_id != request_id:
                return False
            token.request()
            return True

    def peer_loop(self) -> None:
        if self.is_owner:
            raise RuntimeError("owner cannot enter the peer worker loop")
        while True:
            command = self._broadcast(None)
            if command.kind is CommandKind.RUN:
                try:
                    self._execute_run(command)
                except BaseException:
                    # Runtime boundaries propagate request failures to all ranks. The
                    # owner records the public terminal state.
                    if self._fatal_error is not None:
                        raise
            elif command.kind is CommandKind.HEARTBEAT:
                continue
            elif command.kind is CommandKind.SHUTDOWN:
                return

    def heartbeat(self) -> None:
        command = self._next_command(CommandKind.HEARTBEAT)
        self._broadcast(command)
        self._last_heartbeat_at = time.time()

    def health_snapshot(self) -> dict[str, object]:
        with self._active_lock:
            active_token = self._active_token
            heartbeat_at = (
                active_token.last_checkpoint_at
                if active_token is not None
                else self._last_heartbeat_at
            )
        return {
            "ready": not self._closed and self._fatal_error is None,
            "world_size": int(self.context.world_size),
            "last_all_rank_heartbeat_at": heartbeat_at,
            "active_request_id": (
                active_token.request_id if active_token is not None else None
            ),
            "fatal_error": self._fatal_error,
        }

    @property
    def fatal_error(self) -> str | None:
        return self._fatal_error

    def mark_fatal(self, message: str) -> None:
        self._fatal_error = str(message)

    def close(self) -> None:
        if self._closed:
            return
        if self.is_owner and self._fatal_error is None:
            command = self._next_command(CommandKind.SHUTDOWN)
            self._broadcast(command)
        self.session.close()
        self._closed = True


__all__ = ("ResidentWorkerGroup",)
