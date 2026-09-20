"""Model-agnostic HTTP service entrypoint.

The pipeline is selected by ``--pipeline-name``; its service contract supplies
the request schema, sampling-parameter builder and capability id, so serving a
new model needs no change here (``vibe/plan.md`` M2).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import uvicorn

from config.server_args import ServerArgs
from config.service_args import ServiceArgs
from entrypoints.http_server import create_http_server_app
from entrypoints.server.storage import create_result_storage
from pipelines.registry import DEFAULT_PIPELINE, PipelineRegistry
from entrypoints.server.artifacts import TaskArtifactManager
from config.service_contract import resolve_service_contract
from entrypoints.server.scheduler import ServiceScheduler
from entrypoints.server.task_store import TaskStore
from entrypoints.server.worker import ResidentWorkerGroup
from utils.distributed_runtime import (
    destroy_runtime_distributed,
    initialize_runtime_distributed,
)
from utils.logging_utils import init_logger

logger = init_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a local erase pipeline.")
    parser.add_argument("--pipeline-name", default=DEFAULT_PIPELINE)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--input-allowed-root", action="append", default=[])
    parser.add_argument("--max-queued-tasks", type=int, default=8)
    parser.add_argument("--max-upload-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--terminal-task-ttl-seconds", type=int, default=86400)
    parser.add_argument("--max-terminal-tasks", type=int, default=128)
    parser.add_argument("--health-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--cancel-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--result-storage-mode", choices=["local", "s3"], default="local")
    parser.add_argument("--result-storage-bucket")
    parser.add_argument("--result-storage-endpoint-url")
    parser.add_argument("--result-storage-region")
    parser.add_argument("--result-storage-public-base-url")
    parser.add_argument("--result-storage-key-prefix", default="mgerase/results")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resource-policy",
        default="fullgpu",
        choices=["fullgpu", "fullgpu_pin_memory", "dynamic_offload", "component_offload"],
    )
    parser.add_argument("--runtime-mode", default=None)
    parser.add_argument("--max-weight-usage", type=int, default=5 * 1024**3,
                        help="dynamic offload managed-weight budget in bytes (excludes activations)")
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False,
                        help="pin small unwrapped weights; dynamic extents always use pinned mirrors")
    parser.add_argument(
        "--attention-backend",
        default="sdpa",
        choices=["auto", "sdpa", "flash_attn", "sage_attn", "sage_fp8"],
    )
    parser.add_argument(
        "--enable-torch-compile", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--operator-fusion-backend",
        default="disabled",
        choices=["disabled", "auto", "triton"],
    )
    parser.add_argument("--operator-fusion-ops", default=None)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup-steps", type=int, default=1)
    return parser


def _resolve_pipeline_config(
    pipeline_cls: type, dtype: str
) -> tuple[Any, dict[str, str]]:
    config_cls = getattr(pipeline_cls, "pipeline_config_cls", None)
    if config_cls is None:
        return SimpleNamespace(), {}
    try:
        config = config_cls(
            dit_precision=dtype, vae_precision=dtype, text_encoder_precision=dtype
        )
    except TypeError:
        config = config_cls()
    architectures = dict(getattr(config, "component_architectures", {}) or {})
    return config, architectures


def _default_runtime_mode(pipeline_cls: type) -> str:
    params_cls = getattr(pipeline_cls, "sampling_params_cls", None)
    if params_cls is not None:
        try:
            return str(params_cls().runtime_mode)
        except Exception:  # pragma: no cover - defensive
            pass
    return "windowed_streaming"


def _build_server_args(args: argparse.Namespace, pipeline_cls: type) -> ServerArgs:
    config, architectures = _resolve_pipeline_config(pipeline_cls, args.dtype)
    return ServerArgs(
        model_path=str(Path(args.model_path).expanduser().resolve()),
        pipeline_class_name=args.pipeline_name,
        device=args.device,
        weight_dtype=args.dtype,
        resource_policy=args.resource_policy,
        max_weight_usage=args.max_weight_usage,
        pin_memory=args.pin_memory,
        pipeline_config=config,
        component_architectures=architectures,
        attention_backend=args.attention_backend,
        enable_torch_compile=bool(args.enable_torch_compile),
        operator_fusion_backend=args.operator_fusion_backend,
        operator_fusion_ops=args.operator_fusion_ops,
    )


def _build_service_args(args: argparse.Namespace) -> ServiceArgs:
    return ServiceArgs(
        host=args.host,
        port=args.port,
        task_root=args.task_root,
        input_allowed_roots=tuple(args.input_allowed_root),
        max_queued_tasks=args.max_queued_tasks,
        max_upload_bytes=args.max_upload_bytes,
        terminal_task_ttl_seconds=args.terminal_task_ttl_seconds,
        max_terminal_tasks=args.max_terminal_tasks,
        health_timeout_seconds=args.health_timeout_seconds,
        cancel_timeout_seconds=args.cancel_timeout_seconds,
        result_storage_mode=args.result_storage_mode,
        result_storage_bucket=args.result_storage_bucket,
        result_storage_endpoint_url=args.result_storage_endpoint_url,
        result_storage_region=args.result_storage_region,
        result_storage_public_base_url=args.result_storage_public_base_url,
        result_storage_key_prefix=args.result_storage_key_prefix,
        result_storage_access_key=os.environ.get("MGERASE_RESULT_STORAGE_ACCESS_KEY"),
        result_storage_secret_key=os.environ.get("MGERASE_RESULT_STORAGE_SECRET_KEY"),
    )


def _effective_acceleration(
    server_args: ServerArgs, worker_group: ResidentWorkerGroup
) -> dict[str, object]:
    """Report the settings that actually took effect, including auto fallbacks.

    ``vibe/plan.md`` M2: the startup config only shows what was requested; the
    attention preflight and the fusion decision resolve later, so the service
    reports both.
    """
    pipeline = getattr(worker_group.session, "pipeline", None)
    attention = dict(getattr(pipeline, "attention_backend_report", {}) or {})
    decision = getattr(server_args, "operator_fusion_decision", None)
    fusion = decision.as_dict() if hasattr(decision, "as_dict") else {}
    return {
        "resource_policy": server_args.resolve_resource_policy().as_dict(),
        "memory_runtime": (
            pipeline._memory_adapter.snapshot()
            if getattr(pipeline, "_memory_adapter", None) is not None else {}
        ),
        "attention_backend": {
            "requested": server_args.attention_backend,
            "report": attention,
            "resolved": bool(attention),
        },
        "transformer_cache": {
            "scope": "request",
            "default": "off",
            "supported_modes": ["off", "teacache", "cache_dit"],
            "experimental": True,
            "compatible_with_torch_compile": False,
            "effective_report": "task.metrics.transformer_cache_history",
        },
        "operator_fusion": fusion,
        "torch_compile": {
            "requested": bool(server_args.enable_torch_compile),
            "active": bool(getattr(server_args, "enable_torch_compile", False)),
        },
        "notes": (
            "values are captured after the resident session is built; entries are "
            "empty until the corresponding preflight has run"
        ),
    }


def main() -> None:
    args = _build_parser().parse_args()
    pipeline_cls, pipeline_name = PipelineRegistry.resolve(args.pipeline_name)
    contract = resolve_service_contract(pipeline_name)
    runtime_mode = args.runtime_mode or _default_runtime_mode(pipeline_cls)
    server_args = _build_server_args(args, pipeline_cls)
    service_args = _build_service_args(args)

    worker_group = None
    scheduler = None
    try:
        distributed_context = initialize_runtime_distributed(server_args)
        result_storage = (
            create_result_storage(service_args)
            if distributed_context.is_main_process
            else None
        )
        task_store = (
            TaskStore(
                service_args.task_root,
                terminal_ttl_seconds=service_args.terminal_task_ttl_seconds,
                max_terminal_tasks=service_args.max_terminal_tasks,
            )
            if distributed_context.is_main_process
            else None
        )
        artifacts = (
            TaskArtifactManager(
                service_args.task_root,
                max_upload_bytes=service_args.max_upload_bytes,
            )
            if distributed_context.is_main_process
            else None
        )
        if artifacts is not None:
            artifacts.cleanup_orphan_staging()
        worker_group = ResidentWorkerGroup(
            server_args,
            runtime_mode=runtime_mode,
            task_store=task_store,
            service_contract=contract,
        )
        if not distributed_context.is_main_process:
            worker_group.peer_loop()
            return
        assert task_store is not None and artifacts is not None and result_storage
        scheduler = ServiceScheduler(
            task_store,
            worker_group,
            result_storage,
            max_queued_tasks=service_args.max_queued_tasks,
            warmup_steps=(args.warmup_steps if args.warmup else None),
            cancel_timeout_seconds=service_args.cancel_timeout_seconds,
        )
        app = create_http_server_app(
            service_args=service_args,
            scheduler=scheduler,
            task_store=task_store,
            artifact_manager=artifacts,
            server_summary={
                "pipeline": pipeline_name,
                "runtime_mode": runtime_mode,
                "dtype": args.dtype,
                "device": args.device,
                "resource_policy": args.resource_policy,
                "attention_backend": args.attention_backend,
                "torch_compile": bool(args.enable_torch_compile),
                "operator_fusion_backend": args.operator_fusion_backend,
                "operator_fusion_ops": args.operator_fusion_ops,
                "result_storage": result_storage.summary(),
            },
            model_summary={
                "id": Path(server_args.model_path).name,
                "capability": contract.capability,
                "pipeline": pipeline_name,
                "request_modes": list(contract.request_modes),
            },
            service_contract=contract,
            effective_acceleration=lambda: _effective_acceleration(
                server_args, worker_group
            ),
        )
        logger.info(
            "service ready pipeline=%s capability=%s on %s:%d",
            pipeline_name,
            contract.capability,
            service_args.host,
            service_args.port,
        )
        uvicorn.run(app, host=service_args.host, port=service_args.port, log_level="info")
    finally:
        if scheduler is not None:
            scheduler.shutdown(timeout=service_args.cancel_timeout_seconds)
        if worker_group is not None:
            worker_group.close()
        destroy_runtime_distributed()


if __name__ == "__main__":
    main()
