"""Launch the resident HTTP service for the local LTX095 erase pipeline."""

from __future__ import annotations

import argparse
import os
import signal
from pathlib import Path

import uvicorn

from config.ltx095 import LTX095EraseSamplingParams
from config.service_args import ServiceArgs
from entrypoints.cli.erase_ltx095 import (
    _build_server_args,
    _resolve_ltx095_sequence_parallel_contract,
)
from entrypoints.http_server import create_http_server_app
from entrypoints.server.storage import create_result_storage
from parallel.stage_policy import synchronize_stage_error
from entrypoints.server.artifacts import TaskArtifactManager
from entrypoints.server.scheduler import ServiceScheduler
from entrypoints.server.task_store import TaskStore
from entrypoints.server.worker import ResidentWorkerGroup
from config.service_contracts.ltx095 import LTX095_SERVICE_CONTRACT
from utils.distributed_runtime import (
    barrier_if_distributed,
    destroy_runtime_distributed,
    initialize_runtime_distributed,
)
from utils.logging_utils import init_logger

logger = init_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the local MGErase LTX095 runtime."
    )
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
    parser.add_argument(
        "--result-storage-mode", choices=["local", "s3"], default="local"
    )
    parser.add_argument("--result-storage-bucket")
    parser.add_argument("--result-storage-endpoint-url")
    parser.add_argument("--result-storage-region")
    parser.add_argument("--result-storage-public-base-url")
    parser.add_argument("--result-storage-key-prefix", default="mgerase/results")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--distributed-backend", default="auto", choices=["auto", "nccl", "gloo"]
    )
    parser.add_argument("--distributed-init-timeout-seconds", type=int, default=1800)
    parser.add_argument("--writer-rank", type=int, default=0)
    parser.add_argument("--progress-rank", type=int, default=0)
    parser.add_argument(
        "--parallel-mode", choices=["disabled", "auto", "manual"], default="auto"
    )
    parser.add_argument("--sp-degree", type=int, default=0)
    parser.add_argument("--cfg-parallel-degree", type=int, default=0)
    parser.add_argument("--vae-parallel-degree", type=int, default=0)
    parser.add_argument(
        "--distributed-compute-mode",
        default="entry_only",
        choices=["auto", "entry_only", "official_vae_parallel"],
    )
    parser.add_argument("--vae-max-parallelism", type=int, default=0)
    parser.add_argument("--vae-max-inflight-tiles", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--resource-policy",
        default="dynamic_offload",
        choices=["fullgpu", "fullgpu_pin_memory", "dynamic_offload"],
    )
    parser.add_argument(
        "--pin_memory", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--dynamic_offload", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max_weight_usage", type=int, default=5 * 1024**3)
    parser.add_argument(
        "--runtime-mode",
        default="windowed_streaming",
        choices=["auto", "full", "windowed", "windowed_preload", "windowed_streaming"],
    )
    parser.add_argument(
        "--attention-backend",
        default="sdpa",
        choices=["auto", "sdpa", "flash_attn", "sage_attn", "sage_fp8"],
    )
    parser.add_argument(
        "--enable-torch-compile", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--transformer-quantization",
        choices=["none", "fp8_w8a8", "fp8_w8a8_triton_selective", "int8_w8a8_viditq"],
        default="none",
    )
    parser.add_argument(
        "--text-encoder-quantization",
        choices=["none", "int8_w8a8_viditq"],
        default="none",
        help="Quantize only the signed 48-Linear LTX095 T5 production policy.",
    )
    parser.add_argument(
        "--fp8-linear-backend",
        choices=["auto", "native_scaled_mm"],
        default="auto",
    )
    parser.add_argument(
        "--fp8-linear-granularity",
        choices=["per_row"],
        default="per_row",
    )
    parser.add_argument(
        "--fp8-fast-accum",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--operator-fusion-backend",
        default="disabled",
        choices=["disabled", "auto", "triton"],
    )
    parser.add_argument("--operator-fusion-ops", default=None)
    parser.add_argument(
        "--warmup", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--warmup-steps", type=int, default=1)
    return parser


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


def main() -> None:
    args = _build_parser().parse_args()
    if args.writer_rank != 0:
        raise ValueError("the service HTTP owner and writer_rank must both be rank 0")
    server_args = _build_server_args(args)
    service_args = _build_service_args(args)
    worker_group = None
    scheduler = None
    result_storage = None
    try:
        distributed_context = initialize_runtime_distributed(server_args)
        capability_error = None
        try:
            _resolve_ltx095_sequence_parallel_contract(
                server_args,
                LTX095EraseSamplingParams(),
                distributed_context,
            )
        except Exception as error:
            capability_error = error
        synchronize_stage_error(capability_error, server_args.parallel_context)
        storage_error = None
        if distributed_context.is_main_process:
            try:
                result_storage = create_result_storage(service_args)
            except Exception as error:
                storage_error = error
        synchronize_stage_error(storage_error, server_args.parallel_context)
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
            runtime_mode=args.runtime_mode,
            task_store=task_store,
        )
        barrier_if_distributed()
        if not distributed_context.is_main_process:
            # Under torchrun, Ctrl-C is forwarded to every local rank.  Peers must
            # remain in the command broadcast so rank 0 can issue SHUTDOWN and
            # tear down the process group in protocol order.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            worker_group.peer_loop()
            return
        assert task_store is not None and artifacts is not None
        assert result_storage is not None
        scheduler = ServiceScheduler(
            task_store,
            worker_group,
            result_storage,
            max_queued_tasks=service_args.max_queued_tasks,
            warmup_steps=(args.warmup_steps if args.warmup else None),
            cancel_timeout_seconds=service_args.cancel_timeout_seconds,
        )
        plan = server_args.parallel_context.plan
        app = create_http_server_app(
            service_args=service_args,
            scheduler=scheduler,
            task_store=task_store,
            artifact_manager=artifacts,
            server_summary={
                "runtime_mode": args.runtime_mode,
                "dtype": args.dtype,
                "device": args.device,
                "attention_backend": args.attention_backend,
                "transformer_quantization": args.transformer_quantization,
                "text_encoder_quantization": args.text_encoder_quantization,
                "fp8_linear_backend": args.fp8_linear_backend,
                "torch_compile": bool(args.enable_torch_compile),
                "operator_fusion_backend": args.operator_fusion_backend,
                "operator_fusion_ops": args.operator_fusion_ops,
                "resource_policy": args.resource_policy,
                "sp_degree": plan.sp_degree,
                "cfg_parallel_degree": plan.cfg_degree,
                "vae_parallel_degree": plan.vae_degree,
                "world_size": plan.world_size,
                "result_storage": result_storage.summary(),
            },
            service_contract=LTX095_SERVICE_CONTRACT,
            model_summary={
                "id": Path(server_args.model_path).name,
                "capability": "ltx095_video_erase",
                "request_modes": ["multipart_upload", "controlled_local_paths"],
            },
        )
        logger.info(
            "MGErase service ready on %s:%d", service_args.host, service_args.port
        )
        uvicorn.run(
            app, host=service_args.host, port=service_args.port, log_level="info"
        )
    finally:
        if scheduler is not None:
            scheduler.shutdown(timeout=service_args.cancel_timeout_seconds)
        if worker_group is not None:
            worker_group.close()
        destroy_runtime_distributed()


if __name__ == "__main__":
    main()
