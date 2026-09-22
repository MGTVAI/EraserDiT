"""Resident EraserDiT videoerase pipeline session."""

from __future__ import annotations

import os
import time
from copy import deepcopy

import torch

from config.eraserdit import EraserDiTEraseSamplingParams
from config.server_args import ServerArgs, set_global_server_args
from nodes.schedule_batch import Req
from pipelines.registry import PipelineRegistry
from utils.cpu_resources import (
    configure_process_cpu_resources,
    process_cpu_resources_snapshot,
)
from utils.determinism import enable_deterministic_mode
from parallel.runtime import (
    barrier_if_distributed,
    initialize_runtime_distributed,
)
from utils.inference_timing import diagnostic_timing_enabled


def _build_request_generator(
    server_args: ServerArgs,
    sampling_params: EraserDiTEraseSamplingParams,
) -> torch.Generator | None:
    seed = getattr(sampling_params, "seed", None)
    if seed is None:
        return None
    return torch.Generator(device=server_args.device).manual_seed(int(seed))


class EraseSession:
    """Own one resident pipeline and execute isolated requests through it."""

    def __init__(self, server_args: ServerArgs) -> None:
        self.server_args = server_args
        self._closed = False
        self._cpu_resources = None
        # Must run before any CUDA work so the cuBLAS workspace config and the
        # deterministic kernels are in place for the whole session.
        enable_deterministic_mode()
        initialize_start = time.perf_counter()
        local_world_size = int(
            os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1"))
        )
        if os.environ.get("MGERASE_CPU_AFFINITY_MAP") or local_world_size > 1:
            self._cpu_resources = configure_process_cpu_resources()
        distributed_context = getattr(server_args, "distributed_context", None)
        if distributed_context is None:
            distributed_context = initialize_runtime_distributed(server_args)
        self.distributed_context = distributed_context
        from layers.operator_fusion.registry import (
            resolve_operator_fusion_decision,
        )

        server_args.operator_fusion_decision = resolve_operator_fusion_decision(
            server_args
        )
        set_global_server_args(server_args)
        self.distributed_setup_s = time.perf_counter() - initialize_start
        pipeline_start = time.perf_counter()
        pipeline_cls, pipeline_class_name = PipelineRegistry.resolve(
            getattr(server_args, "pipeline_class_name", None)
        )
        server_args.pipeline_class_name = pipeline_class_name
        self.pipeline = pipeline_cls(
            model_path=server_args.model_path,
            server_args=server_args,
        )
        self.pipeline_initialize_s = time.perf_counter() - pipeline_start

    def build_request(
        self,
        sampling_params: EraserDiTEraseSamplingParams,
        *,
        request_extra: dict[str, object] | None = None,
    ) -> Req:
        if self._closed:
            raise RuntimeError("EraserDiT erase session is closed")
        req = Req(
            sampling_params=sampling_params,
            generator=_build_request_generator(self.server_args, sampling_params),
        )
        req.extra["runtime_distributed_metadata"] = self.distributed_context.as_dict()
        if request_extra:
            req.extra.update(request_extra)
        req.suppress_logs = bool(req.suppress_logs) or not (
            self.distributed_context.is_main_process
            or self.distributed_context.is_progress_rank
        )
        return req

    def _forward_with_operator_fusion(self, req: Req) -> Req:
        from layers.operator_fusion.runtime import (
            operator_fusion_request_scope,
        )

        decision = self.server_args.operator_fusion_decision
        with operator_fusion_request_scope(decision) as fusion_stats:
            result = self.pipeline.forward(req, self.server_args)
        result.extra["operator_fusion"] = fusion_stats.snapshot()
        return result

    def run(
        self,
        sampling_params: EraserDiTEraseSamplingParams,
        *,
        warmup_steps: int | None = None,
        request_extra: dict[str, object] | None = None,
    ) -> Req:
        request_start = time.perf_counter()
        req = self.build_request(sampling_params, request_extra=request_extra)
        request_prepare_s = time.perf_counter() - request_start
        warmup_summary: dict[str, object] | None = None
        if warmup_steps is not None:
            warmup_req = self.build_request(deepcopy(sampling_params))
            warmup_req.set_as_warmup(int(warmup_steps))
            warmup_start = time.perf_counter()
            warmup_result = self._forward_with_operator_fusion(warmup_req)
            warmup_duration_s = time.perf_counter() - warmup_start
            warmup_metrics = getattr(warmup_result, "metrics", None)
            warmup_summary = {
                "mode": (
                    "compiled"
                    if bool(getattr(self.server_args, "enable_torch_compile", False))
                    else "eager"
                ),
                "steps": int(warmup_steps),
                "duration_seconds": warmup_duration_s,
                "pipeline_seconds": warmup_duration_s,
                "stage_breakdown_ms": (
                    dict(warmup_metrics.stages) if warmup_metrics is not None else {}
                ),
                "torch_compile": warmup_result.extra.get("torch_compile"),
                "operator_fusion": warmup_result.extra.get("operator_fusion"),
            }
        pipeline_start = time.perf_counter()
        result = self._forward_with_operator_fusion(req)
        pipeline_duration_s = time.perf_counter() - pipeline_start
        if warmup_summary is not None:
            result.extra["warmup"] = warmup_summary
        if result.metrics is not None:
            result.metrics.total_duration_ms = pipeline_duration_s * 1000.0
        if self._cpu_resources is not None:
            result.extra["cpu_resources"] = process_cpu_resources_snapshot(
                self._cpu_resources
            )
        barrier_start = time.perf_counter()
        barrier_if_distributed()
        post_pipeline_barrier_s = time.perf_counter() - barrier_start
        if diagnostic_timing_enabled():
            result.extra["diagnostic_runner_phase_seconds"] = {
                "distributed_setup": self.distributed_setup_s,
                "pipeline_initialize": self.pipeline_initialize_s,
                "request_prepare": request_prepare_s,
                "pipeline_forward": pipeline_duration_s,
                "post_pipeline_barrier": post_pipeline_barrier_s,
                "through_post_pipeline_barrier": time.perf_counter() - request_start,
            }
        return result

    def close(self) -> dict[str, object] | None:
        if self._closed:
            return None
        self._closed = True
        return self.pipeline.close(terminal=True)

    def __enter__(self) -> "EraseSession":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = ("EraseSession",)
