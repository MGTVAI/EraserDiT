"""Local runner for the minimal LTX0.9.5 erase pipeline."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch

from config.ltx095 import LTX095EraseSamplingParams
from config.server_args import ServerArgs
from nodes.schedule_batch import Req
from pipelines.session import LTX095EraseSession
from utils.distributed_runtime import get_runtime_distributed_context
from utils.inference_timing import diagnostic_timing_enabled
from utils.logging_utils import init_logger

logger = init_logger(__name__)


def run_ltx095_erase(
    server_args: ServerArgs,
    sampling_params: LTX095EraseSamplingParams,
) -> Req:
    """Build and run the local LTX0.9.5 erase pipeline."""
    runner_start = time.perf_counter()
    shutdown_snapshot: dict[str, object] | None = None
    pipeline_shutdown_s = 0.0
    session: LTX095EraseSession | None = None
    forward_error: BaseException | None = None
    try:
        session = LTX095EraseSession(server_args)
        result = session.run(
            sampling_params,
            warmup_steps=(
                int(getattr(server_args, "warmup_steps", 1))
                if bool(getattr(server_args, "warmup", False))
                else None
            ),
        )
    except BaseException as error:
        forward_error = error
        raise
    finally:
        pipeline_shutdown_start = time.perf_counter()
        try:
            if session is not None:
                shutdown_snapshot = session.close()
        except BaseException as cleanup_error:
            if forward_error is None:
                raise
            forward_error.add_note(
                "pipeline terminal cleanup failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
            logger.exception(
                "LTX095 pipeline terminal cleanup failed after inference error"
            )
        finally:
            pipeline_shutdown_s = time.perf_counter() - pipeline_shutdown_start
    if shutdown_snapshot is not None:
        result.extra["memory_runtime_summary"] = shutdown_snapshot
    if diagnostic_timing_enabled():
        runner_phases = result.extra.setdefault(
            "diagnostic_runner_phase_seconds", {}
        )
        runner_phases["pipeline_shutdown"] = pipeline_shutdown_s
        request_through_barrier = float(
            runner_phases.get("through_post_pipeline_barrier", 0.0) or 0.0
        )
        runner_phases["through_post_pipeline_barrier"] = (
            float(runner_phases.get("distributed_setup", 0.0) or 0.0)
            + float(runner_phases.get("pipeline_initialize", 0.0) or 0.0)
            + request_through_barrier
        )
        runner_phases["through_pipeline_shutdown"] = time.perf_counter() - runner_start
        affinity = (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else []
        )
        result.extra["diagnostic_thread_config"] = {
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cpu_affinity_count": len(affinity),
            "cpu_affinity": affinity,
            "logical_cpu_count": os.cpu_count(),
        }

    output_file_path = result.extra.get("output_file_path")
    final_shape = result.extra.get("runtime_final_video_shape")
    video_meta = result.extra.get("runtime_video_metadata", {})
    audio_muxed = result.extra.get("output_audio_muxed")
    audio_mux_error = result.extra.get("output_audio_mux_error")
    current_context = get_runtime_distributed_context()
    logger.info(
        "LTX095 erase finished: output=%s fps=%s frames=%s final_shape=%s audio_muxed=%s audio_mux_error=%s distributed=%s rank=%d world_size=%d writer_rank=%d distributed_compute_mode=%s native_vae_parallel=%s degree=%d mode=%s",
        output_file_path,
        video_meta.get("fps"),
        video_meta.get("num_frames"),
        final_shape,
        audio_muxed,
        audio_mux_error,
        current_context.distributed_enabled,
        current_context.rank,
        current_context.world_size,
        current_context.writer_rank,
        video_meta.get("distributed_compute_mode", "entry_only"),
        bool(video_meta.get("vae_parallel_enabled", False)),
        int(video_meta.get("vae_parallel_degree", 1) or 1),
        video_meta.get("vae_parallel_mode", "disabled"),
    )
    return result


def resolve_output_file_name(
    output_path: str,
    default_name: str = "mgerase_ltx095_output.mp4",
) -> tuple[str, str]:
    """Split user CLI output path into output dir + file name."""
    output = Path(output_path).expanduser().resolve()
    if output.suffix:
        return str(output.parent), output.name
    return str(output), default_name
