"""Local CLI for the EraserDiT erase pipeline.

Accepts either a single task on the command line or a JSON task file holding an
array of tasks; tasks run sequentially through one resident session and results
are written per task id.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from config.eraserdit import (
    ERASERDIT_NEGATIVE_PROMPT,
    EraserDiTEraseSamplingParams,
    EraserDiTPipelineConfig,
)
from config.server_args import ServerArgs
from entrypoints.erase_runner import resolve_output_file_name
from parallel.runtime import destroy_runtime_distributed
from utils.inference_timing import build_ltx095_pure_timing_payload
from utils.determinism import enable_deterministic_mode
from utils.logging_utils import init_logger
from pipelines.session import EraseSession
from models.adapters.eraserdit.cfg import validate_cfg_parallel
from models.adapters.eraserdit.mesh import resolve_mesh

logger = init_logger(__name__)

PIPELINE_NAME = "EraserDiTErasePipeline"

# Task-file keys that map straight onto sampling-parameter fields.
from config.eraserdit_cache import CACHE_DEFAULTS, resolve_eraserdit_cache_params

_PARAM_FIELDS = {
    "prompt",
    "negative_prompt",
    "seed",
    "num_inference_steps",
    "guidance_scale",
    "strength",
    "infer_len",
    "overlap",
    "frame_rate",
    "decode_timestep",
    "decode_noise_scale",
    "max_sequence_length",
    "runtime_workdir",
    "scenes",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the EraserDiT erase pipeline (windowed, optional dual-GPU CFG)."
    )
    parser.add_argument("--transformer-quantization", choices=("none", "int8_w8a8_native"), default="none")
    parser.add_argument("--quantization-scope", choices=("blocks", "ffn"), default="blocks")
    parser.add_argument("--model-path", type=str, required=False)
    parser.add_argument("--video-input", type=str, default=None)
    parser.add_argument("--mask-input", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument(
        "--task-file",
        type=str,
        default=None,
        help="JSON array of tasks; each task overrides the matching CLI option.",
    )
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--negative-prompt", type=str, default=ERASERDIT_NEGATIVE_PROMPT)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--infer-len", type=int, default=121)
    parser.add_argument("--overlap", type=int, default=9)
    parser.add_argument("--frame-rate", type=int, default=25)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    for name, default in CACHE_DEFAULTS.items():
        options = {"default": default}
        if name == "transformer_cache_mode":
            options["choices"] = ("off", "teacache", "cache_dit")
        elif name == "cache_residual_predictor":
            options["choices"] = ("none", "linear")
        elif name == "cache_text_projections" or isinstance(default, bool):
            options["action"] = argparse.BooleanOptionalAction
        else:
            options["type"] = type(default)
        parser.add_argument("--" + name.replace("_", "-"), **options)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cfg-parallel-device", default=None,
                        help="secondary local CUDA device for negative CFG, e.g. cuda:1")
    parser.add_argument("--sp-degree", type=int, default=1)
    parser.add_argument("--sp-linear-mode", choices=["reference", "sharded"], default="reference",
                        help="reference preserves full GEMM shape; sharded is experimental BF16 numerics")
    parser.add_argument("--cfg-degree", type=int, default=1)
    parser.add_argument("--vae-degree", type=int, default=1)
    parser.add_argument("--parallel-devices", type=lambda s: tuple(int(i) for i in s.split(',')), default=None,
                        help="ordered local CUDA indices, e.g. 0,1,2,3")
    parser.add_argument("--vae-tiling", action=argparse.BooleanOptionalAction, default=False,
                        help="explicit spatial tiling; numerics differ from untiled VAE")
    parser.add_argument("--vae-tile-size", type=int, default=512)
    parser.add_argument("--vae-tile-stride", type=int, default=448)
    parser.add_argument(
        "--runtime-mode",
        type=str,
        default="windowed_preload",
        choices=["windowed_preload", "windowed_streaming"],
        help="windowed_preload keeps uint8 frame caches (baseline-exact); "
        "windowed_streaming uses bf16 caches and is only for very long inputs",
    )
    parser.add_argument("--runtime-workdir", type=str, default=None)
    parser.add_argument("--max-weight-usage", type=int, default=5 * 1024**3,
                        help="dynamic offload managed-weight budget in bytes (excludes activations)")
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False,
                        help="pin small unwrapped weights; dynamic extents always use pinned mirrors")
    parser.add_argument(
        "--resource-policy", default="fullgpu",
        choices=["fullgpu", "fullgpu_pin_memory", "dynamic_offload", "component_offload"],
        help="component_offload keeps only the current compute component on GPU",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="sdpa",
        choices=["auto", "sdpa", "flash_attn", "sage_attn", "sage_fp8"],
    )
    parser.add_argument(
        "--enable-torch-compile", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--operator-fusion-backend",
        type=str,
        default="disabled",
        choices=["disabled", "auto", "triton"],
    )
    parser.add_argument("--operator-fusion-ops", type=str, default=None)
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run an optional request-level warmup before the formal request.  "
            "This is where torch.compile pays its Inductor autotune, so it is "
            "what keeps that one-off cost out of the task's own timing."
        ),
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="Denoising steps used by the optional request-level warmup.",
    )
    return parser


def _build_server_args(args: argparse.Namespace) -> ServerArgs:
    pipeline_config = EraserDiTPipelineConfig(
        quantization_scope=args.quantization_scope,
        dit_precision=args.dtype,
        vae_precision=args.dtype,
        text_encoder_precision=args.dtype,
        cfg_parallel_device=args.cfg_parallel_device,
        sp_degree=args.sp_degree, cfg_degree=args.cfg_degree, vae_degree=args.vae_degree,
        sp_linear_mode=args.sp_linear_mode,
        parallel_devices=args.parallel_devices, vae_tiling=args.vae_tiling,
        vae_tile_size=args.vae_tile_size, vae_tile_stride=args.vae_tile_stride,
    )
    return ServerArgs(
        model_path=str(Path(args.model_path).expanduser().resolve()),
        pipeline_class_name=PIPELINE_NAME,
        device=args.device,
        transformer_quantization=args.transformer_quantization,
        weight_dtype=args.dtype,
        resource_policy=args.resource_policy,
        max_weight_usage=args.max_weight_usage,
        pin_memory=args.pin_memory,
        pipeline_config=pipeline_config,
        component_architectures=dict(pipeline_config.component_architectures),
        attention_backend=args.attention_backend,
        enable_torch_compile=bool(args.enable_torch_compile),
        operator_fusion_backend=args.operator_fusion_backend,
        operator_fusion_ops=args.operator_fusion_ops,
        warmup=bool(args.warmup),
        warmup_steps=args.warmup_steps,
    )


def _task_to_sampling_params(
    task: dict[str, Any], defaults: argparse.Namespace
) -> EraserDiTEraseSamplingParams:
    def pick(key: str) -> Any:
        if key in task and task[key] is not None:
            return task[key]
        return getattr(defaults, key, None)

    video_input = task.get("video") or task.get("video_input") or defaults.video_input
    mask_input = task.get("mask") or task.get("mask_input") or defaults.mask_input
    if video_input is None or mask_input is None:
        raise ValueError("each task needs 'video' and 'mask' (or --video-input/--mask-input)")

    output_path = task.get("output") or task.get("output_path") or defaults.output_path
    if output_path is None:
        raise ValueError("each task needs 'output' (or --output-path)")

    output_dir, output_file_name = resolve_output_file_name(output_path)
    overrides = {
        key: task[key] for key in (_PARAM_FIELDS | CACHE_DEFAULTS.keys()) if key in task and task[key] is not None
    }
    for name, default in CACHE_DEFAULTS.items():
        overrides.setdefault(name, getattr(defaults, name, default))
    scenes = overrides.pop("scenes", None)
    if scenes is not None:
        overrides["scenes"] = [tuple(scene) for scene in scenes]

    return EraserDiTEraseSamplingParams(
        prompt=overrides.pop("prompt", defaults.prompt),
        negative_prompt=overrides.pop("negative_prompt", defaults.negative_prompt),
        seed=overrides.pop("seed", defaults.seed),
        num_inference_steps=overrides.pop(
            "num_inference_steps", defaults.num_inference_steps
        ),
        guidance_scale=overrides.pop("guidance_scale", defaults.guidance_scale),
        strength=overrides.pop("strength", defaults.strength),
        infer_len=overrides.pop("infer_len", defaults.infer_len),
        overlap=overrides.pop("overlap", defaults.overlap),
        frame_rate=overrides.pop("frame_rate", defaults.frame_rate),
        max_sequence_length=overrides.pop(
            "max_sequence_length", defaults.max_sequence_length
        ),
        fps=defaults.fps,
        video_input_path=str(Path(video_input).expanduser().resolve()),
        mask_input_path=str(Path(mask_input).expanduser().resolve()),
        output_path=output_dir,
        output_file_name=output_file_name,
        save_output=True,
        suppress_logs=False,
        runtime_mode=defaults.runtime_mode,
        runtime_workdir=(
            str(Path(defaults.runtime_workdir).expanduser().resolve())
            if defaults.runtime_workdir
            else None
        ),
        **overrides,
    )


def _load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.task_file:
        return [{}]
    payload = json.loads(Path(args.task_file).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("task file must hold a non-empty JSON array of tasks")
    for index, task in enumerate(payload):
        if not isinstance(task, dict):
            raise ValueError(f"task #{index} is not a JSON object")
    return payload


def main() -> None:
    # The frozen baseline pipes frames through ffmpeg-python without
    # ``-threads``, so ffmpeg picks its own default.  x264's thread count
    # changes rate control and therefore the decoded pixels (measured ~2 dB
    # at 1080x1920), so pin the same default here.
    os.environ.setdefault("MGERASE_FFMPEG_THREADS", "auto")
    determinism = enable_deterministic_mode()
    parser = _build_parser()
    args = parser.parse_args()
    if args.model_path is None:
        parser.error("--model-path is required")
    logger.info("Deterministic profile: %s", determinism)
    tasks = _load_tasks(args)
    server_args = _build_server_args(args)
    task_params = [_task_to_sampling_params(task, args) for task in tasks]
    for params in task_params:
        resolve_eraserdit_cache_params(params, enable_torch_compile=server_args.enable_torch_compile)
        validate_cfg_parallel(server_args, params)
        resolve_mesh(server_args, params)

    session = None
    try:
        load_start = time.perf_counter()
        session = EraseSession(server_args)
        # t_load is reported on its own and excluded from the end-to-end and
        # memory numbers, so both counters start after the weights are resident.
        load_seconds = time.perf_counter() - load_start
        cuda_device = str(server_args.device).startswith("cuda")
        if cuda_device:
            torch.cuda.reset_peak_memory_stats()
        results: list[dict[str, Any]] = []
        for index, task in enumerate(tasks):
            task_id = str(task.get("id") or f"task{index:03d}")
            params = task_params[index]
            if cuda_device:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            logger.info(
                "Starting EraserDiT erase task id=%s video=%s mask=%s output=%s",
                task_id,
                params.video_input_path,
                params.mask_input_path,
                Path(params.output_path, params.output_file_name),
            )
            started = time.perf_counter()
            # The pipeline is resident for the whole task list, so the warmup
            # is worth paying exactly once.
            result = session.run(
                params,
                warmup_steps=(
                    int(server_args.warmup_steps) if args.warmup and index == 0 else None
                ),
                request_extra={"task_id": task_id},
            )
            elapsed = time.perf_counter() - started
            # `elapsed` is the raw wall clock of the whole call, warmup
            # included; the warmup is listed separately and netted out of the
            # reported end-to-end so a first-run cost never masquerades as
            # steady-state throughput.
            warmup = result.extra.get("warmup") or {}
            warmup_seconds = float(warmup.get("duration_seconds") or 0.0)
            output_file_path = result.extra.get("output_file_path")
            video_meta = result.extra.get("runtime_video_metadata", {})
            logger.info(
                "Finished task id=%s in %.1fs path=%s fps=%s frames=%s",
                task_id,
                elapsed,
                output_file_path,
                video_meta.get("fps"),
                video_meta.get("num_frames"),
            )
            results.append(
                {
                    "id": task_id,
                    "output_file_path": output_file_path,
                    "elapsed_seconds": elapsed,
                    "e2e_seconds_excluding_warmup": elapsed - warmup_seconds,
                    "warmup": warmup,
                    "runtime_video_metadata": video_meta,
                    "resource_policy": server_args.resolve_resource_policy().as_dict(),
                    "memory_runtime": result.extra.get("memory_runtime"),
                    "cfg_parallel": result.extra.get("cfg_parallel"),
                    "quantization": result.extra.get("quantization"),
                    "parallel_history": result.extra.get("parallel_history", []),
                    "transformer_cache_history": result.extra.get("transformer_cache_history", []),
                    "timing": build_ltx095_pure_timing_payload(
                        result.metrics,
                        extra={
                            "torch_compile": result.extra.get("torch_compile"),
                            "attention_backend": result.extra.get(
                                "attention_backend"
                            ),
                            "operator_fusion": result.extra.get("operator_fusion"),
                            "runtime_timing_seconds": result.extra.get(
                                "runtime_timing_seconds"
                            ),
                            "load_seconds": load_seconds,
                            "peak_allocated_gib": (
                                torch.cuda.max_memory_allocated() / (1024**3)
                                if cuda_device
                                else None
                            ),
                            "peak_reserved_gib": (
                                torch.cuda.max_memory_reserved() / (1024**3)
                                if cuda_device
                                else None
                            ),
                        },
                    ),
                }
            )
        print(json.dumps({"tasks": results}, ensure_ascii=False, indent=2))
    finally:
        if session is not None:
            session.close()
        destroy_runtime_distributed()


if __name__ == "__main__":
    main()
