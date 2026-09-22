"""Minimal local CLI for the LTX0.9.5 erase pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from config.ltx095 import LTX095EraseSamplingParams, LTX095PipelineConfig
from config.server_args import ServerArgs
from config.transformer_cache import validate_transformer_cache_request
from entrypoints.erase_runner import (
    resolve_output_file_name,
    run_ltx095_erase,
)
from models.dits.ltx095_parallel import (
    LTX095SequenceParallelContract,
    LTX095SequenceParallelRuntimeOptions,
    validate_ltx095_sequence_parallel_capability,
)
from parallel.stage_policy import synchronize_stage_error
from utils.hf_diffusers_utils import load_json_dict
from utils.inference_timing import (
    build_ltx095_pure_timing_payload,
    maybe_write_pure_timing_payload,
)
from parallel.runtime import (
    destroy_runtime_distributed,
    initialize_runtime_distributed,
    resolve_ltx095_native_vae_parallel_status,
)
from utils.logging_utils import init_logger

logger = init_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the minimal local MGErase LTX0.9.5 erase pipeline."
    )
    parser.add_argument("--model-path", required=True, help="LTX0.9.5 weight root")
    parser.add_argument("--video-input", required=True, help="Input video path")
    parser.add_argument("--mask-input", required=True, help="Input mask video path")
    parser.add_argument("--bbox-path", default=None, help="Optional bbox csv/json path")
    parser.add_argument(
        "--output-path", required=True, help="Output mp4 path or directory"
    )
    parser.add_argument("--prompt", default="good quality", help="Positive prompt")
    parser.add_argument(
        "--negative-prompt",
        default=(
            "Colorful color tone, overexposure, static, blurry details, subtitles, "
            "style, artwork, picture, static, overall graying, worst quality, "
            "low-quality, JPEG compression residue, ugly, incomplete, extra fingers, "
            "poorly painted hands, poorly painted faces, deformed, disfigured, "
            "deformed limbs, finger fusion, still image, cluttered background, "
            "three legs, many people in the background, walking backwards, no noise"
        ),
        help="Negative prompt",
    )
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Noisy generator seed for reproducible latent initialization.",
    )
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--infer-len", type=int, default=121)
    parser.add_argument("--overlap", type=int, default=9)
    parser.add_argument("--min-pixels", type=int, default=409600)
    parser.add_argument("--max-pixels", type=int, default=2088960)
    parser.add_argument("--scale-area-ratio", type=float, default=2.0)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--distributed-backend",
        default="auto",
        choices=["auto", "nccl", "gloo"],
        help="Distributed backend selection for torchrun-style launches.",
    )
    parser.add_argument(
        "--distributed-init-timeout-seconds",
        type=int,
        default=1800,
        help="Distributed init timeout in seconds.",
    )
    parser.add_argument(
        "--writer-rank",
        type=int,
        default=0,
        help="Global rank responsible for final output writing.",
    )
    parser.add_argument(
        "--progress-rank",
        type=int,
        default=0,
        help="Global rank responsible for local progress rendering.",
    )
    parser.add_argument(
        "--parallel-mode",
        choices=["disabled", "auto", "manual"],
        default="auto",
    )
    parser.add_argument("--sp-degree", type=int, default=0)
    parser.add_argument("--cfg-parallel-degree", type=int, default=0)
    parser.add_argument("--vae-parallel-degree", type=int, default=0)
    parser.add_argument(
        "--distributed-compute-mode",
        default="entry_only",
        choices=["auto", "entry_only", "official_vae_parallel"],
        help="Distributed compute partitioning mode for multi-rank runs.",
    )
    parser.add_argument(
        "--vae-max-parallelism",
        type=int,
        default=0,
        help="Origin-compatible VAE parallelism contract: 0=auto, 1=disable, >1=force enable.",
    )
    parser.add_argument(
        "--vae-max-inflight-tiles",
        type=int,
        choices=[1, 2],
        default=1,
        help="Maximum in-flight VAE tiles per rank; bounded to 1 or 2.",
    )
    parser.add_argument(
        "--resource-policy",
        default="dynamic_offload",
        choices=[
            "fullgpu",
            "fullgpu_pin_memory",
            "dynamic_offload",
        ],
        help="Runtime resource policy preset.",
    )
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CPU pinned memory registration.",
    )
    parser.add_argument(
        "--dynamic_offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable dynamic weight offload independent of pin_memory.",
    )
    parser.add_argument(
        "--max_weight_usage",
        type=int,
        default=2 * 1024**3,
        help="Maximum per-rank CUDA bytes used by dynamically loaded weights.",
    )
    parser.add_argument(
        "--runtime-mode",
        default="windowed_streaming",
        choices=["auto", "full", "windowed", "windowed_preload", "windowed_streaming"],
        help=(
            "Runtime execution mode. `windowed` is retained as a compatibility alias "
            "for `windowed_preload`."
        ),
    )
    parser.add_argument(
        "--runtime-workdir",
        default=None,
        help="Optional runtime tmp workdir for 4k/windowed memmap files",
    )
    parser.add_argument(
        "--attention-backend",
        default="sdpa",
        choices=["auto", "sdpa", "flash_attn", "sage_attn", "sage_fp8"],
        help=(
            "Select only the LTX095 self-attention backend; cross-attention is "
            "always routed through the common PyTorch SDPA backend."
        ),
    )
    parser.add_argument(
        "--enable-torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the LTX095 Transformer forward path.",
    )
    parser.add_argument(
        "--transformer-quantization",
        choices=["none", "fp8_w8a8", "fp8_w8a8_triton_selective", "int8_w8a8_viditq"],
        default="none",
        help="Quantize only the LTX095 Transformer Linear layers.",
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
        help="Use fast accumulation for FP8 scaled GEMM; conservative is default.",
    )
    parser.add_argument(
        "--operator-fusion-backend",
        default="disabled",
        choices=["disabled", "auto", "triton"],
        help="Select eager operator fusion independently from torch.compile.",
    )
    parser.add_argument(
        "--operator-fusion-ops",
        default=None,
        help=(
            "Comma-separated fused ops. An unset value selects every signed op; "
            "supported names are qk_rmsnorm_rope and rmsnorm_adaln."
        ),
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run an optional request-level warmup before the formal request.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="Denoising steps used by the optional request-level warmup.",
    )
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--force-crop-align", action="store_true")
    parser.add_argument("--mask-dilate-iter", type=int, default=7)
    parser.add_argument("--mask-dilate-kernel", type=int, default=7)
    parser.add_argument("--use-dynamic-num-frames", action="store_true")
    parser.add_argument(
        "--dynamic-cfg", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--cfg-step", type=int, default=12)
    parser.add_argument(
        "--dynamic-cfg-space",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--transformer-cache-mode",
        choices=("off", "teacache", "cache_dit"),
        default="off",
    )
    parser.add_argument("--teacache-threshold", type=float, default=0.03)
    parser.add_argument("--max-teacache-consecutive-skip", type=int, default=1)
    parser.add_argument(
        "--teacache-coefficient-policy",
        default="ltx095_checkpoint_206k",
    )
    parser.add_argument(
        "--do-teacache-calibrate",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--cache-dit-front-blocks", type=int, default=1)
    parser.add_argument("--cache-dit-back-blocks", type=int, default=0)
    parser.add_argument("--cache-dit-warmup-steps", type=int, default=4)
    parser.add_argument(
        "--cache-dit-residual-diff-threshold",
        type=float,
        default=0.24,
    )
    parser.add_argument(
        "--cache-dit-max-consecutive-cached-steps",
        type=int,
        default=3,
    )
    parser.add_argument("--cache-dit-end-guard-steps", type=int, default=1)
    parser.add_argument(
        "--direct-out",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the origin-aligned generated window directly. "
            "Use --no-direct-out for the legacy mask blend."
        ),
    )
    parser.add_argument("--disable-colorfix", action="store_true")
    parser.add_argument("--postprocess-dilate-kernel-size", type=int, default=5)
    parser.add_argument("--guss-dialate-iter", type=int, default=20)
    parser.add_argument("--guss-dialate-sigma", type=float, default=0.8)
    parser.add_argument("--remain-distance", type=int, default=2)
    parser.add_argument("--colorfix-per-channel", action="store_true")
    return parser


def _build_server_args(args: argparse.Namespace) -> ServerArgs:
    pipeline_config = LTX095PipelineConfig(
        dit_precision=args.dtype,
        vae_precision=args.dtype,
        text_encoder_precision=args.dtype,
    )
    return ServerArgs(
        model_path=str(Path(args.model_path).expanduser().resolve()),
        pipeline_class_name="LTX095ErasePipeline",
        device=args.device,
        weight_dtype=args.dtype,
        resource_policy=args.resource_policy,
        dynamic_offload=args.dynamic_offload,
        pin_memory=args.pin_memory,
        max_weight_usage=args.max_weight_usage,
        distributed_backend=(
            None if args.distributed_backend == "auto" else args.distributed_backend
        ),
        distributed_init_timeout_seconds=args.distributed_init_timeout_seconds,
        writer_rank=args.writer_rank,
        progress_rank=args.progress_rank,
        parallel_mode=args.parallel_mode,
        sp_degree=args.sp_degree,
        cfg_parallel_degree=args.cfg_parallel_degree,
        vae_parallel_degree=args.vae_parallel_degree,
        distributed_compute_mode=args.distributed_compute_mode,
        vae_max_parallelism=args.vae_max_parallelism,
        vae_max_inflight_tiles=args.vae_max_inflight_tiles,
        attention_backend=args.attention_backend,
        transformer_quantization=args.transformer_quantization,
        text_encoder_quantization=args.text_encoder_quantization,
        fp8_linear_backend=args.fp8_linear_backend,
        fp8_linear_granularity=args.fp8_linear_granularity,
        fp8_fast_accum=args.fp8_fast_accum,
        enable_torch_compile=args.enable_torch_compile,
        operator_fusion_backend=args.operator_fusion_backend,
        operator_fusion_ops=args.operator_fusion_ops,
        warmup=args.warmup,
        warmup_steps=args.warmup_steps,
        pipeline_config=pipeline_config,
    )


def _build_runtime_timing_extra(
    result_extra: Mapping[str, object],
    *,
    video_input_path: str,
    mask_input_path: str,
    runtime_mode: str,
    resource_policy: str,
) -> dict[str, object]:
    return {
        "output_file_path": result_extra.get("output_file_path"),
        "video_input_path": video_input_path,
        "mask_input_path": mask_input_path,
        "runtime_mode": runtime_mode,
        "resource_policy": resource_policy,
        "transformer_quantization": result_extra.get("transformer_quantization"),
        "text_encoder_quantization": result_extra.get("text_encoder_quantization"),
        "text_embedding_prefill_timing": result_extra.get("text_embedding_prefill_timing"),
        "attention_backend": result_extra.get("attention_backend"),
        "transformer_profile": result_extra.get("transformer_profile"),
        "torch_compile": result_extra.get("torch_compile"),
        "operator_fusion": result_extra.get("operator_fusion"),
        "warmup": result_extra.get("warmup"),
        "diagnostic_runner_phase_seconds": result_extra.get(
            "diagnostic_runner_phase_seconds"
        ),
        "diagnostic_thread_config": result_extra.get("diagnostic_thread_config"),
        "ltx095_vae_parallel_history": result_extra.get(
            "ltx095_vae_parallel_history"
        ),
        "transformer_cache_history": result_extra.get(
            "transformer_cache_history"
        ),
        "runtime_encoding_profile": result_extra.get("runtime_encoding_profile"),
        "runtime_timing_seconds": result_extra.get("runtime_timing_seconds"),
        "runtime_timing_counts": result_extra.get("runtime_timing_counts"),
        "runtime_transfer_bytes": result_extra.get("runtime_transfer_bytes"),
        "runtime_transfer_counts": result_extra.get("runtime_transfer_counts"),
        "runtime_phase_events": result_extra.get("runtime_phase_events"),
        "runtime_window_reclaim_events": result_extra.get(
            "runtime_window_reclaim_events"
        ),
        "memory_registration_summary": result_extra.get(
            "memory_registration_summary"
        ),
        "memory_runtime_summary": result_extra.get("memory_runtime_summary"),
    }


def _build_sampling_params(args: argparse.Namespace) -> LTX095EraseSamplingParams:
    output_dir, output_file_name = resolve_output_file_name(args.output_path)
    kernel = (args.mask_dilate_kernel, args.mask_dilate_kernel)
    return LTX095EraseSamplingParams(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        video_input_path=str(Path(args.video_input).expanduser().resolve()),
        mask_input_path=str(Path(args.mask_input).expanduser().resolve()),
        bbox_path=(
            str(Path(args.bbox_path).expanduser().resolve()) if args.bbox_path else None
        ),
        output_path=output_dir,
        output_file_name=output_file_name,
        save_output=True,
        suppress_logs=False,
        fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        infer_len=args.infer_len,
        overlap=args.overlap,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        scale_area_ratio=args.scale_area_ratio,
        runtime_mode=args.runtime_mode,
        runtime_workdir=(
            str(Path(args.runtime_workdir).expanduser().resolve())
            if args.runtime_workdir
            else None
        ),
        force_crop_align=args.force_crop_align,
        mask_dilate_iter=args.mask_dilate_iter,
        mask_dilate_kernel=kernel,
        use_dynamic_num_frames=args.use_dynamic_num_frames,
        dynamic_cfg=args.dynamic_cfg,
        cfg_step=args.cfg_step,
        enable_dynamic_cfg_space=args.dynamic_cfg_space,
        transformer_cache_mode=args.transformer_cache_mode,
        teacache_threshold=args.teacache_threshold,
        max_teacache_consecutive_skip=args.max_teacache_consecutive_skip,
        do_teacache_calibrate=args.do_teacache_calibrate,
        teacache_coefficient_policy=args.teacache_coefficient_policy,
        cache_dit_front_blocks=args.cache_dit_front_blocks,
        cache_dit_back_blocks=args.cache_dit_back_blocks,
        cache_dit_warmup_steps=args.cache_dit_warmup_steps,
        cache_dit_residual_diff_threshold=(
            args.cache_dit_residual_diff_threshold
        ),
        cache_dit_max_consecutive_cached_steps=(
            args.cache_dit_max_consecutive_cached_steps
        ),
        cache_dit_end_guard_steps=args.cache_dit_end_guard_steps,
        direct_out=args.direct_out,
        enable_colorfix=not args.disable_colorfix,
        postprocess_dilate_kernel_size=args.postprocess_dilate_kernel_size,
        guss_dialate_iter=args.guss_dialate_iter,
        guss_dialate_sigma=args.guss_dialate_sigma,
        remain_distance=args.remain_distance,
        colorfix_per_channel=args.colorfix_per_channel,
    )


def _format_resolved_acceleration_plan(server_args: ServerArgs) -> str:
    parallel_context = getattr(server_args, "parallel_context", None)
    plan = getattr(parallel_context, "plan", None)
    if plan is None:
        raise RuntimeError("resolved acceleration plan is unavailable")
    return (
        f"requested_mode={server_args.parallel_mode} "
        f"enabled={plan.enabled} world_size={plan.world_size} "
        f"sp_degree={plan.sp_degree} cfg_degree={plan.cfg_degree} "
        f"vae_degree={plan.vae_degree} writer_rank={plan.writer_rank}"
    )


def _resolve_ltx095_sequence_parallel_contract(
    server_args: ServerArgs,
    sampling_params: LTX095EraseSamplingParams,
    distributed_context,
) -> LTX095SequenceParallelContract:
    parallel_context = getattr(server_args, "parallel_context", None)
    plan = getattr(parallel_context, "plan", None)
    if plan is None:
        raise RuntimeError("resolved acceleration plan is unavailable")
    transformer_config = MappingProxyType(
        load_json_dict(Path(server_args.model_path, "transformer", "config.json"))
    )
    transformer_cache_mode = getattr(
        sampling_params,
        "transformer_cache_mode",
        "off",
    )
    validate_transformer_cache_request(
        mode=transformer_cache_mode,
        enable_torch_compile=bool(
            getattr(server_args, "enable_torch_compile", False)
        ),
    )
    contract = validate_ltx095_sequence_parallel_capability(
        plan=plan,
        attention_backend=server_args.attention_backend,
        transformer_config=transformer_config,
        runtime_options=LTX095SequenceParallelRuntimeOptions(
            parallel_mode=server_args.parallel_mode,
            distributed_compute_mode=server_args.distributed_compute_mode,
            transformer_cache_mode=transformer_cache_mode,
            teacache_threshold=sampling_params.teacache_threshold,
            do_teacache_calibrate=sampling_params.do_teacache_calibrate,
            global_rank=distributed_context.rank,
            local_rank=distributed_context.local_rank,
            device=server_args.device,
            enable_torch_compile=bool(
                getattr(server_args, "enable_torch_compile", False)
            ),
        ),
    )
    server_args.ltx095_sequence_parallel_contract = contract
    return contract


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    server_args = _build_server_args(args)
    sampling_params = _build_sampling_params(args)
    try:
        distributed_context = initialize_runtime_distributed(server_args)
        official_parallel_context = getattr(
            server_args, "official_parallel_context", None
        )
        acceleration_plan = _format_resolved_acceleration_plan(server_args)
        capability_error = None
        sequence_parallel_contract = None
        try:
            sequence_parallel_contract = _resolve_ltx095_sequence_parallel_contract(
                server_args,
                sampling_params,
                distributed_context,
            )
        except Exception as error:
            capability_error = error
        synchronize_stage_error(capability_error, server_args.parallel_context)
        (
            native_vae_parallel_enabled,
            native_vae_parallel_degree,
            native_vae_parallel_mode,
        ) = resolve_ltx095_native_vae_parallel_status(server_args)
        logger.info(
            "Resolved LTX095 P4 parallel capability contract: %s",
            sequence_parallel_contract,
        )
        logger.info(
            "Starting LTX095 erase: model=%s video=%s mask=%s bbox=%s output=%s steps=%d guidance=%.3f strength=%.3f device=%s dtype=%s runtime_mode=%s resource_policy=%s requested_self_attention_backend=%s cross_attention_backend=torch_sdpa operator_fusion_backend=%s operator_fusion_ops=%s torch_compile=%s distributed=%s rank=%d local_rank=%d world_size=%d writer_rank=%d progress_rank=%d parallel_plan={%s} distributed_compute_mode=%s native_vae_parallel=%s degree=%d mode=%s",
            server_args.model_path,
            sampling_params.video_input_path,
            sampling_params.mask_input_path,
            sampling_params.bbox_path,
            Path(sampling_params.output_path, sampling_params.output_file_name),
            sampling_params.num_inference_steps,
            sampling_params.guidance_scale,
            sampling_params.strength,
            server_args.device,
            server_args.weight_dtype,
            sampling_params.runtime_mode,
            server_args.resource_policy,
            server_args.attention_backend,
            server_args.operator_fusion_backend,
            server_args.operator_fusion_ops,
            server_args.enable_torch_compile,
            distributed_context.distributed_enabled,
            distributed_context.rank,
            distributed_context.local_rank,
            distributed_context.world_size,
            distributed_context.writer_rank,
            distributed_context.progress_rank,
            acceleration_plan,
            getattr(
                official_parallel_context, "distributed_compute_mode", "entry_only"
            ),
            native_vae_parallel_enabled,
            native_vae_parallel_degree,
            native_vae_parallel_mode,
        )
        result = run_ltx095_erase(server_args, sampling_params)
        output_file_path = result.extra.get("output_file_path")
        video_meta = result.extra.get("runtime_video_metadata", {})
        audio_muxed = result.extra.get("output_audio_muxed")
        audio_mux_error = result.extra.get("output_audio_mux_error")
        timing_payload = build_ltx095_pure_timing_payload(
            result.metrics,
            extra=_build_runtime_timing_extra(
                result.extra,
                video_input_path=sampling_params.video_input_path,
                mask_input_path=sampling_params.mask_input_path,
                runtime_mode=sampling_params.runtime_mode,
                resource_policy=server_args.resource_policy,
            ),
        )
        # All ranks execute this CLI in a torchrun job, but the timing sidecar is
        # a single experiment-level artifact.  Restrict its write to the output
        # owner so concurrent ranks cannot race and leave an arbitrary payload.
        timing_path = (
            maybe_write_pure_timing_payload(timing_payload)
            if distributed_context.is_writer_rank
            else None
        )
        if timing_payload.get("pure_inference_seconds") is not None:
            logger.info(
                "Pure inference timing: pure_inference=%.3fs pipeline_total=%s sidecar=%s",
                timing_payload["pure_inference_seconds"],
                (
                    f"{timing_payload['pipeline_total_seconds']:.3f}s"
                    if timing_payload.get("pipeline_total_seconds") is not None
                    else "n/a"
                ),
                timing_path,
            )
        logger.info(
            "Saved erase output: path=%s fps=%s frames=%s final_shape=%s audio_muxed=%s audio_mux_error=%s",
            output_file_path,
            video_meta.get("fps"),
            video_meta.get("num_frames"),
            result.extra.get("runtime_final_video_shape"),
            audio_muxed,
            audio_mux_error,
        )
    finally:
        destroy_runtime_distributed()


if __name__ == "__main__":
    main()
