"""EraserDiT-specific configuration for the composed runtime.

Sampling defaults define the model's inference behavior; see ``docs/cli.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config.sampling_params import SamplingParams

# Default negative prompt; callers may override it per request.
ERASERDIT_NEGATIVE_PROMPT = (
    "Colorful color tone, overexposure, static, blurry details, subtitles, style, "
    "artwork, picture, static, overall graying, worst quality, low-quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly painted hands, "
    "poorly painted faces, deformed, disfigured, deformed limbs, finger fusion, "
    "still image, cluttered background, three legs, many people in the background, "
    "walking backwards, no noise"
)


@dataclass
class EraserDiTPipelineConfig:
    """Pipeline-level defaults for the EraserDiT erase path."""

    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    text_encoder_precision: str = "bf16"
    vae_spatial_compression_ratio: int = 32
    vae_temporal_compression_ratio: int = 8
    # Local CUDA index after CUDA_VISIBLE_DEVICES; None preserves serial CFG.
    cfg_parallel_device: str | None = None
    quantization_scope: str = "blocks"
    sp_degree: int = 1
    sp_linear_mode: str = "reference"
    cfg_degree: int = 1
    vae_degree: int = 1
    parallel_devices: tuple[int, ...] | None = None
    vae_tiling: bool = False
    vae_tile_size: int = 512
    vae_tile_stride: int = 448
    # The adapter declares its own component class names instead of
    # relying on the checkpoint's ``model_index.json`` / ``_class_name``.
    component_architectures: dict[str, str] = field(
        default_factory=lambda: {
            "transformer": "EraserDiTLTXVideoTransformer3DModel",
            "vae": "EraserDiTAutoencoderKLLTXVideo",
            "scheduler": "FlowMatchEulerDiscreteScheduler",
        }
    )


@dataclass
class EraserDiTEraseSamplingParams(SamplingParams):
    """Request parameters for the EraserDiT erase pipeline."""

    seed: int | None = 42
    # Frozen baseline sampling configuration (``utils/inference_utils.py:56-63``).
    num_inference_steps: int = 50
    guidance_scale: float = 3.0
    strength: float = 0.8
    video_input_path: str | None = None
    mask_input_path: str | None = None
    bbox_path: str | None = None
    scenes: list[tuple[int, int]] | None = None
    # Window arithmetic.  ``shift_alpha`` is ``1 * 8 + 1`` (``inference.py:47``)
    # and equals ``overlap``; ``infer_len`` is TEMP_INFER_LEN.
    infer_len: int = 121
    overlap: int = 9
    time_sample: int = 8
    time_shift: int = 1
    # Whole-frame erase: the runtime's crop bbox must resolve to (0, 0, W, H).
    scale_area_ratio: float = 1.0
    min_pixels: int = 0
    max_pixels: int = 8192 * 8192
    force_crop_align: bool = False
    # Spatial alignment / mask morphology (``utils/pre.py:13`` + ``inference.py:45``).
    align_h: int = 32
    align_w: int = 32
    mask_ksize: tuple[int, int] = (9, 9)
    mask_dilate_iter: int = 9
    mask_threshold: float = 0.039
    mask_enable_approximate: bool = True
    # RoPE temporal scale uses a *fixed* frame_rate of 25; it does not follow the
    # input video frame rate.
    frame_rate: int = 25
    decode_timestep: float = 0.0
    decode_noise_scale: float = 0.0
    max_sequence_length: int = 128
    negative_prompt: str | None = ERASERDIT_NEGATIVE_PROMPT
    # Whole-frame erase: no bbox cropping (plan §4.7).
    crop_flag: bool = False
    # The streaming runtime keeps its frame caches in bf16, which quantises both
    # the model input and the committed frames; measured against the frozen
    # baseline that costs ~1.2 dB on the non-erase region.  The preload runtime
    # uses uint8 caches and matches the baseline exactly.  Streaming stays
    # available (``--runtime-mode windowed_streaming``) for very long inputs.
    runtime_mode: str = "windowed_preload"
    runtime_workdir: str | None = None
    # Output write contract (plan §4.10): libx264 / yuv420p / bit rate
    # ``bit_rate // 1e6`` M, overriding the shared profile builder.
    output_encode_baseline_contract: bool = True
    # Colour alignment applied on write (``utils/post_pkg.py``).
    colorfix_type: str = "RGB"
    colorfix_per_channel: bool = True
    # ``overlap_fuse_mode`` stays at the framework default ("before") so the
    # overlap region keeps the previously committed pixels, matching the original
    # ``pre_video_shift`` semantics (plan §1).
    overlap_fuse_mode: str = "before"
    runtime_state: dict[str, object] = field(default_factory=dict)

    transformer_cache_mode: str = "off"
    transformer_cache_force_compute: bool = False
    cache_text_projections: bool | None = None
    cache_residual_predictor: str = "none"
    teacache_threshold: float = 0.005
    max_teacache_consecutive_skip: int = 1
    teacache_warmup_steps: int = 4
    cache_dit_front_blocks: int = 1
    cache_dit_back_blocks: int = 0
    cache_dit_warmup_steps: int = 4
    cache_dit_residual_diff_threshold: float = 0.03
    cache_dit_max_consecutive_cached_steps: int = 1
    cache_end_guard_steps: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        from config.eraserdit_cache import resolve_eraserdit_cache_params
        resolve_eraserdit_cache_params(self)
        if self.mask_dilate_iter < 0:
            raise ValueError("mask_dilate_iter must be non-negative")
        if self.overlap >= self.infer_len:
            raise ValueError("overlap must be smaller than infer_len")
