"""LTX0.9.5-specific configuration for the minimal MGErase runtime."""

from __future__ import annotations

from dataclasses import dataclass, field

from config.cache_dit import resolve_ltx095_cache_dit_params
from config.sampling_params import SamplingParams
from config.teacache import resolve_ltx095_teacache_params
from config.transformer_cache import (
    TransformerCacheMode,
    resolve_transformer_cache_mode,
)


@dataclass
class LTX095PipelineConfig:
    """Pipeline-level defaults for the LTX0.9.5 erase path."""

    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    text_encoder_precision: str = "bf16"
    vae_tiling: bool = False
    vae_spatial_compression_ratio: int = 32
    vae_temporal_compression_ratio: int = 8
    flow_shift: float | None = None


@dataclass
class LTX095EraseSamplingParams(SamplingParams):
    """Request parameters required by the future LTX0.9.5 erase pipeline."""

    seed: int | None = 42
    num_inference_steps: int = 50
    guidance_scale: float = 7.0
    strength: float = 1.0
    video_input_path: str | None = None
    mask_input_path: str | None = None
    bbox_path: str | None = None
    scenes: list[tuple[int, int]] | None = None
    # Default to the validated long-window profile used by the production 4K
    # path.  Callers targeting the legacy 1080p profile can still override it.
    infer_len: int = 121
    min_infer_len: int = 41
    time_sample: int = 8
    time_shift: int = 1
    overlap: int = 9
    scale_area_ratio: float = 2.0
    min_pixels: int = 640 * 640
    max_pixels: int = 1920 * 1088
    force_crop_align: bool = False
    mask_dilate_iter: int = 7
    mask_dilate_kernel: tuple[int, int] = (7, 7)
    mask_one_channel: int = 8
    overlap_fuse_mode: str = "before"
    use_dynamic_num_frames: bool = False
    direct_out: bool = True
    enable_colorfix: bool = True
    postprocess_dilate_kernel_size: int = 5
    guss_dialate_iter: int = 20
    guss_dialate_sigma: float = 0.8
    remain_distance: int = 2
    colorfix_type: str = "RGB"
    colorfix_per_channel: bool = False
    runtime_mode: str = "windowed_streaming"
    runtime_workdir: str | None = None
    dynamic_cfg: bool = True
    cfg_step: int = 12
    enable_dynamic_cfg_space: bool = False
    transformer_cache_mode: str = "off"
    teacache_threshold: float = 0.03
    max_teacache_consecutive_skip: int = 1
    do_teacache_calibrate: bool = False
    teacache_coefficient_policy: str = "ltx095_checkpoint_206k"
    cache_dit_front_blocks: int = 1
    cache_dit_back_blocks: int = 0
    cache_dit_warmup_steps: int = 4
    cache_dit_residual_diff_threshold: float = 0.24
    cache_dit_max_consecutive_cached_steps: int = 3
    cache_dit_end_guard_steps: int = 1
    runtime_state: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        mode = resolve_transformer_cache_mode(self.transformer_cache_mode)
        self.transformer_cache_mode = mode.value
        if mode is TransformerCacheMode.TEACACHE:
            resolve_ltx095_teacache_params(
                mode=mode,
                threshold=self.teacache_threshold,
                max_consecutive_skip=self.max_teacache_consecutive_skip,
                calibrate=self.do_teacache_calibrate,
                coefficient_policy=self.teacache_coefficient_policy,
            )
        elif mode is TransformerCacheMode.CACHE_DIT:
            resolve_ltx095_cache_dit_params(
                mode=mode,
                front_blocks=self.cache_dit_front_blocks,
                back_blocks=self.cache_dit_back_blocks,
                warmup_steps=self.cache_dit_warmup_steps,
                residual_diff_threshold=self.cache_dit_residual_diff_threshold,
                max_consecutive_cached_steps=(
                    self.cache_dit_max_consecutive_cached_steps
                ),
                end_guard_steps=self.cache_dit_end_guard_steps,
            )
