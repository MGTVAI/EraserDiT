"""LTX0.9.5 video-erase service contract.

The request schema, the sampling-parameter builder and the capability id for the
LTX095 model.  Behaviour is unchanged from the values that used to live in
``entrypoints/server/protocol.py`` and ``service/worker.py``; they simply moved
here so the service skeleton no longer references the model directly.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from config.transformer_cache import validate_transformer_cache_request
from service.contract import PipelineServiceContract

DEFAULT_NEGATIVE_PROMPT = (
    "Colorful color tone, overexposure, static, blurry details, subtitles, "
    "style, artwork, picture, static, overall graying, worst quality, "
    "low-quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly painted hands, poorly painted faces, deformed, disfigured, "
    "deformed limbs, finger fusion, still image, cluttered background, "
    "three legs, many people in the background, walking backwards, no noise"
)


class VideoSamplingRequest(BaseModel):
    """Only request-time knobs already implemented by the local CLI."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str = "good quality"
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    seed: int = 42
    fps: Annotated[int, Field(ge=1, le=240)] = 25
    num_inference_steps: Annotated[int, Field(ge=1, le=1000)] = 50
    guidance_scale: Annotated[float, Field(ge=0.0)] = 7.0
    strength: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    infer_len: Annotated[int, Field(ge=1)] = 121
    overlap: Annotated[int, Field(ge=0)] = 9
    min_pixels: Annotated[int, Field(ge=1)] = 409600
    max_pixels: Annotated[int, Field(ge=1)] = 2088960
    scale_area_ratio: Annotated[float, Field(gt=0.0)] = 2.0
    dynamic_cfg: bool = True
    cfg_step: Annotated[int, Field(ge=0)] = 12
    dynamic_cfg_space: bool = False
    transformer_cache_mode: Literal["off", "teacache", "cache_dit"] = "off"
    teacache_threshold: Annotated[float, Field(gt=0.0)] = 0.03
    max_teacache_consecutive_skip: Annotated[int, Field(ge=1)] = 1
    do_teacache_calibrate: bool = False
    teacache_coefficient_policy: Literal["ltx095_checkpoint_206k"] = (
        "ltx095_checkpoint_206k"
    )
    cache_dit_front_blocks: Annotated[int, Field(ge=1)] = 1
    cache_dit_back_blocks: Annotated[int, Field(ge=0)] = 0
    cache_dit_warmup_steps: Annotated[int, Field(ge=0)] = 4
    cache_dit_residual_diff_threshold: Annotated[
        float, Field(gt=0.0, lt=1.0)
    ] = 0.24
    cache_dit_max_consecutive_cached_steps: Annotated[int, Field(ge=1)] = 3
    cache_dit_end_guard_steps: Annotated[int, Field(ge=0)] = 1
    force_crop_align: bool = False
    mask_dilate_iter: Annotated[int, Field(ge=0)] = 7
    mask_dilate_kernel: Annotated[int, Field(ge=1)] = 7
    use_dynamic_num_frames: bool = False
    direct_out: bool = True
    enable_colorfix: bool = True
    postprocess_dilate_kernel_size: Annotated[int, Field(ge=1)] = 5
    guss_dialate_iter: Annotated[int, Field(ge=0)] = 20
    guss_dialate_sigma: Annotated[float, Field(ge=0.0)] = 0.8
    remain_distance: Annotated[int, Field(ge=0)] = 2
    colorfix_per_channel: bool = False

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "VideoSamplingRequest":
        if (
            self.cache_dit_front_blocks + self.cache_dit_back_blocks
            >= 28
        ):
            raise ValueError(
                "cache_dit_front_blocks + cache_dit_back_blocks must be "
                "smaller than 28"
            )
        if self.min_pixels > self.max_pixels:
            raise ValueError("min_pixels must not exceed max_pixels")
        if self.overlap >= self.infer_len:
            raise ValueError("overlap must be smaller than infer_len")
        if self.mask_dilate_kernel % 2 == 0:
            raise ValueError("mask_dilate_kernel must be odd")
        if self.postprocess_dilate_kernel_size % 2 == 0:
            raise ValueError("postprocess_dilate_kernel_size must be odd")
        return self


class LocalVideoCreateRequest(VideoSamplingRequest):
    video_path: str
    mask_path: str
    bbox_path: str | None = None


class MultipartVideoParameters(VideoSamplingRequest):
    """JSON carried by the optional multipart ``parameters`` field."""


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    message: str
    phase: str | None = None




class LTX095SamplingDefaults:
    """Namespace kept so callers can introspect the model's default schema."""


def build_ltx095_sampling_params(payload: dict[str, Any], *, runtime_mode: str):
    from config.ltx095 import LTX095EraseSamplingParams

    values = dict(payload["sampling"])
    kernel = int(values.pop("mask_dilate_kernel"))
    dynamic_cfg_space = bool(values.pop("dynamic_cfg_space"))
    return LTX095EraseSamplingParams(
        **values,
        video_input_path=str(payload["video_input_path"]),
        mask_input_path=str(payload["mask_input_path"]),
        bbox_path=payload.get("bbox_input_path"),
        output_path=str(payload["output_dir"]),
        output_file_name=str(payload["output_file_name"]),
        save_output=True,
        suppress_logs=True,
        runtime_mode=runtime_mode,
        runtime_workdir=str(payload["runtime_workdir"]),
        mask_dilate_kernel=(kernel, kernel),
        enable_dynamic_cfg_space=dynamic_cfg_space,
    )


def _validate_ltx095_request(payload: dict[str, Any], server_args: Any) -> None:
    sampling = payload["sampling"]
    validate_transformer_cache_request(
        mode=sampling["transformer_cache_mode"],
        enable_torch_compile=bool(getattr(server_args, "enable_torch_compile", False)),
    )


LTX095_SERVICE_CONTRACT = PipelineServiceContract(
    capability="ltx095_video_erase",
    request_schema_cls=LocalVideoCreateRequest,
    multipart_schema_cls=MultipartVideoParameters,
    build_sampling_params=build_ltx095_sampling_params,
    validate_request=_validate_ltx095_request,
)


__all__ = (
    "DEFAULT_NEGATIVE_PROMPT",
    "LTX095_SERVICE_CONTRACT",
    "LTX095SamplingDefaults",
    "LocalVideoCreateRequest",
    "MultipartVideoParameters",
    "VideoSamplingRequest",
    "build_ltx095_sampling_params",
)
