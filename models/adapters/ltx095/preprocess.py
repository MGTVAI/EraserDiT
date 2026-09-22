"""Erase preprocessing helpers for LTX0.9.5 single-window flow."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.bbox import resolve_single_window_crop_bbox, scale_and_align_bbox
from utils.mask import dilate_mask
from media.video_io import crop_video
from utils.windowing import infer_window_num_frames, pad_single_window


@dataclass
class SingleWindowPreprocessResult:
    crop_bbox: tuple[int, int, int, int]
    crop_video: torch.Tensor
    crop_mask: torch.Tensor
    padded_video: torch.Tensor
    padded_mask: torch.Tensor
    masked_video: torch.Tensor




def preprocess_single_window(
    video: torch.Tensor,
    mask: torch.Tensor,
    bbox: tuple[int, int, int, int] | None,
    infer_len: int,
    align_h: int,
    align_w: int,
    mask_dilate_iter: int,
    mask_dilate_kernel: tuple[int, int],
    scale_area_ratio: float,
    min_pixels: int,
    max_pixels: int,
    force_crop_align: bool,
    use_dynamic_num_frames: bool = False,
    time_sample: int = 8,
    time_shift: int = 1,
    prealigned_crop_bbox: tuple[int, int, int, int] | None = None,
) -> SingleWindowPreprocessResult:
    if video.ndim != 5 or mask.ndim != 5:
        raise ValueError("video and mask must be 5D tensors [B,C,F,H,W]")
    if video.shape[0] != 1 or mask.shape[0] != 1:
        raise ValueError("single-window flow expects batch size 1")

    if prealigned_crop_bbox is None:
        aligned_bbox = resolve_single_window_crop_bbox(
            bbox=bbox,
            video_width=int(video.shape[-1]),
            video_height=int(video.shape[-2]),
            align_h=align_h,
            align_w=align_w,
            scale_area_ratio=scale_area_ratio,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            force_crop_align=force_crop_align,
        )
        crop_video_tensor = crop_video(video[0].permute(1, 0, 2, 3), aligned_bbox)
        crop_mask_tensor = crop_video(mask[0].permute(1, 0, 2, 3), aligned_bbox)
    else:
        aligned_bbox = tuple(int(value) for value in prealigned_crop_bbox)
        if tuple(video.shape[-2:]) != (aligned_bbox[3], aligned_bbox[2]):
            raise ValueError(
                "prealigned video shape does not match global crop bbox: "
                f"video={tuple(video.shape)} bbox={aligned_bbox}"
            )
        if tuple(mask.shape[-2:]) != tuple(video.shape[-2:]):
            raise ValueError("prealigned mask/video spatial shapes must match")
        crop_video_tensor = video[0].permute(1, 0, 2, 3)
        crop_mask_tensor = mask[0].permute(1, 0, 2, 3)

    crop_mask_tensor = dilate_mask(
        crop_mask_tensor,
        dialate_iter=mask_dilate_iter,
        ksize=mask_dilate_kernel,
    )
    pad_num_frames = infer_window_num_frames(
        current_num=int(crop_video_tensor.shape[0]),
        expect_infer_len=infer_len,
        time_sample=time_sample,
        time_shift=time_shift,
        use_dynamic=use_dynamic_num_frames,
    )
    padded_video, padded_mask = pad_single_window(
        crop_video_tensor,
        crop_mask_tensor,
        infer_len=pad_num_frames,
        align_h=align_h,
        align_w=align_w,
    )
    masked_video = padded_video * (1 - padded_mask)
    return SingleWindowPreprocessResult(
        crop_bbox=aligned_bbox,
        crop_video=crop_video_tensor.permute(1, 0, 2, 3).unsqueeze(0),
        crop_mask=crop_mask_tensor.permute(1, 0, 2, 3).unsqueeze(0),
        padded_video=padded_video.permute(1, 0, 2, 3).unsqueeze(0),
        padded_mask=padded_mask.permute(1, 0, 2, 3).unsqueeze(0),
        masked_video=masked_video.permute(1, 0, 2, 3).unsqueeze(0),
    )
