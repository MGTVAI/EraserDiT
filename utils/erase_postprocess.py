"""Erase postprocess helpers for the local LTX0.9.5 single-window flow."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.mask import gaussian_dilate_mask
from utils.video_io import splice_video


@dataclass
class SingleWindowPostprocessResult:
    crop_video_modified: torch.Tensor
    output_video: torch.Tensor | None
    paste_mask: torch.Tensor | None = None


def _calc_mean_std_mask(
    feat: torch.Tensor,
    mask: torch.Tensor | None = None,
    per_channel: bool = False,
    eps: float = 1e-8,
    unbiased_variance: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if feat.ndim != 4:
        raise ValueError(f"Expected 4D tensor [F,C,H,W], got {tuple(feat.shape)}")
    if mask is not None and mask.shape != feat.shape:
        raise ValueError(
            f"mask and feat shape must match, got {tuple(mask.shape)} and {tuple(feat.shape)}"
        )

    if mask is None:
        if per_channel:
            line = feat.view(feat.shape[0], feat.shape[1], -1)
            mean = line.mean(dim=2, keepdim=True).view(feat.shape[0], feat.shape[1], 1, 1)
            std = (line.var(dim=2, keepdim=True) + eps).sqrt().view(feat.shape[0], feat.shape[1], 1, 1)
            return mean, std
        line = feat.view(feat.shape[0], -1)
        mean = line.mean(dim=1, keepdim=True).view(feat.shape[0], 1, 1, 1).expand(-1, feat.shape[1], -1, -1)
        std = (line.var(dim=1, keepdim=True) + eps).sqrt().view(feat.shape[0], 1, 1, 1).expand(-1, feat.shape[1], -1, -1)
        return mean, std

    feat_f = feat.to(torch.float32)
    mask_f = mask.to(device=feat.device, dtype=torch.float32)
    if per_channel:
        reduce_dims = (2, 3)
        counts = mask_f.sum(dim=reduce_dims, keepdim=True)
        safe_counts = counts.clamp_min(1.0)
        summed = (feat_f * mask_f).sum(dim=reduce_dims, keepdim=True)
        mean = summed / safe_counts
        centered = (feat_f - mean) * mask_f
        variance_counts = (
            (counts - 1.0).clamp_min(1.0)
            if unbiased_variance
            else safe_counts
        )
        var = (centered * centered).sum(
            dim=reduce_dims, keepdim=True
        ) / variance_counts
        std = (var + eps).sqrt()
        empty = counts <= 0
        mean = torch.where(empty, torch.zeros_like(mean), mean)
        std = torch.where(empty, torch.full_like(std, eps), std)
        return mean, std

    reduce_dims = (1, 2, 3)
    counts = mask_f.sum(dim=reduce_dims, keepdim=True)
    safe_counts = counts.clamp_min(1.0)
    summed = (feat_f * mask_f).sum(dim=reduce_dims, keepdim=True)
    mean_scalar = summed / safe_counts
    centered = (feat_f - mean_scalar) * mask_f
    variance_counts = (
        (counts - 1.0).clamp_min(1.0)
        if unbiased_variance
        else safe_counts
    )
    var_scalar = (centered * centered).sum(
        dim=reduce_dims, keepdim=True
    ) / variance_counts
    std_scalar = (var_scalar + eps).sqrt()
    empty = counts <= 0
    mean_scalar = torch.where(empty, torch.zeros_like(mean_scalar), mean_scalar)
    std_scalar = torch.where(empty, torch.full_like(std_scalar, eps), std_scalar)
    mean = mean_scalar.expand(-1, feat.shape[1], -1, -1)
    std = std_scalar.expand(-1, feat.shape[1], -1, -1)
    return mean, std


def adaptive_instance_normalization_mask(
    content_feat: torch.Tensor,
    style_feat: torch.Tensor,
    refer_mask: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    per_channel: bool = False,
    gain: float = 1.0,
    iterations: int = 1,
    eps: float = 1e-8,
    unbiased_variance: bool = False,
    formula_eps: float | None = None,
) -> torch.Tensor:
    dtype = content_feat.dtype
    content = content_feat.to(torch.float32)
    style = style_feat.to(torch.float32)
    refer_mask = refer_mask.to(torch.float32) if refer_mask is not None else None
    valid_mask = valid_mask.to(torch.float32) if valid_mask is not None else None
    formula_eps = eps if formula_eps is None else float(formula_eps)

    style_mean_refer, style_std_refer = _calc_mean_std_mask(
        style,
        mask=refer_mask,
        per_channel=per_channel,
        eps=eps,
        unbiased_variance=unbiased_variance,
    )
    result = content
    for _ in range(max(int(iterations), 1)):
        content_mean_refer, content_std_refer = _calc_mean_std_mask(
            result,
            mask=refer_mask,
            per_channel=per_channel,
            eps=eps,
            unbiased_variance=unbiased_variance,
        )
        content_mean, content_std = _calc_mean_std_mask(
            result,
            mask=valid_mask,
            per_channel=per_channel,
            eps=eps,
            unbiased_variance=unbiased_variance,
        )
        normalized = (result - content_mean) / content_std.clamp_min(formula_eps)
        new_mean = (style_mean_refer + formula_eps) * content_mean * gain / (content_mean_refer + formula_eps)
        new_std = (style_std_refer + formula_eps) * content_std * gain / (content_std_refer + formula_eps)
        result = normalized * new_std + new_mean
    return result.clamp(0.0, 1.0).to(dtype)


def postprocess_single_window(
    original_video: torch.Tensor,
    masked_video: torch.Tensor,
    padded_mask: torch.Tensor,
    generated_video: torch.Tensor,
    crop_bbox: tuple[int, int, int, int],
    original_num_frames: int,
    crop_height: int,
    crop_width: int,
    direct_out: bool = False,
    enable_colorfix: bool = True,
    dialate_kernel_size: int = 5,
    guss_dialate_iter: int = 20,
    guss_dialate_sigma: float = 0.8,
    remain_distance: int = 2,
    colorfix_per_channel: bool = False,
    colorfix_unbiased_variance: bool = False,
    colorfix_formula_eps: float | None = None,
    materialize_full_output: bool = True,
) -> SingleWindowPostprocessResult:
    if (
        original_video.ndim != 5
        or masked_video.ndim != 5
        or padded_mask.ndim != 5
        or generated_video.ndim != 5
    ):
        raise ValueError("original_video/masked_video/padded_mask/generated_video must be 5D [B,C,F,H,W]")
    if original_video.shape[0] != 1 or generated_video.shape[0] != 1 or masked_video.shape[0] != 1:
        raise ValueError("single-window postprocess expects batch size 1")

    crop_video_modified = generated_video[
        :,
        :,
        :original_num_frames,
        :crop_height,
        :crop_width,
    ]
    crop_mask = padded_mask[
        :,
        :,
        :original_num_frames,
        :crop_height,
        :crop_width,
    ]
    crop_mask_rgb = crop_mask.expand(-1, 3, -1, -1, -1).to(device=crop_video_modified.device, dtype=crop_video_modified.dtype)
    crop_mask_4d = crop_mask_rgb[0].permute(1, 0, 2, 3)
    paste_mask = None

    if not direct_out:
        paste_mask_4d = gaussian_dilate_mask(
            crop_mask_4d,
            dialate_iter=guss_dialate_iter,
            ksize=(dialate_kernel_size, dialate_kernel_size),
            sigma=guss_dialate_sigma,
            remain_distance=remain_distance,
        )
        crop_video_input = masked_video[
            :,
            :,
            :original_num_frames,
            :crop_height,
            :crop_width,
        ]
        crop_input_4d = crop_video_input[0].permute(1, 0, 2, 3).to(
            device=crop_video_modified.device,
            dtype=crop_video_modified.dtype,
        )
        crop_gen_4d = crop_video_modified[0].permute(1, 0, 2, 3)

        if enable_colorfix:
            refer_mask = (1.0 - crop_mask_4d).clamp(0.0, 1.0)
            crop_gen_4d = adaptive_instance_normalization_mask(
                content_feat=crop_gen_4d,
                style_feat=crop_input_4d,
                refer_mask=refer_mask,
                valid_mask=crop_mask_4d,
                per_channel=colorfix_per_channel,
                unbiased_variance=colorfix_unbiased_variance,
                formula_eps=colorfix_formula_eps,
            )

        crop_patch = (
            crop_input_4d * (1.0 - paste_mask_4d) + crop_gen_4d * paste_mask_4d
        ).to(crop_video_modified.dtype)
        crop_video_modified = crop_patch.permute(1, 0, 2, 3).unsqueeze(0)
        paste_mask = paste_mask_4d.permute(1, 0, 2, 3).unsqueeze(0)

    output_video = None
    if materialize_full_output:
        base_video = original_video[0].permute(1, 0, 2, 3).to(
            device=crop_video_modified.device, dtype=crop_video_modified.dtype
        )
        crop_patch = crop_video_modified[0].permute(1, 0, 2, 3)
        output_video = (
            splice_video(base_video, crop_patch, crop_bbox[:2])
            .permute(1, 0, 2, 3)
            .unsqueeze(0)
            .to(device=original_video.device, dtype=original_video.dtype)
        )
    return SingleWindowPostprocessResult(
        crop_video_modified=crop_video_modified,
        output_video=output_video,
        paste_mask=paste_mask,
    )
