"""EraserDiT window preprocessing.

Semantics-preserving port of ``VideoInpaintPre`` (``utils/pre.py``) from the frozen baseline
at commit ``9944867``: 32-pixel edge alignment, the mirrored tail padding rule and
the ``mask_video_nchw`` combination of video masking and mask compression.

The baseline works on uint8 frames and divides by 255 at the very end; every step
here is linear in the frame values, so the runtime may hand in ``[0, 1]`` floats
and skip the division.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.eraserdit_mask import (
    ERASERDIT_MASK_CHANNELS,
    binarize_and_dilate,
    compress_mask_temporal,
    gray_normalize_mask,
)

__all__ = [
    "EraserDiTWindowPreprocess",
    "align_nchw",
    "compact_tail_infer_len",
    "expand_mask_channels",
    "pad_window_frames",
    "preprocess_eraserdit_window",
    "window_pad_plan",
]


@dataclass
class EraserDiTWindowPreprocess:
    """Result of preprocessing one window's newly-loaded frames."""

    # [F, C, H, W] masked video, mask region zeroed, at the model frame length.
    masked_video: torch.Tensor
    # [L, 1, H, W] grey-compressed mask at latent resolution.
    mask_latents: torch.Tensor
    # Frame length after alignment/mirror padding (the model's frame count).
    padded_frames: int


def expand_mask_channels(mask: torch.Tensor) -> torch.Tensor:
    """Repeat a single-channel mask to the three channels the grey kernel expects."""
    if mask.shape[1] == ERASERDIT_MASK_CHANNELS:
        return mask
    if mask.shape[1] != 1:
        raise ValueError(f"unexpected mask channel count {mask.shape[1]}")
    return mask.repeat(1, ERASERDIT_MASK_CHANNELS, 1, 1)


def align_nchw(video: torch.Tensor, align_h: int, align_w: int) -> torch.Tensor:
    """Right/bottom edge-replicate pad to a multiple of ``align_h`` / ``align_w``."""
    if video.ndim != 4:
        raise ValueError(f"expected 4D [N,C,H,W], got {tuple(video.shape)}")
    _, _, height, width = video.shape
    height_pad = math.ceil(height / align_h) * align_h - height
    width_pad = math.ceil(width / align_w) * align_w - width
    if height_pad == 0 and width_pad == 0:
        return video
    return F.pad(video, (0, width_pad, 0, height_pad), mode="replicate")


def window_pad_plan(
    num_frames: int,
    *,
    head_batch: bool,
    infer_len: int,
    shift_alpha: int,
) -> tuple[int, int]:
    """``(batch_size, num_frames_padded)`` from ``VideoInpaintPre.__call__``."""
    wrapping_size = infer_len - shift_alpha
    if not head_batch:
        batch_size = math.ceil(num_frames / wrapping_size)
        num_frames_padded = batch_size * wrapping_size - num_frames
    else:
        batch_size = max(1, math.ceil((num_frames - shift_alpha) / wrapping_size))
        num_frames_padded = (batch_size * wrapping_size + shift_alpha) - num_frames
    return int(batch_size), int(num_frames_padded)


def compact_tail_infer_len(num_new_frames: int, *, head_batch: bool, infer_len: int,
                           overlap: int) -> int:
    """Keep real frames and the overlap; remove only excess mirrored context.

    The first/full windows retain their original shape. A non-head window adds
    an 8-aligned number of new frames to its 8k+1 overlap prefix.
    """
    if head_batch:
        return infer_len
    return min(infer_len, overlap + max(8, ((num_new_frames + 7) // 8) * 8))


def pad_window_frames(
    video: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_frames_padded: int,
    head_batch: bool,
    infer_len: int,
    shift_alpha: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append mirrored frames ``flip(0)[1:-1]`` until the window frame count fits.

    Both end points are excluded from the mirror, so a single padded frame copies
    the second-to-last source frame.  When more frames are needed than the mirror
    can supply the mirrored block is repeated and truncated to the window length.
    """
    if num_frames_padded <= 0:
        return video, mask

    mirror_video = video.flip(0)[1 : len(video) - 1]
    mirror_mask = mask.flip(0)[1 : len(mask) - 1]

    if num_frames_padded <= len(video) - 2:
        video = torch.cat([video, mirror_video[:num_frames_padded]], dim=0)
        mask = torch.cat([mask, mirror_mask[:num_frames_padded]], dim=0)
        return video, mask

    video = torch.cat([video, mirror_video], dim=0)
    mask = torch.cat([mask, mirror_mask], dim=0)
    inf_len = infer_len if head_batch else infer_len - shift_alpha
    repeat_num = max(1, math.ceil(inf_len / len(video)))
    video = video.repeat(repeat_num, 1, 1, 1)[:inf_len]
    mask = mask.repeat(repeat_num, 1, 1, 1)[:inf_len]
    return video, mask


def preprocess_eraserdit_window(
    video: torch.Tensor,
    mask: torch.Tensor,
    *,
    head_batch: bool,
    infer_len: int = 121,
    shift_alpha: int = 9,
    align_h: int = 32,
    align_w: int = 32,
    ksize: tuple[int, int] = (9, 9),
    dilate_iter: int = 9,
    threshold: float = 0.039,
    enable_approximate: bool = True,
    mask_chunk_frames: int = 8,
) -> EraserDiTWindowPreprocess:
    """Align, mirror-pad, dilate and compress one window's newly-loaded frames.

    ``video`` / ``mask`` are ``[N, C, H, W]``; the video holds ``[0, 1]`` values
    (the baseline's ``/255`` already applied) and the mask is the raw mask binarised
    to ``{0, 1}``.  ``head_batch`` selects the first-window branch of the baseline's
    frame accounting.

    ``mask_chunk_frames`` bounds spatial morphology and masking temporaries;
    temporal compression still runs on the complete padded sequence.
    """
    if video.ndim != 4 or mask.ndim != 4:
        raise ValueError("video and mask must be 4D [N,C,H,W]")
    if video.shape[0] != mask.shape[0]:
        raise ValueError("video and mask frame counts differ")
    if mask_chunk_frames < 1:
        raise ValueError("mask_chunk_frames must be positive")

    video = align_nchw(video, align_h, align_w)
    mask = align_nchw(mask, align_w=align_w, align_h=align_h)

    _, num_frames_padded = window_pad_plan(
        video.shape[0], head_batch=head_batch, infer_len=infer_len, shift_alpha=shift_alpha
    )
    video, mask = pad_window_frames(
        video,
        mask,
        num_frames_padded=num_frames_padded,
        head_batch=head_batch,
        infer_len=infer_len,
        shift_alpha=shift_alpha,
    )

    # The baseline keeps the mask in raw 0..255 so that the binarisation threshold
    # ``255/2*0.039`` is meaningful; the runtime delivers {0, 1}.
    # Spatial morphology is independent for each frame. Bound convolution and
    # threshold temporaries while preserving RGB grey-compression semantics.
    dilated = torch.empty(
        (len(mask), ERASERDIT_MASK_CHANNELS, *mask.shape[-2:]),
        dtype=torch.uint8, device=mask.device,
    )
    masked_video = torch.empty_like(video)
    for start in range(0, len(mask), mask_chunk_frames):
        stop = start + mask_chunk_frames
        chunk = binarize_and_dilate(
            expand_mask_channels(mask[start:stop]) * 255.0,
            ksize=ksize,
            dilate_iter=dilate_iter,
            threshold=threshold,
            enable_approximate=enable_approximate,
        )
        dilated[start:stop].copy_(chunk)
        masked_video[start:stop].copy_(video[start:stop] * (1 - chunk.to(video.dtype)))

    compressed = compress_mask_temporal(dilated, head_batch=head_batch)
    mask_latents = gray_normalize_mask(compressed)

    return EraserDiTWindowPreprocess(
        masked_video=masked_video,
        mask_latents=mask_latents,
        padded_frames=int(video.shape[0]),
    )
