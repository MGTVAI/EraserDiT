"""EraserDiT erase – window postprocess stage.

Builds the frame block the commit machinery splices into the output stream:

* ``[prefix_len : prefix_len + new_frames]`` -- colour-aligned, 8-bit-quantised
  frames, which is exactly what the baseline writes for this window.
* the whole block -- the baseline's ``pre_video_shift`` role is *not* taken from
  here; the raw (pre-colour-fix) tail is stashed separately so the next window's
  model input matches ``output_frames[-shift_alpha:]``.

The committed pixels still come from this tensor, so the overlap region that the
runtime copies forward is the colour-aligned version -- matching the baseline,
which writes the colour-aligned frames and feeds the raw ones forward.
"""

from __future__ import annotations

import torch

from config.server_args import ServerArgs
from nodes.schedule_batch import Req
from nodes.stages.base import PipelineStage
from nodes.stages.model_specific_stages.eraserdit_erase._common import (
    NEW_FRAMES_KEY,
    ORIG_SIZE_KEY,
    PREFIX_LEN_KEY,
    STYLE_MASK_KEY,
    STYLE_VIDEO_KEY,
    field_summary,
    get_task_state,
)
from utils.eraserdit_postprocess import eraser_dit_window_output


class EraserDiTEraseWindowPostprocessStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        spec = batch.extra.get("window_spec") or {}
        decoded = batch.decoded_video
        if decoded is None:
            raise ValueError("window postprocess requires decoding to run first")

        state = get_task_state(batch)
        prefix_len = int(batch.extra[PREFIX_LEN_KEY])
        new_frames = int(batch.extra[NEW_FRAMES_KEY])
        orig_h, orig_w = batch.extra[ORIG_SIZE_KEY]
        style_video = batch.extra[STYLE_VIDEO_KEY]
        style_mask = batch.extra[STYLE_MASK_KEY]
        input_len = int(batch.padded_video.shape[2])

        # [1, C, F, H, W] -> [F, C, H, W] cropped to the original frame extent.
        decoded_frames = decoded[0].permute(1, 0, 2, 3).to(torch.float32)
        aligned_h, aligned_w = int(decoded_frames.shape[-2]), int(decoded_frames.shape[-1])

        # Tail retained for the *next* window's model input: raw, un-colour-fixed,
        # still at the aligned spatial size (baseline keeps the whole padded frame).
        overlap_right = int(spec.get("overlap_right", 0))
        if overlap_right > 0:
            # Frames are dim 0: this is the baseline's ``output_frames[-shift_alpha:]``.
            state.prev_raw_tail = decoded_frames[-overlap_right:].detach().clone()

        # Baseline writes ``output_frames[shift_alpha:][:ori_frames]``: the model
        # renders the whole prefix + padded-new window, but only the newly loaded
        # frames are colour-fixed and committed.
        window_frames = decoded_frames[prefix_len : prefix_len + new_frames]
        if window_frames.shape[0] != new_frames:
            raise ValueError(
                f"decoded window has {window_frames.shape[0]} frames after the "
                f"prefix, expected {new_frames}"
            )
        # Already [F, C, H, W], the layout the baseline hands to the colour fix.
        generated = window_frames[..., :orig_h, :orig_w].contiguous()

        aligned = eraser_dit_window_output(
            generated,
            style_video.to(generated.device, torch.float32),
            style_mask.to(generated.device, torch.float32),
            colorfix_type=str(batch.colorfix_type),
            per_channel=bool(batch.colorfix_per_channel),
        )

        # The prefix block is always replaced by the commit's overlap blend
        # (mode "before" restores the previously committed pixels), so it is left
        # at zero here.
        patch = torch.zeros(
            (input_len, 3, orig_h, orig_w),
            dtype=torch.float32,
            device=decoded.device,
        )
        patch[prefix_len : prefix_len + new_frames] = aligned.to(
            device=decoded.device, dtype=patch.dtype
        )

        batch.crop_video_modified = patch.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        batch.output_video = None
        batch.output = batch.crop_video_modified
        batch.decoded_video = None
        state.windows_seen += 1
        self.log_info(
            "%s | prefix=%d new=%d",
            field_summary("crop_video_modified", batch.crop_video_modified),
            prefix_len,
            new_frames,
        )
        return batch
