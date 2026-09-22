"""EraserDiT window output postprocessing.

Port of ``utils/post.py:post_stream_normalized`` +
``utils/post_pkg.py:torch_nhwc_to_video_stream`` from the frozen baseline at
commit ``9944867``.

Two behaviours are load-bearing and must not be "cleaned up":

* There is **no** ``orig*(1-mask) + gen*mask`` compositing.  Everything outside
  the erase region is model output as well, merely colour-aligned by a whole-frame
  per-channel AdaIN.
* The reference-mask argument is ``1 - mask_ori`` with ``mask_ori`` in raw 0..255
  scale.  Every non-zero value (including ``1 - 255 = -254``) casts to ``True``, so
  the AdaIN statistics are whole-frame.  Reproduced as-is.
"""

from __future__ import annotations

import torch

from utils.colorfix_wmask import adaptive_instance_normalization_mask

__all__ = ["quantize_like_baseline", "eraserdit_window_output"]


def quantize_like_baseline(frames: torch.Tensor) -> torch.Tensor:
    """``(x * 255).to(uint8)`` as the baseline performs it, returned as floats.

    PyTorch's float -> uint8 cast truncates toward zero for in-range values, which
    is *not* the rounding ``utils/video_io.frames_tensor_to_uint8`` applies on the
    write path.  Pre-quantising here keeps the adapter's frames on exact
    ``k/255`` values so the writer's rounding is a no-op.
    """
    return (frames * 255).to(torch.uint8).to(torch.float32) / 255.0


def eraser_dit_window_output(
    generated: torch.Tensor,
    style: torch.Tensor,
    mask_ori: torch.Tensor,
    *,
    colorfix_type: str = "RGB",
    per_channel: bool = True,
) -> torch.Tensor:
    """Colour-align a window's generated frames and quantise them as the baseline does.

    ``generated`` is ``[F, C, H, W]`` model output already cropped to the original
    frame count and spatial size; ``style`` is the corresponding ``[F, C, H, W]``
    source window in ``[0, 1]``; ``mask_ori`` is the raw window mask in 0..255
    scale.  Returns ``[F, C, H, W]`` floats holding exact ``k/255`` values.
    """
    if generated.ndim != 4 or style.ndim != 4 or mask_ori.ndim != 4:
        raise ValueError("generated/style/mask_ori must be 4D [F,C,H,W]")

    # The baseline runs this on CPU: ``torch_nhwc_to_video_stream`` starts with
    # ``tensor.cpu()`` and ``video_ori``/``mask_ori`` come from a CPU decord
    # reader, while the statistics helper only sends the flat feature view to the
    # GPU.  Running it on GPU tensors trips the CPU/GPU mix inside that helper.
    generated = generated.cpu()
    style = style.cpu()
    mask_ori = mask_ori.cpu()

    quantised = quantize_like_baseline(generated)
    aligned = adaptive_instance_normalization_mask(
        content_feat=quantised,
        style_feat=style,
        refer_mask=(1 - mask_ori),
        valid_mask=None,
        type=colorfix_type,
        per_channel=per_channel,
    )
    return quantize_like_baseline(aligned)
