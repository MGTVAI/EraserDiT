"""Bound VAE operator temporaries without splitting receptive fields."""

import torch
from diffusers.models.normalization import RMSNorm


class ChunkedRMSNorm(RMSNorm):
    """Run the original channel reduction on bounded groups of video frames.

    VAE callers pass NTHWC views. No reduction crosses the temporal axis, so
    every chunk keeps the complete normalization dimension and original dtype
    conversions. The output retains the input layout for subsequent convolutions.
    Training and small tensors use the upstream implementation.
    """

    chunk_size = 0

    def forward(self, hidden_states):
        if (not self.chunk_size or torch.is_grad_enabled()
                or hidden_states.ndim != 5 or hidden_states.numel() <= self.chunk_size):
            return super().forward(hidden_states)
        frames = hidden_states.shape[1]
        per_frame = hidden_states.numel() // frames
        step = max(1, self.chunk_size // per_frame)
        output = None
        for start in range(0, frames, step):
            value = super().forward(hidden_states[:, start:start + step])
            if output is None:
                output = torch.empty_like(hidden_states, dtype=value.dtype)
            output[:, start:start + step].copy_(value)
            del value
        return output


def configure_vae_memory(vae, enabled):
    from models.vaes.eraserdit_vae import LTXVideoCausalConv3d
    for module in vae.modules():
        if isinstance(module, (ChunkedRMSNorm, LTXVideoCausalConv3d)):
            module.chunk_size = 16 * 1024 * 1024 if enabled else 0


def chunked_causal_conv(module, inputs):
    """Evaluate Conv3d output slices using their exact temporal receptive field.

    Padding matches LTXVideoCausalConv3d (replicate at global time boundaries).
    Interior slices read the original neighboring frames, never replicate chunk
    boundaries. Spatial dimensions and padding remain unchanged. cuDNN can choose
    a different kernel for these shapes, so BF16 results need quality validation.
    """
    conv = module.conv
    kernel = module.kernel_size[0]
    left = kernel - 1 if module.is_causal else (kernel - 1) // 2
    right = 0 if module.is_causal else (kernel - 1) // 2
    frames = inputs.shape[2]
    extent = conv.dilation[0] * (kernel - 1) + 1
    stride = conv.stride[0]
    output_frames = (frames + left + right - extent) // stride + 1
    per_frame = inputs.numel() // frames
    step = max(1, module.chunk_size // per_frame)
    output = None
    for start in range(0, output_frames, step):
        end = min(start + step, output_frames)
        first = start * stride - left
        last = (end - 1) * stride - left + extent
        value = inputs[:, :, max(0, first):min(frames, last)]
        parts = []
        if first < 0:
            parts.append(inputs[:, :, :1].expand(-1, -1, -first, -1, -1))
        parts.append(value)
        if last > frames:
            parts.append(inputs[:, :, -1:].expand(-1, -1, last - frames, -1, -1))
        value = torch.cat(parts, dim=2) if len(parts) > 1 else value
        value = conv(value)
        if output is None:
            shape = list(value.shape)
            shape[2] = output_frames
            output = value.new_empty(shape)
        output[:, :, start:end].copy_(value)
        del value, parts
    return output
