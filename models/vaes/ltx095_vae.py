"""Local LTX0.9.5 VAE wrapper for the minimal runtime."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from diffusers.models.autoencoders.autoencoder_kl_ltx import (
    AutoencoderKLLTXVideo as _DiffusersAutoencoderKLLTXVideo,
)
from diffusers.models.autoencoders.vae import (
    DecoderOutput,
    DiagonalGaussianDistribution,
)
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.utils.accelerate_utils import apply_forward_hook


class AutoencoderKLLTXVideo(_DiffusersAutoencoderKLLTXVideo):
    """Official LTX VAE with a thin, serial-only compatibility wrapper."""

    supports_official_parallel = False

    @property
    def scaling_factor(self) -> float:
        return float(self.config.scaling_factor)

    @property
    def device(self):
        return next(self.parameters()).device

    @apply_forward_hook
    def encode(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        if self.use_slicing and x.shape[0] > 1:
            h = torch.cat([self._encode(x_slice) for x_slice in x.split(1)])
        else:
            h = self._encode(x)

        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(
        self,
        z: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        if self.use_slicing and z.shape[0] > 1:
            temb_slices = temb.split(1) if temb is not None else (None,) * z.shape[0]
            decoded = torch.cat(
                [
                    self._decode(z_slice, temb_slice).sample
                    for z_slice, temb_slice in zip(
                        z.split(1),
                        temb_slices,
                        strict=True,
                    )
                ]
            )
        else:
            decoded = self._decode(z, temb).sample

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)


EntryClass = AutoencoderKLLTXVideo
