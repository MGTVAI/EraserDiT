"""Local LTX0.9.5 transformer wrapper for the minimal runtime."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers import transformer_ltx as _diffusers_transformer_ltx
from diffusers.models.transformers.transformer_ltx import (
    LTXVideoTransformer3DModel as _DiffusersLTXVideoTransformer3DModel,
)

from config.server_args import get_global_server_args
from layers.attention import SequenceParallelMetadata
from layers.operator_fusion.config import RMSNORM_ADALN_OP
from layers.operator_fusion.registry import (
    OperatorFusionDecision,
    get_operator_fusion_decision,
)
from layers.operator_fusion.rmsnorm_adaln import (
    apply_fused_rmsnorm_adaln,
)
from models.dits.ltx095_attention import (
    LTX095SequenceParallelAttentionContext,
    LTXVideo2VideoAttentionProcessor2_0,
)
from models.dits.ltx095_parallel import (
    LTX095SequenceParallelControlBinding,
    create_ltx095_sequence_parallel_control_binding,
    synchronize_ltx095_sequence_parallel_phase,
    synchronize_ltx095_sequence_parallel_preflight,
)

if TYPE_CHECKING:
    from distributed.group_coordinator import GroupCoordinator


@dataclass
class _LTX095ForwardPrefix:
    hidden_states: torch.Tensor
    lora_scale: float
    video_mode: bool
    num_frames: int | None
    height: int | None
    width: int | None


@dataclass
class _LTX095ForwardPreparation:
    hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor
    encoder_attention_mask: torch.Tensor | None
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor]
    temb: torch.Tensor
    embedded_timestep: torch.Tensor
    lora_scale: float
    video_mode: bool
    num_frames: int | None
    height: int | None
    width: int | None
    sequence_parallel_context: LTX095SequenceParallelAttentionContext | None


def normalize_latents(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return (latents - latents_mean) * scaling_factor / latents_std


def denormalize_latents(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return latents * latents_std / scaling_factor + latents_mean


def pack_latents(
    latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1
) -> torch.Tensor:
    if latents.ndim != 5:
        raise ValueError(f"Expected a 5D tensor, got shape {tuple(latents.shape)}")

    batch_size, _num_channels, num_frames, height, width = latents.shape
    if (
        num_frames % patch_size_t != 0
        or height % patch_size != 0
        or width % patch_size != 0
    ):
        raise ValueError(
            "Input shape must be divisible by patch sizes "
            f"(patch_size={patch_size}, patch_size_t={patch_size_t}), got {tuple(latents.shape)}."
        )

    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(
        batch_size,
        -1,
        post_patch_num_frames,
        patch_size_t,
        post_patch_height,
        patch_size,
        post_patch_width,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def unpack_latents(
    latents: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> torch.Tensor:
    batch_size = latents.size(0)
    latents = latents.reshape(
        batch_size,
        num_frames,
        height,
        width,
        -1,
        patch_size_t,
        patch_size,
        patch_size,
    )
    latents = (
        latents.permute(0, 4, 1, 5, 2, 6, 3, 7)
        .flatten(6, 7)
        .flatten(4, 5)
        .flatten(2, 3)
    )
    return latents


def _execute_block_with_rmsnorm_adaln(
    block: torch.nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    encoder_attention_mask: torch.Tensor | None,
    *,
    decision: OperatorFusionDecision,
) -> torch.Tensor:
    """Run the upstream LTX block with only its two AdaLN sites optimized."""

    batch_size = hidden_states.size(0)
    norm_hidden_states = block.norm1(hidden_states)

    num_ada_params = block.scale_shift_table.shape[0]
    ada_values = block.scale_shift_table[None, None] + temb.reshape(
        batch_size,
        temb.size(1),
        num_ada_params,
        -1,
    )
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        ada_values.unbind(dim=2)
    )
    norm_hidden_states = apply_fused_rmsnorm_adaln(
        norm_hidden_states,
        scale_msa,
        shift_msa,
        block.norm1,
        decision=decision,
        reference=lambda: norm_hidden_states * (1 + scale_msa) + shift_msa,
    )

    attn_hidden_states = block.attn1(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=None,
        image_rotary_emb=image_rotary_emb,
    )
    hidden_states = hidden_states + attn_hidden_states * gate_msa

    attn_hidden_states = block.attn2(
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        image_rotary_emb=None,
        attention_mask=encoder_attention_mask,
    )
    hidden_states = hidden_states + attn_hidden_states
    norm_hidden_states = block.norm2(hidden_states)
    norm_hidden_states = apply_fused_rmsnorm_adaln(
        norm_hidden_states,
        scale_mlp,
        shift_mlp,
        block.norm2,
        decision=decision,
        reference=lambda: norm_hidden_states * (1 + scale_mlp) + shift_mlp,
    )

    ff_output = block.ff(norm_hidden_states)
    hidden_states = hidden_states + ff_output * gate_mlp
    return hidden_states


class LTXVideoTransformer3DModel(_DiffusersLTXVideoTransformer3DModel):
    """Official LTX transformer with local video/mask packing support."""

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 2048,
        num_layers: int = 28,
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 4096,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            num_layers=num_layers,
            activation_fn=activation_fn,
            qk_norm=qk_norm,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            caption_channels=caption_channels,
            attention_bias=attention_bias,
            attention_out_bias=attention_out_bias,
        )
        self.in_channels = self.config.in_channels
        self.out_channels = self.config.out_channels
        self.hidden_size = (
            self.config.num_attention_heads * self.config.attention_head_dim
        )
        self.num_attention_heads = self.config.num_attention_heads
        self.num_channels_latents = self.config.out_channels
        self.transformer_spatial_patch_size = self.config.patch_size
        self.transformer_temporal_patch_size = self.config.patch_size_t
        shared_attention_processor = LTXVideo2VideoAttentionProcessor2_0()
        self._shared_attention_processor = shared_attention_processor
        self.operator_fusion_decision = get_operator_fusion_decision(
            get_global_server_args()
        )
        for block in self.transformer_blocks:
            block.attn1.set_processor(shared_attention_processor)
            block.attn2.set_processor(shared_attention_processor)
        self._sequence_parallel_control_binding: (
            LTX095SequenceParallelControlBinding | None
        ) = None
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def sequence_parallel_control_binding(
        self,
    ) -> LTX095SequenceParallelControlBinding | None:
        return self._sequence_parallel_control_binding

    def bind_sequence_parallel_control(
        self,
        coordinator: GroupCoordinator,
    ) -> None:
        """Freeze the trusted control group once, after model device placement."""
        if self._sequence_parallel_control_binding is not None:
            raise RuntimeError("sequence-parallel control is already bound")
        self._sequence_parallel_control_binding = (
            create_ltx095_sequence_parallel_control_binding(
                coordinator,
                control_device=self.device,
            )
        )

    def __getstate__(self) -> dict[str, object]:
        state = super().__getstate__()
        state["_sequence_parallel_control_binding"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        super().__setstate__(state)
        self._sequence_parallel_control_binding = None

    def _merge_video_inputs(
        self,
        hidden_states: torch.Tensor,
        cond_latents: torch.Tensor | None,
        mask_values: torch.Tensor | None,
    ) -> tuple[torch.Tensor, bool, int, int, int]:
        if hidden_states.ndim == 5:
            tensors = [hidden_states]
            if cond_latents is not None:
                if cond_latents.ndim != 5:
                    raise ValueError("cond_latents must match hidden_states rank")
                tensors.append(cond_latents)
            if mask_values is not None:
                if mask_values.ndim != 5:
                    raise ValueError("mask_values must match hidden_states rank")
                tensors.append(mask_values)

            merged = torch.cat(tensors, dim=1)
            packed = pack_latents(
                merged, self.config.patch_size, self.config.patch_size_t
            )
            if packed.shape[-1] != self.config.in_channels:
                raise ValueError(
                    "Packed input feature size does not match model config: "
                    f"{packed.shape[-1]} != {self.config.in_channels}"
                )
            return (
                packed,
                True,
                hidden_states.shape[2],
                hidden_states.shape[3],
                hidden_states.shape[4],
            )

        if hidden_states.ndim == 3:
            tensors = [hidden_states]
            if cond_latents is not None:
                if cond_latents.ndim != 3:
                    raise ValueError("cond_latents must match hidden_states rank")
                tensors.append(cond_latents)
            if mask_values is not None:
                if mask_values.ndim != 3:
                    raise ValueError("mask_values must match hidden_states rank")
                tensors.append(mask_values)
            merged = torch.cat(tensors, dim=-1)
            return merged, False, -1, -1, -1

        raise ValueError(f"Unsupported hidden_states rank: {hidden_states.ndim}")

    def sequence_latent(self, latents: torch.Tensor) -> torch.Tensor:
        return pack_latents(
            latents,
            patch_size=self.config.patch_size,
            patch_size_t=self.config.patch_size_t,
        )

    def unsequence_latent(
        self,
        latents: torch.Tensor,
        *,
        num_frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        return unpack_latents(
            latents,
            num_frames=num_frames,
            height=height,
            width=width,
            patch_size=self.config.patch_size,
            patch_size_t=self.config.patch_size_t,
        )

    def _prepare_forward_prefix(
        self,
        hidden_states: torch.Tensor,
        *,
        num_frames: int | None,
        height: int | None,
        width: int | None,
        attention_kwargs: dict[str, Any] | None,
        cond_latents: torch.Tensor | None,
        mask_values: torch.Tensor | None,
        sequence_parallel_metadata: SequenceParallelMetadata | None,
        sequence_parallel_coordinator: GroupCoordinator | None,
        sequence_parallel_active: bool,
    ) -> _LTX095ForwardPrefix:
        if sequence_parallel_active:
            if (sequence_parallel_metadata is None) != (
                sequence_parallel_coordinator is None
            ):
                raise ValueError(
                    "sequence_parallel_metadata and sequence_parallel_coordinator "
                    "must be provided together"
                )
            if sequence_parallel_metadata is None:
                raise ValueError(
                    "active sequence parallel requires per-call metadata and "
                    "data coordinator"
                )
        elif (
            sequence_parallel_metadata is not None
            or sequence_parallel_coordinator is not None
        ):
            raise RuntimeError(
                "bind sequence-parallel control before providing per-call data"
            )

        video_mode = hidden_states.ndim == 5
        if video_mode:
            hidden_states, _, inferred_num_frames, inferred_height, inferred_width = (
                self._merge_video_inputs(hidden_states, cond_latents, mask_values)
            )
            num_frames = inferred_num_frames if num_frames is None else num_frames
            height = inferred_height if height is None else height
            width = inferred_width if width is None else width
        elif cond_latents is not None or mask_values is not None:
            hidden_states, _, _, _, _ = self._merge_video_inputs(
                hidden_states, cond_latents, mask_values
            )

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        return _LTX095ForwardPrefix(
            hidden_states=hidden_states,
            lora_scale=lora_scale,
            video_mode=video_mode,
            num_frames=num_frames,
            height=height,
            width=width,
        )

    def _prepare_forward_inputs(
        self,
        prefix: _LTX095ForwardPrefix,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor | None,
        *,
        rope_interpolation_scale: tuple[float, float, float] | torch.Tensor | None,
        video_coords: torch.Tensor | None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        sequence_parallel_metadata: SequenceParallelMetadata | None,
        sequence_parallel_coordinator: GroupCoordinator | None,
        sequence_parallel_active: bool,
    ) -> _LTX095ForwardPreparation:
        hidden_states = prefix.hidden_states
        if image_rotary_emb is None:
            image_rotary_emb = self.rope(
                hidden_states,
                prefix.num_frames,
                prefix.height,
                prefix.width,
                rope_interpolation_scale,
                video_coords,
            )

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        batch_size = hidden_states.size(0)
        hidden_states = self.proj_in(hidden_states)

        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(
            batch_size, -1, embedded_timestep.size(-1)
        )

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(
            batch_size, -1, hidden_states.size(-1)
        )

        sequence_parallel_context = None
        if sequence_parallel_active:
            first_attention = (
                self.transformer_blocks[0].attn1
                if len(self.transformer_blocks) > 0
                else None
            )
            sequence_parallel_context = (
                self._shared_attention_processor.prepare_sequence_parallel_call(
                    first_attention,
                    hidden_states,
                    image_rotary_emb,
                    sequence_parallel_metadata,
                    sequence_parallel_coordinator,
                    self._sequence_parallel_control_binding,
                )
            )

        return _LTX095ForwardPreparation(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            image_rotary_emb=image_rotary_emb,
            temb=temb,
            embedded_timestep=embedded_timestep,
            lora_scale=prefix.lora_scale,
            video_mode=prefix.video_mode,
            num_frames=prefix.num_frames,
            height=prefix.height,
            width=prefix.width,
            sequence_parallel_context=sequence_parallel_context,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        rope_interpolation_scale: Optional[
            Union[Tuple[float, float, float], torch.Tensor]
        ] = None,
        video_coords: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        cond_latents: torch.Tensor | None = None,
        mask_values: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        teacache: Any | None = None,
        cache_dit: Any | None = None,
        time_stemp_index: int = 0,
        sequence_parallel_metadata: SequenceParallelMetadata | None = None,
        sequence_parallel_coordinator: GroupCoordinator | None = None,
    ) -> torch.Tensor:
        control_binding = self._sequence_parallel_control_binding
        preparation = None
        lora_scale = 1.0
        lora_scaled = False
        try:
            if control_binding is None:
                prefix = self._prepare_forward_prefix(
                    hidden_states,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    attention_kwargs=attention_kwargs,
                    cond_latents=cond_latents,
                    mask_values=mask_values,
                    sequence_parallel_metadata=sequence_parallel_metadata,
                    sequence_parallel_coordinator=sequence_parallel_coordinator,
                    sequence_parallel_active=False,
                )
                lora_scale = prefix.lora_scale
                lora_scaled = _diffusers_transformer_ltx.USE_PEFT_BACKEND
                if lora_scaled:
                    _diffusers_transformer_ltx.scale_lora_layers(self, lora_scale)
                elif attention_kwargs is not None and "scale" in attention_kwargs:
                    _diffusers_transformer_ltx.logger.warning(
                        "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                    )
                preparation = self._prepare_forward_inputs(
                    prefix,
                    encoder_hidden_states,
                    timestep,
                    encoder_attention_mask,
                    rope_interpolation_scale=rope_interpolation_scale,
                    video_coords=video_coords,
                    image_rotary_emb=image_rotary_emb,
                    sequence_parallel_metadata=sequence_parallel_metadata,
                    sequence_parallel_coordinator=sequence_parallel_coordinator,
                    sequence_parallel_active=False,
                )
            else:
                local_preflight_error: BaseException | None = None
                try:
                    prefix = self._prepare_forward_prefix(
                        hidden_states,
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        attention_kwargs=attention_kwargs,
                        cond_latents=cond_latents,
                        mask_values=mask_values,
                        sequence_parallel_metadata=sequence_parallel_metadata,
                        sequence_parallel_coordinator=sequence_parallel_coordinator,
                        sequence_parallel_active=True,
                    )
                    lora_scale = prefix.lora_scale
                    lora_scaled = _diffusers_transformer_ltx.USE_PEFT_BACKEND
                    if lora_scaled:
                        _diffusers_transformer_ltx.scale_lora_layers(
                            self,
                            lora_scale,
                        )
                    elif attention_kwargs is not None and "scale" in attention_kwargs:
                        _diffusers_transformer_ltx.logger.warning(
                            "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                        )
                    preparation = self._prepare_forward_inputs(
                        prefix,
                        encoder_hidden_states,
                        timestep,
                        encoder_attention_mask,
                        rope_interpolation_scale=rope_interpolation_scale,
                        video_coords=video_coords,
                        image_rotary_emb=image_rotary_emb,
                        sequence_parallel_metadata=sequence_parallel_metadata,
                        sequence_parallel_coordinator=sequence_parallel_coordinator,
                        sequence_parallel_active=True,
                    )
                except BaseException as error:
                    local_preflight_error = error

                # This one collective covers local input preparation and the
                # static contract before the first Ulysses kernel. Runtime
                # kernel/OOM failures after it succeeds are outside this boundary.
                synchronize_ltx095_sequence_parallel_preflight(
                    local_preflight_error,
                    binding=control_binding,
                )
                if preparation is None:
                    raise AssertionError(
                        "sequence-parallel preparation did not complete"
                    )

            hidden_states = preparation.hidden_states
            encoder_hidden_states = preparation.encoder_hidden_states
            encoder_attention_mask = preparation.encoder_attention_mask
            image_rotary_emb = preparation.image_rotary_emb
            temb = preparation.temb
            embedded_timestep = preparation.embedded_timestep
            video_mode = preparation.video_mode
            num_frames = preparation.num_frames
            height = preparation.height
            width = preparation.width
            sequence_parallel_context = preparation.sequence_parallel_context

            if teacache is not None and cache_dit is not None:
                raise ValueError("TeaCache and Cache-DiT are request-level mutually exclusive")

            def execute_block(
                block: torch.nn.Module,
                current_hidden_states: torch.Tensor,
            ) -> torch.Tensor:
                if RMSNORM_ADALN_OP in self.operator_fusion_decision.effective_ops:
                    return _execute_block_with_rmsnorm_adaln(
                        block,
                        current_hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        encoder_attention_mask,
                        decision=self.operator_fusion_decision,
                    )
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    return self._gradient_checkpointing_func(
                        block,
                        current_hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        encoder_attention_mask,
                    )
                return block(
                    hidden_states=current_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                )

            def execute_block_range(
                start: int,
                end: int,
                current_hidden_states: torch.Tensor,
            ) -> torch.Tensor:
                if not 0 <= start <= end <= len(self.transformer_blocks):
                    raise ValueError(
                        f"invalid LTX block range [{start}, {end}) for "
                        f"{len(self.transformer_blocks)} blocks"
                    )
                for block_index in range(start, end):
                    block = self.transformer_blocks[block_index]
                    if sequence_parallel_context is None:
                        current_hidden_states = execute_block(
                            block, current_hidden_states
                        )
                        continue

                    local_block_error: BaseException | None = None
                    with self._shared_attention_processor.sequence_parallel_block_scope(
                        block_index
                    ) as block_control:
                        try:
                            current_hidden_states = execute_block(
                                block, current_hidden_states
                            )
                        except BaseException as error:
                            local_block_error = error

                        if not block_control.before_first_collective_joined:
                            try:
                                self._shared_attention_processor.synchronize_before_collective(
                                    1,
                                    local_block_error,
                                )
                            except BaseException as error:
                                if local_block_error is None:
                                    local_block_error = error

                        if not block_control.before_second_collective_joined:
                            try:
                                self._shared_attention_processor.synchronize_before_collective(
                                    2,
                                    local_block_error,
                                )
                            except BaseException as error:
                                if local_block_error is None:
                                    local_block_error = error

                        synchronize_ltx095_sequence_parallel_phase(
                            local_block_error,
                            binding=control_binding,
                            phase=f"block {block_index} completion",
                        )
                return current_hidden_states

            with ExitStack() as sequence_parallel_stack:
                if sequence_parallel_context is not None:
                    sequence_parallel_stack.enter_context(
                        self._shared_attention_processor.sequence_parallel_scope(
                            sequence_parallel_context,
                        )
                    )

                if teacache is not None:
                    input_hidden_states = hidden_states
                    skip_blocks = teacache.check(
                        step=time_stemp_index,
                        t_mod=temb,
                        sequence_length=hidden_states.size(1),
                        sequence_parallel_metadata=sequence_parallel_metadata,
                        hidden_width=hidden_states.size(-1),
                        hidden_dtype=hidden_states.dtype,
                        hidden_device=hidden_states.device,
                    )
                    if skip_blocks:
                        hidden_states = teacache.update(
                            temb,
                            hidden_states,
                            step=time_stemp_index,
                        )
                    else:
                        hidden_states = execute_block_range(
                            0, len(self.transformer_blocks), hidden_states
                        )
                        teacache.store_truth(
                            step=time_stemp_index,
                            t_mod=temb,
                            input_latent=input_hidden_states,
                            output_latent=hidden_states,
                            sequence_length=hidden_states.size(1),
                        )
                elif cache_dit is not None:
                    if cache_dit.num_transformer_blocks != len(
                        self.transformer_blocks
                    ):
                        raise ValueError(
                            "Cache-DiT controller block count does not match "
                            "the LTX Transformer"
                        )
                    input_hidden_states = hidden_states
                    hidden_states = execute_block_range(
                        0, cache_dit.front_end, hidden_states
                    )
                    front_output_hidden_states = hidden_states
                    reuse_middle = cache_dit.check(
                        step=time_stemp_index,
                        input_hidden_states=input_hidden_states,
                        front_output_hidden_states=front_output_hidden_states,
                        sequence_parallel_metadata=sequence_parallel_metadata,
                    )
                    if reuse_middle:
                        hidden_states = cache_dit.update(
                            front_output_hidden_states,
                            step=time_stemp_index,
                        )
                    else:
                        hidden_states = execute_block_range(
                            cache_dit.front_end,
                            cache_dit.middle_end,
                            hidden_states,
                        )
                        cache_dit.store_truth(
                            step=time_stemp_index,
                            front_output_hidden_states=front_output_hidden_states,
                            middle_output_hidden_states=hidden_states,
                        )
                    if cache_dit.back_start < len(self.transformer_blocks):
                        hidden_states = execute_block_range(
                            cache_dit.back_start,
                            len(self.transformer_blocks),
                            hidden_states,
                        )
                    cache_dit.complete(step=time_stemp_index)
                else:
                    # Keep the cache-off route structurally identical to the
                    # pre-cache P6 loop.  In particular, do not call the
                    # range helper here: Dynamo otherwise traces its dynamic
                    # range boundary differently on SP peers and can leave a
                    # rank in a compiled fragment while its peer has already
                    # entered the first Ulysses collective.
                    for block_index, block in enumerate(self.transformer_blocks):
                        if sequence_parallel_context is None:
                            hidden_states = execute_block(block, hidden_states)
                            continue

                        local_block_error: BaseException | None = None
                        with self._shared_attention_processor.sequence_parallel_block_scope(
                            block_index
                        ) as block_control:
                            try:
                                hidden_states = execute_block(block, hidden_states)
                            except BaseException as error:
                                local_block_error = error

                            if not block_control.before_first_collective_joined:
                                try:
                                    self._shared_attention_processor.synchronize_before_collective(
                                        1,
                                        local_block_error,
                                    )
                                except BaseException as error:
                                    if local_block_error is None:
                                        local_block_error = error

                            if not block_control.before_second_collective_joined:
                                try:
                                    self._shared_attention_processor.synchronize_before_collective(
                                        2,
                                        local_block_error,
                                    )
                                except BaseException as error:
                                    if local_block_error is None:
                                        local_block_error = error

                            synchronize_ltx095_sequence_parallel_phase(
                                local_block_error,
                                binding=control_binding,
                                phase=f"block {block_index} completion",
                            )

            scale_shift_values = (
                self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
            )
            shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

            hidden_states = self.norm_out(hidden_states)
            hidden_states = hidden_states * (1 + scale) + shift
            output = self.proj_out(hidden_states)

            if not video_mode:
                if return_dict:
                    return Transformer2DModelOutput(sample=output)
                return (output,)
            output = self.unsequence_latent(
                output,
                num_frames=int(num_frames or 0),
                height=int(height or 0),
                width=int(width or 0),
            )
            if return_dict:
                return Transformer2DModelOutput(sample=output)
            return (output,)
        finally:
            if lora_scaled:
                _diffusers_transformer_ltx.unscale_lora_layers(self, lora_scale)


EntryClass = LTXVideoTransformer3DModel
