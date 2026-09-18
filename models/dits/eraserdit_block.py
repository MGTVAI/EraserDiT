"""EraserDiT transformer block with the two AdaLN sites routed through the
operator-fusion dispatcher.

The block body is a verbatim copy of
``models/dits/eraserdit_transformer.py::LTXVideoTransformerBlock.forward`` with
``norm * (1 + scale) + shift`` replaced by ``apply_fused_rmsnorm_adaln``.  When
fusion is disabled the dispatcher evaluates exactly the original expression, so
the two paths are numerically identical by construction.

The ``rmsnorm_adaln`` capability contract (hidden width 2048, eps 1e-6, bf16,
batch 1, non-affine norm, modulation of shape ``(1, 1, 2048)``) is met here: the
denoising stage issues two batch-1 forwards for classifier-free guidance and
``temb`` carries a single timestep, so ``scale_msa``/``shift_msa`` broadcast to
``(1, 1, 2048)``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from layers.operator_fusion.registry import OperatorFusionDecision
from layers.operator_fusion.rmsnorm_adaln import apply_fused_rmsnorm_adaln

__all__ = ["forward_eraserdit_block"]


def forward_eraserdit_block(
    block,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    *,
    decision: OperatorFusionDecision,
) -> torch.Tensor:
    batch_size = hidden_states.size(0)
    norm_hidden_states = block.norm1(hidden_states)

    num_ada_params = block.scale_shift_table.shape[0]
    ada_values = block.scale_shift_table[None, None] + temb.reshape(
        batch_size, temb.size(1), num_ada_params, -1
    )
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(
        dim=2
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
