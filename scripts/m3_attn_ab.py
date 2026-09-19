#!/usr/bin/env python3
"""M3 diagnostic: new attention processor vs the reference processor.

The M1b A/B is blind to this rewrite because both of its paths share one
transformer instance.  This isolates the attention path itself: one forward of
the loaded transformer with ``EraserDiTAttentionProcessor``, then the same
forward with every block swapped back to the vendored
``LTXVideoAttentionProcessor2_0``.  Any difference is a processor regression.

    CUDA_VISIBLE_DEVICES=N HF_HUB_OFFLINE=1 PYTHONPATH=. python scripts/m3_attn_ab.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from config.server_args import ServerArgs, set_global_server_args
from models.dits.eraserdit_attention import EraserDiTAttentionProcessor
from models.dits.eraserdit_transformer import (
    LTXVideoAttentionProcessor2_0,
    EraserDiTLTXVideoTransformer3DModel,
)
from utils.determinism import enable_deterministic_mode

SNAPSHOT = Path(
    "/root/.cache/huggingface/hub/models--jieeliu--EraserDiT/snapshots/"
    "904fb412da76235085dbbccaefdbde4979fa3d29"
)
# Real window-0 geometry from a 121-frame 1080x1920 clip (16 latent frames, 60x34).
SHAPE = (1, 128, 16, 60, 34)
TEXT_LEN = 128


def main() -> int:
    print("determinism:", enable_deterministic_mode(), flush=True)
    device = torch.device("cuda")
    set_global_server_args(
        ServerArgs(
            model_path=str(SNAPSHOT),
            device="cuda",
            weight_dtype="bf16",
            component_architectures={
                "transformer": "EraserDiTLTXVideoTransformer3DModel",
                "vae": "EraserDiTAutoencoderKLLTXVideo",
                "scheduler": "FlowMatchEulerDiscreteScheduler",
            },
        )
    )
    torch.manual_seed(0)
    transformer = (
        EraserDiTLTXVideoTransformer3DModel.from_pretrained(
            f"{SNAPSHOT}/transformer",
            use_safetensor=True,
            low_cpu_mem_usage=False,
            local_files_only=True,
        )
        .to(device, torch.bfloat16)
        .eval()
    )

    blocks = transformer.transformer_blocks
    print(
        f"blocks={len(blocks)} norm_q={type(blocks[0].attn1.norm_q).__name__} "
        f"cross_norm_q={type(blocks[0].attn2.norm_q).__name__}",
        flush=True,
    )

    g = torch.Generator(device=device).manual_seed(1)
    latents = torch.randn(SHAPE, generator=g, device=device, dtype=torch.bfloat16)
    cond = torch.randn(SHAPE, generator=g, device=device, dtype=torch.bfloat16)
    mask = torch.rand(
        (SHAPE[0], 1) + SHAPE[2:], generator=g, device=device, dtype=torch.bfloat16
    )
    mask = (mask > 0.5).to(torch.bfloat16)
    embeds = torch.randn(
        (1, TEXT_LEN, 4096), generator=g, device=device, dtype=torch.bfloat16
    )
    ts = torch.tensor([500.0], device=device, dtype=torch.float32)

    def forward() -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return transformer(
                hidden_states=latents,
                encoder_hidden_states=embeds,
                timestep=ts,
                encoder_attention_mask=None,
                num_frames=SHAPE[2],
                height=SHAPE[3],
                width=SHAPE[4],
                rope_interpolation_scale=(0.32, 32.0, 32.0),
                attention_kwargs=None,
                return_dict=False,
                cond_latents=cond,
                mask_values=mask,
            )[0].float()

    new = forward()
    print("new processor forward done", flush=True)

    legacy = LTXVideoAttentionProcessor2_0()
    for block in blocks:
        block.attn1.set_processor(legacy)
        block.attn2.set_processor(legacy)
    ref = forward()
    print("reference processor forward done", flush=True)

    diff = (new - ref).abs()
    print(
        f"\nmax|d|={diff.max().item():.6e} mean|d|={diff.mean().item():.6e} "
        f"ref_mean|v|={ref.abs().mean().item():.6e}"
    )
    scale = ref.abs().mean().clamp_min(1e-12)
    print(f"relative={float(diff.mean() / scale):.6e}")
    print("bit_equal" if torch.equal(new, ref) else "DIFFERS")
    return 0 if torch.equal(new, ref) else 1


if __name__ == "__main__":
    raise SystemExit(main())
