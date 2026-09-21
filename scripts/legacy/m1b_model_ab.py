#!/usr/bin/env python3
"""M1b diagnostic: A/B the model path of the original pipeline vs the new stages.

Both sides run in one process on one GPU, on the *same* window input, the same
weights and the same seed, so any difference in the decoded frames is a genuine
implementation difference rather than run-to-run noise.

    CUDA_VISIBLE_DEVICES=N PYTHONPATH=. python scripts/legacy/m1b_model_ab.py \
        --video data/10268234.mp4 --mask data/10268234_mask.mp4 \
        --prompt "There is a bridge over the lake."

Checkpoints reported: cond_latents (VAE encode + normalise), the initial noisy
latents, and the final decoded frames.  The new-side stage chain is driven
directly through the same ``Req`` contract the runtime uses.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from config.eraserdit import ERASERDIT_NEGATIVE_PROMPT, EraserDiTEraseSamplingParams
from config.server_args import ServerArgs, set_global_server_args
from models.dits.eraserdit_transformer import EraserDiTLTXVideoTransformer3DModel
from models.registry import ModelRegistry
from models.vaes.eraserdit_vae import EraserDiTAutoencoderKLLTXVideo
from nodes.schedule_batch import Req
from nodes.stages.model_specific_stages.eraserdit_erase import (
    EraserDiTEraseConditionEncodingStage,
    EraserDiTEraseDecodingStage,
    EraserDiTEraseDenoisingStage,
    EraserDiTEraseLatentPreparationStage,
    EraserDiTEraseTextEncodingStage,
    EraserDiTEraseTimestepPreparationStage,
)
from nodes.stages.model_specific_stages.eraserdit_erase._common import (
    TASK_STATE_KEY,
    EraserDiTTaskState,
)
from pipelines.eraserdit_video2video import LTXVideoToVideoPipeline
from transformers import T5EncoderModel, T5Tokenizer
from utils.determinism import enable_deterministic_mode
from utils.eraserdit_preprocess import preprocess_eraserdit_window
from utils.video_io import read_mask_array, read_video_array

SNAPSHOT = Path(
    "/root/.cache/huggingface/hub/models--jieeliu--EraserDiT/snapshots/"
    "904fb412da76235085dbbccaefdbde4979fa3d29"
)


def report(label: str, a: torch.Tensor, b: torch.Tensor) -> None:
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    denom = a.abs().mean().clamp_min(1e-12)
    print(
        f"{label:34s} equal={bool(torch.equal(a, b))!s:5s} "
        f"max={diff.max().item():.3e} mean={diff.mean().item():.3e} "
        f"rel={float(diff.mean() / denom):.3e}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/10268234.mp4")
    parser.add_argument("--mask", default="data/10268234_mask.mp4")
    parser.add_argument("--prompt", default="There is a bridge over the lake.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.8)
    args = parser.parse_args()

    print("determinism:", enable_deterministic_mode(), flush=True)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    server_args = ServerArgs(
        model_path=str(SNAPSHOT),
        device="cuda",
        weight_dtype="bf16",
        component_architectures={
            "transformer": "EraserDiTLTXVideoTransformer3DModel",
            "vae": "EraserDiTAutoencoderKLLTXVideo",
            "scheduler": "FlowMatchEulerDiscreteScheduler",
        },
    )
    set_global_server_args(server_args)

    vae = (
        EraserDiTAutoencoderKLLTXVideo.from_pretrained(
            f"{SNAPSHOT}/vae", use_safetensor=True, low_cpu_mem_usage=False,
            local_files_only=True,
        )
        .to(device, dtype)
        .eval()
    )
    transformer = (
        EraserDiTLTXVideoTransformer3DModel.from_pretrained(
            f"{SNAPSHOT}/transformer", use_safetensor=True, low_cpu_mem_usage=False,
            local_files_only=True,
        )
        .to(device, dtype)
        .eval()
    )
    text_encoder = (
        T5EncoderModel.from_pretrained(
            f"{SNAPSHOT}/text_encoder", revision="main", variant=None,
            torch_dtype=dtype, local_files_only=True,
        )
        .to(device)
        .eval()
    )
    tokenizer = T5Tokenizer.from_pretrained(f"{SNAPSHOT}/tokenizer", local_files_only=True)
    scheduler_cls, _ = ModelRegistry.resolve_model_cls("FlowMatchEulerDiscreteScheduler")
    scheduler = scheduler_cls.from_pretrained(f"{SNAPSHOT}/scheduler", local_files_only=True)
    print("models loaded", flush=True)

    # ── identical window input for both paths ────────────────────────────────
    frames, _ = read_video_array(args.video)
    mask_array, _ = read_mask_array(args.mask)
    mask_hwc = (
        torch.from_numpy(mask_array)[..., None].repeat(1, 1, 1, 3).float() / 255.0
    )
    window = preprocess_eraserdit_window(
        torch.from_numpy(frames).float().permute(0, 3, 1, 2).contiguous() / 255.0,
        mask_hwc.permute(0, 3, 1, 2).contiguous(),
        head_batch=True,
    )
    video_4d = window.masked_video                        # [F, C, H, W]
    masks_4d = window.mask_latents                        # [L, 1, H, W]
    frame_count = int(video_4d.shape[0])
    height = int(video_4d.shape[-2])
    width = int(video_4d.shape[-1])
    print(f"window frames={frame_count} height={height} width={width}", flush=True)

    # ── path A: the original pipeline, verbatim ─────────────────────────────
    captured: dict[str, torch.Tensor] = {}
    original_prepare = LTXVideoToVideoPipeline.prepare_latents

    def prepare_spy(self, *a, **kw):
        latents, init = original_prepare(self, *a, **kw)
        captured["cond_latents"] = init.detach().clone()
        captured["noisy_latents"] = latents.detach().clone()
        return latents, init

    LTXVideoToVideoPipeline.prepare_latents = prepare_spy
    try:
        pipe = LTXVideoToVideoPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            transformer=transformer, scheduler=scheduler,
        )
        pipe.to(device, dtype)
        generator_a = torch.Generator(device=device).manual_seed(args.seed)
        # inference_batch wraps the whole pipeline call in autocast(bf16).
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            out_a = pipe(
                video=video_4d,
                masks=masks_4d,
                prompt=args.prompt,
                negative_prompt=ERASERDIT_NEGATIVE_PROMPT,
                num_frames=frame_count,
                height=height,
                width=width,
                num_inference_steps=args.steps,
                generator=generator_a,
                output_type="pt",
                strength=args.strength,
                decode_timestep=0.0,
                decode_noise_scale=0.0,
            ).frames[0]
    finally:
        LTXVideoToVideoPipeline.prepare_latents = original_prepare
    print("path A (original pipeline) done", flush=True)

    # ── path B: the new stage chain, same Req contract as the runtime ───────
    params = EraserDiTEraseSamplingParams(
        prompt=args.prompt,
        negative_prompt=ERASERDIT_NEGATIVE_PROMPT,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=3.0,
        strength=args.strength,
        infer_len=frame_count,
        overlap=9,
        fps=25,
    )
    generator_b = torch.Generator(device=device).manual_seed(args.seed)
    req = Req(sampling_params=params, generator=generator_b)
    req.modules = {
        "vae": vae,
        "transformer": transformer,
        "scheduler": scheduler,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
    }
    req.extra[TASK_STATE_KEY] = EraserDiTTaskState(generator=generator_b)
    req.extra["window_spec"] = {"overlap_left": 0, "overlap_right": 0}
    req.extra["window_index"] = 0
    req.padded_video = video_4d.permute(1, 0, 2, 3).unsqueeze(0)
    req.padded_mask = masks_4d.permute(1, 0, 2, 3).unsqueeze(0)
    req.max_sequence_length = 128

    stages = [
        EraserDiTEraseTextEncodingStage(text_encoder=text_encoder, tokenizer=tokenizer),
        EraserDiTEraseConditionEncodingStage(),
        EraserDiTEraseLatentPreparationStage(scheduler=scheduler),
        EraserDiTEraseTimestepPreparationStage(),
        EraserDiTEraseDenoisingStage(transformer=transformer, scheduler=scheduler),
        EraserDiTEraseDecodingStage(),
    ]
    cond_b = noisy_b = None
    for stage in stages:
        req = stage(req, server_args)
        if isinstance(stage, EraserDiTEraseConditionEncodingStage):
            cond_b = req.cond_latents.detach().clone()
        if isinstance(stage, EraserDiTEraseLatentPreparationStage):
            noisy_b = req.latents.detach().clone()
    out_b = req.decoded_video[0].permute(1, 0, 2, 3).contiguous()
    print("path B (new stages) done", flush=True)

    # ── comparisons ─────────────────────────────────────────────────────────
    print("\n=== checkpoint comparison (A = original pipeline, B = new stages) ===")
    report("cond_latents", captured["cond_latents"], cond_b)
    report("initial latents (scale_noise)", captured["noisy_latents"], noisy_b)
    report("decoded video frames", out_a, out_b)
    diff = (out_a.float() - out_b.float()).abs()
    print(
        f"\ndecoded: max|d|={diff.max().item():.6f} "
        f"frac|d|>1/255={(diff > 1/255).float().mean().item():.4f} "
        f"frac|d|>0={(diff > 0).float().mean().item():.4f}"
    )

    # Dump the decoded window so the write path can be analysed offline (CPU only).
    dump = Path(os.environ.get("M1B_DUMP", "/tmp/m1b_decoded.pt"))
    torch.save(
        {
            "decoded": out_a.detach().cpu(),
            "style_video": video_4d.detach().cpu(),
            "style_mask": masks_4d.detach().cpu(),
            "source_frames": torch.from_numpy(frames),
            "source_mask_raw": torch.from_numpy(mask_array),
        },
        dump,
    )
    print(f"dumped decoded window to {dump}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
