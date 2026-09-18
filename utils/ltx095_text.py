"""Shared LTX095 text embedding helpers."""

from __future__ import annotations

import time
from typing import Any

import torch

from utils.resource_policy import maybe_pin_tensor, module_device, module_dtype


LTX095_MAX_SEQUENCE_LENGTH = 512


def apply_cached_ltx095_text_embeddings(
    batch: Any,
    cached: dict[str, torch.Tensor],
) -> None:
    batch.prompt_embeds = cached["prompt_embeds"]
    batch.prompt_attention_mask = cached["prompt_attention_mask"]
    batch.negative_prompt_embeds = cached["negative_prompt_embeds"]
    batch.negative_attention_mask = cached["negative_attention_mask"]
    batch.max_sequence_length = LTX095_MAX_SEQUENCE_LENGTH


def encode_ltx095_text(
    *,
    tokenizer: Any,
    text_encoder: Any,
    text: str,
    pin_memory: bool,
    timing_sink: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    text_inputs = tokenizer(
        [text],
        padding="max_length",
        max_length=LTX095_MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids
    attention_mask = text_inputs.attention_mask.bool()
    if pin_memory:
        input_ids = maybe_pin_tensor(input_ids, enable=True)
        attention_mask = maybe_pin_tensor(attention_mask, enable=True)
    text_device = module_device(text_encoder)
    attention_mask = attention_mask.to(text_device, non_blocking=pin_memory)
    if text_device.type == "cuda":
        torch.cuda.synchronize(text_device)
    forward_start = time.perf_counter()
    with torch.no_grad():
        prompt_embeds = text_encoder(
            input_ids.to(text_device, non_blocking=pin_memory)
        )[0]
    if text_device.type == "cuda":
        torch.cuda.synchronize(text_device)
    if timing_sink is not None:
        timing_sink["forward_seconds"] = time.perf_counter() - forward_start
    text_dtype = module_dtype(text_encoder)
    prompt_embeds = prompt_embeds.to(device=text_device, dtype=text_dtype)
    return prompt_embeds, attention_mask


def encode_ltx095_text_pair(
    *,
    tokenizer: Any,
    text_encoder: Any,
    prompt: str,
    negative_prompt: str,
    pin_memory: bool,
) -> dict[str, torch.Tensor]:
    prompt_timing: dict[str, float] = {}
    prompt_embeds, prompt_attention_mask = encode_ltx095_text(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        text=prompt,
        pin_memory=pin_memory,
        timing_sink=prompt_timing,
    )
    negative_timing: dict[str, float] = {}
    negative_prompt_embeds, negative_attention_mask = encode_ltx095_text(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        text=negative_prompt,
        pin_memory=pin_memory,
        timing_sink=negative_timing,
    )
    return {
        "prompt_embeds": prompt_embeds,
        "prompt_attention_mask": prompt_attention_mask,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_attention_mask": negative_attention_mask,
        "timing_seconds": {
            "text_prompt_forward_seconds": prompt_timing["forward_seconds"],
            "text_negative_forward_seconds": negative_timing["forward_seconds"],
        },
    }
