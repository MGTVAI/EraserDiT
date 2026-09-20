"""EraserDiT erase – text encoding stage.

Port of ``LTXVideoToVideoPipeline._get_t5_prompt_embeds`` / ``encode_prompt``.
The positive and negative embeddings are kept separate (the baseline concatenates
them with the negative half first and then slices ``[0:1]`` / ``[1:]``; the
denoising stage performs the same split explicitly).
"""

from __future__ import annotations

import torch

from config.server_args import ServerArgs
from memory.policies.component_offload import offload_component
from nodes.schedule_batch import Req
from nodes.stages.base import PipelineStage
from nodes.stages.model_specific_stages.eraserdit_erase._common import field_summary
from utils.logging_utils import init_logger
from utils.resource_policy import module_device

logger = init_logger(__name__)


class EraserDiTEraseTextEncodingStage(PipelineStage):
    def __init__(self, text_encoder, tokenizer):
        super().__init__()
        self._text_encoder = text_encoder
        self._tokenizer = tokenizer

    def _t5_prompt_embeds(
        self,
        prompt: str,
        *,
        max_sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer = self._tokenizer
        text_inputs = tokenizer(
            [prompt],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.bool().to(device)

        untruncated_ids = tokenizer([prompt], padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(
                untruncated_ids[:, max_sequence_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` "
                "is set to %s tokens: %s",
                max_sequence_length,
                removed_text,
            )

        prompt_embeds = self._text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds, prompt_attention_mask.view(1, -1)

    @offload_component("text_encoder")
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        text_encoder = self._text_encoder
        device = module_device(text_encoder)
        dtype = text_encoder.dtype
        max_sequence_length = int(batch.max_sequence_length or 128)

        prompt = batch.prompt or ""
        negative_prompt = batch.negative_prompt or ""

        # The baseline calls ``encode_prompt`` inside ``autocast(bf16)``; the T5
        # layer norms are in autocast's fp32 category, so this changes their
        # outputs and must be reproduced.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            prompt_embeds, prompt_attention_mask = self._t5_prompt_embeds(
                prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
            negative_prompt_embeds, negative_prompt_attention_mask = self._t5_prompt_embeds(
                negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        batch.prompt_embeds = prompt_embeds
        batch.prompt_attention_mask = prompt_attention_mask
        batch.negative_prompt_embeds = negative_prompt_embeds
        batch.negative_attention_mask = negative_prompt_attention_mask
        batch.max_sequence_length = max_sequence_length
        self.log_info(
            "%s | %s",
            field_summary("prompt_embeds", prompt_embeds),
            field_summary("negative_prompt_embeds", negative_prompt_embeds),
        )
        return batch
