"""LTX095 erase – text encoding stage."""

from __future__ import annotations

import torch

from config.server_args import ServerArgs
from nodes.schedule_batch import Req
from nodes.stages.base import PipelineStage
from nodes.stages.model_specific_stages.ltx095_erase._common import (
    _field_summary,
    _maybe_register_pin_memory,
    _onload_module,
    _offload_module,
    _record_official_parallel_event,
    _should_skip_writer_only_stage,
    ltx095_memory_phase_scope,
    record_ltx095_sp_writer_stage_operation,
    record_ltx095_sp_writer_stage_skip,
    should_skip_ltx095_sp_writer_stage,
)
from memory.policies.memory_phase_controller import MemoryPhase
from utils.ltx095_text import (
    apply_cached_ltx095_text_embeddings,
    encode_ltx095_text_pair,
    LTX095_MAX_SEQUENCE_LENGTH,
)


class LTX095EraseTextEncodingStage(PipelineStage):
    def __init__(self, text_encoder, tokenizer):
        super().__init__()
        self._text_encoder = text_encoder
        self._tokenizer = tokenizer

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        skipped = (
            should_skip_ltx095_sp_writer_stage(batch, server_args)
            or _should_skip_writer_only_stage(
                server_args,
                self.__class__.__name__,
            )
            or isinstance(batch.extra.get("cached_text_embeddings"), dict)
        )
        with ltx095_memory_phase_scope(
            batch,
            MemoryPhase.TEXT_ENCODE,
            component_name="text_encoder",
            skipped=skipped,
        ):
            return self._forward_impl(batch, server_args)

    def _forward_impl(self, batch: Req, server_args: ServerArgs) -> Req:
        if should_skip_ltx095_sp_writer_stage(batch, server_args):
            record_ltx095_sp_writer_stage_skip(
                batch,
                stage=self.__class__.__name__,
            )
            return batch
        record_ltx095_sp_writer_stage_operation(
            batch,
            server_args,
            operation="text_encode",
        )
        if _should_skip_writer_only_stage(server_args, self.__class__.__name__):
            batch.prompt_embeds = None
            batch.prompt_attention_mask = None
            batch.negative_prompt_embeds = None
            batch.negative_attention_mask = None
            _record_official_parallel_event(
                batch,
                "stage_skip_non_writer",
                stage=self.__class__.__name__,
                reason="writer_only_stage",
            )
            return batch
        cached = batch.extra.get("cached_text_embeddings")
        if isinstance(cached, dict):
            apply_cached_ltx095_text_embeddings(batch, cached)
            self.log_info(
                "%s | %s | cache_hit=True | cache_device=cpu",
                _field_summary("prompt_embeds", batch.prompt_embeds),
                _field_summary("negative_prompt_embeds", batch.negative_prompt_embeds),
            )
            return batch

        text_encoder, policy = _onload_module(batch, server_args, "text_encoder")
        try:
            prompt = batch.prompt or ""
            negative_prompt = batch.negative_prompt or ""
            if policy.pin_memory:
                _maybe_register_pin_memory(batch, "text_inputs", "tensor", True)
            embeddings = encode_ltx095_text_pair(
                tokenizer=self._tokenizer,
                text_encoder=text_encoder,
                prompt=prompt,
                negative_prompt=negative_prompt,
                pin_memory=policy.pin_memory,
            )
            batch.prompt_embeds = embeddings["prompt_embeds"]
            batch.prompt_attention_mask = embeddings["prompt_attention_mask"]
            batch.negative_prompt_embeds = embeddings["negative_prompt_embeds"]
            batch.negative_attention_mask = embeddings["negative_attention_mask"]
            batch.max_sequence_length = LTX095_MAX_SEQUENCE_LENGTH
            context = batch.extra.get("runtime_context")
            if context is not None:
                from videoerase.windowing.materializer import (
                    cache_ltx095_window_text_embeddings,
                )
                object_index = int(batch.extra.get("object_index", 0))
                scene_index = int(batch.extra.get("scene_index", 0))
                cache_key = (
                    object_index,
                    scene_index,
                    prompt,
                    negative_prompt,
                )
                if cache_key not in context.text_embedding_cache:
                    def _noop_record(*a: object, **kw: object) -> dict[str, object]:
                        return {}
                    cache_ltx095_window_text_embeddings(
                        context=context,
                        object_index=object_index,
                        scene_index=scene_index,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        prompt_embeds=embeddings["prompt_embeds"],
                        prompt_attention_mask=embeddings["prompt_attention_mask"],
                        negative_prompt_embeds=embeddings["negative_prompt_embeds"],
                        negative_attention_mask=embeddings["negative_attention_mask"],
                        record_runtime_event_fn=_noop_record,
                    )

            self.log_info(
                "%s | %s | cache_hit=False",
                _field_summary("prompt_embeds", batch.prompt_embeds),
                _field_summary("negative_prompt_embeds", batch.negative_prompt_embeds),
            )
            return batch
        finally:
            _offload_module(
                batch,
                server_args,
                "text_encoder",
                policy,
                reason="text_encoding_stage",
            )
