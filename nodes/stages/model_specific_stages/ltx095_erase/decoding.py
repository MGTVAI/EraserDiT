"""LTX095 erase - decoding stage."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import torch

from config.server_args import ServerArgs
from models.vaes.ltx095_parallel import (
    LTX095_DECODE_OVERLAP_LATENT_UNITS,
    LTX095_ENCODE_OVERLAP_LATENT_UNITS,
    LTX095VAEParallelAdapter,
)
from nodes.schedule_batch import Req
from nodes.stages.base import PipelineStage
from nodes.stages.model_specific_stages.ltx095_erase._common import (
    _distributed_context,
    _field_summary,
    _is_writer_rank,
    _official_parallel_context,
    _official_vae_parallel_active,
    _onload_module,
    _offload_module,
    _record_official_parallel_event,
    record_ltx095_vae_execution_decision,
    record_ltx095_vae_parallel_metrics,
    ltx095_memory_phase_scope,
    resolve_ltx095_vae_binding,
    resolve_ltx095_vae_control_device,
    resolve_ltx095_vae_execution,
)
from memory.policies.memory_phase_controller import MemoryPhase
from videoerase.windowing.commit_sync import (
    resolve_active_ltx095_window_commit_context,
)
from parallel.stage_policy import StageExecutionPolicy
from parallel.vae_parallel import (
    VAEOperation,
    VAEParallelBinding,
    VAEParallelEngine,
    vae_task_plan_metadata_size,
)
from utils.latent import (
    denormalize_latents,
    postprocess_video_tensor,
)
from utils.resource_policy import (
    module_device,
    module_dtype,
)

_PARALLEL_POSTPROCESS_CHUNK_BYTES = 64 * 1024 * 1024


def _postprocess_parallel_output_bounded(
    video: torch.Tensor,
    *,
    max_chunk_bytes: int = _PARALLEL_POSTPROCESS_CHUNK_BYTES,
) -> torch.Tensor:
    """Postprocess a merged VAE output without two additional full-video buffers."""
    if type(max_chunk_bytes) is not int or max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be a positive plain int")
    if not video.is_contiguous():
        raise ValueError("parallel VAE output must be contiguous")
    flat = video.view(-1)
    elements_per_chunk = max(1, max_chunk_bytes // torch.float32.itemsize)
    for start in range(0, flat.numel(), elements_per_chunk):
        chunk = flat.narrow(
            0,
            start,
            min(elements_per_chunk, flat.numel() - start),
        )
        converted = postprocess_video_tensor(chunk).to(
            device=video.device,
            dtype=video.dtype,
        )
        chunk.copy_(converted)
    return video


def run_serial_decoding(
    batch: Req,
    server_args: ServerArgs,
    *,
    stage: PipelineStage | None = None,
) -> Req:
    active = resolve_active_ltx095_window_commit_context(server_args)
    vae, policy = _onload_module(batch, server_args, "vae")
    if _official_vae_parallel_active(server_args):
        official_parallel_context = _official_parallel_context(server_args)
        _record_official_parallel_event(
            batch,
            "vae_parallel_stage_enter",
            stage="LTX095EraseDecodingStage",
            rank=int(getattr(_distributed_context(server_args), "rank", 0) or 0),
            writer_rank=bool(_is_writer_rank(server_args)),
            vae_parallel_enabled=bool(
                getattr(official_parallel_context, "vae_parallel_enabled", False)
            ),
            vae_parallel_degree=int(
                getattr(official_parallel_context, "vae_parallel_degree", 1) or 1
            ),
        )
        _record_official_parallel_event(
            batch,
            "stage_enter",
            stage="LTX095EraseDecodingStage",
            rank=int(getattr(_distributed_context(server_args), "rank", 0) or 0),
        )
    try:
        vae_dtype = module_dtype(vae)
        latents = denormalize_latents(
            latents=batch.latents.float(),
            latents_mean=vae.latents_mean,
            latents_std=vae.latents_std,
            scaling_factor=float(vae.scaling_factor),
        ).to(device=module_device(vae), dtype=vae_dtype)
        decode_temb = torch.tensor([0.0], device=module_device(vae), dtype=vae_dtype)
        with torch.no_grad():
            if batch.metrics is not None:
                batch.metrics.record_operation("vae_decode")
            decoded = vae.decode(latents, temb=decode_temb)
        decoded_video = decoded.sample if hasattr(decoded, "sample") else decoded
        batch.decoded_video = postprocess_video_tensor(decoded_video).to(
            device=module_device(vae), dtype=vae_dtype
        )
        if active is not None:
            batch.latents = None
        if _official_vae_parallel_active(server_args):
            _record_official_parallel_event(
                batch,
                "stage_complete",
                stage="LTX095EraseDecodingStage",
                rank=int(getattr(_distributed_context(server_args), "rank", 0) or 0),
                decoded_shape=tuple(batch.decoded_video.shape),
            )
        if stage is not None:
            stage.log_info(
                "%s",
                _field_summary("decoded_video", batch.decoded_video),
            )
        return batch
    finally:
        _offload_module(
            batch,
            server_args,
            "vae",
            policy,
            reason="decoding_stage",
        )


def _clear_decode_outputs(batch: Req) -> None:
    batch.latents = None
    batch.decoded_video = None


def run_parallel_decoding(
    batch: Req,
    server_args: ServerArgs,
    binding: VAEParallelBinding,
    *,
    stage: PipelineStage | None = None,
) -> Req:
    engine = VAEParallelEngine(
        binding,
        control_device=resolve_ltx095_vae_control_device(server_args, binding),
        max_inflight_tiles=getattr(server_args, "vae_max_inflight_tiles", 1),
    )
    vae = None
    policy = None
    try:
        onload_error: BaseException | None = None
        try:
            vae, policy = _onload_module(batch, server_args, "vae")
        except BaseException as error:
            onload_error = error
        engine.synchronize_phase_error(onload_error, phase="plan")
        assert vae is not None and policy is not None

        adapter = None
        latents = None
        plan = None
        execution = None
        prepare_error: BaseException | None = None
        try:
            vae_dtype = module_dtype(vae)
            vae_device = module_device(vae)
            spatial_ratio = int(vae.spatial_compression_ratio)
            adapter = LTX095VAEParallelAdapter(
                vae,
                encode_sample_overlap=(
                    LTX095_ENCODE_OVERLAP_LATENT_UNITS * spatial_ratio
                ),
                decode_latent_overlap=LTX095_DECODE_OVERLAP_LATENT_UNITS,
            )
            decode_temb = torch.tensor(
                [0.0],
                device=vae_device,
                dtype=vae_dtype,
            )
            if binding.is_owner:
                latents = denormalize_latents(
                    latents=batch.latents.float(),
                    latents_mean=vae.latents_mean,
                    latents_std=vae.latents_std,
                    scaling_factor=float(vae.scaling_factor),
                ).to(device=vae_device, dtype=vae_dtype)
                output_channels = int(
                    getattr(getattr(vae, "config", None), "out_channels", 3)
                )
        except BaseException as error:
            prepare_error = error
        engine.synchronize_phase_error(prepare_error, phase="plan")
        assert adapter is not None
        decision_error: BaseException | None = None
        try:
            execution = resolve_ltx095_vae_execution(
                batch=batch,
                server_args=server_args,
                operation=VAEOperation.DECODE,
                owner_input=latents,
                adapter=adapter,
            )
            if execution.uses_parallel_tiles and binding.is_owner:
                assert latents is not None
                output_channels = int(
                    getattr(getattr(vae, "config", None), "out_channels", 3)
                )
                plan = adapter.build_decode_plan(
                    latents,
                    group_ranks=binding.vae_group.spec.ranks,
                    output_channels=output_channels,
                )
        except BaseException as error:
            decision_error = error
        engine.synchronize_phase_error(decision_error, phase="plan")
        assert execution is not None
        record_ltx095_vae_execution_decision(batch, execution.decision)
        merger = (
            adapter.create_incremental_merger(plan)
            if execution.uses_parallel_tiles and binding.is_owner and plan is not None
            else None
        )

        if batch.metrics is not None:
            batch.metrics.record_operation("vae_decode")
        result = None
        fallback_decoded = None
        execute_error: BaseException | None = None
        try:
            with torch.no_grad():
                if execution.uses_parallel_tiles:
                    owner_input = [latents] if binding.is_owner else []
                    latents = None
                    result = engine.execute(
                        plan,
                        plan_metadata_size=vae_task_plan_metadata_size(
                            5,
                            5,
                            len(binding.vae_group.spec.ranks),
                        ),
                        full_input=owner_input.pop() if owner_input else None,
                        input_dtype=vae_dtype,
                        output_dtype=vae_dtype,
                        device=vae_device,
                        materialize_input=adapter.materialize_input,
                        execute_local=partial(
                            adapter.execute_local_decode,
                            temb=decode_temb,
                        ),
                        merge_round=(
                            merger
                            if merger is not None
                            else lambda current, outputs: current
                        ),
                    )
                elif binding.is_owner:
                    assert latents is not None
                    decoded = vae.decode(latents, temb=decode_temb)
                    fallback_decoded = (
                        decoded.sample if hasattr(decoded, "sample") else decoded
                    )
        except BaseException as error:
            execute_error = error
        engine.synchronize_phase_error(execute_error, phase="execute")

        publish_error: BaseException | None = None
        published_video = False
        try:
            if binding.is_owner:
                output = (
                    result.output
                    if execution.uses_parallel_tiles and result is not None
                    else fallback_decoded
                )
                if output is None:
                    raise RuntimeError("owner VAE decode did not produce a video")
                if execution.uses_parallel_tiles:
                    batch.decoded_video = _postprocess_parallel_output_bounded(output)
                else:
                    batch.decoded_video = postprocess_video_tensor(output).to(
                        device=vae_device,
                        dtype=vae_dtype,
                    )
                published_video = True
                if resolve_active_ltx095_window_commit_context(server_args) is not None:
                    batch.latents = None
                if stage is not None:
                    stage.log_info(
                        "%s",
                        _field_summary("decoded_video", batch.decoded_video),
                    )
            else:
                _clear_decode_outputs(batch)
        except BaseException as error:
            publish_error = error
        engine.synchronize_phase_error(publish_error, phase="merge")
        if result is not None:
            record_ltx095_vae_parallel_metrics(
                batch,
                replace(
                    result.metrics,
                    published_video=published_video,
                    resolved_degree=execution.decision.resolved_degree,
                    effective_degree=execution.decision.effective_degree,
                    fallback_reason=execution.decision.fallback_reason,
                ),
            )
        return batch
    finally:
        if policy is not None:
            _offload_module(
                batch,
                server_args,
                "vae",
                policy,
                reason="decoding_stage",
            )


class LTX095EraseDecodingStage(PipelineStage):
    @property
    def execution_policy(self) -> StageExecutionPolicy:
        if resolve_ltx095_vae_binding(self.server_args) is not None:
            return StageExecutionPolicy.group("vae")
        return StageExecutionPolicy.replicated()

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        active = resolve_active_ltx095_window_commit_context(server_args)
        skipped = active is not None and not active.is_writer
        with ltx095_memory_phase_scope(
            batch,
            MemoryPhase.VAE_DECODE,
            component_name="vae.decoder",
            skipped=skipped,
        ):
            return self._forward_impl(batch, server_args)

    def _forward_impl(self, batch: Req, server_args: ServerArgs) -> Req:
        binding = resolve_ltx095_vae_binding(server_args)
        if binding is not None:
            return run_parallel_decoding(
                batch,
                server_args,
                binding,
                stage=self,
            )
        active = resolve_active_ltx095_window_commit_context(server_args)
        if active is not None and not active.is_writer:
            _clear_decode_outputs(batch)
            return batch
        return run_serial_decoding(
            batch,
            server_args,
            stage=self,
        )
