"""LTX095 erase – condition encoding stage."""

from __future__ import annotations

from dataclasses import replace

import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from config.server_args import ServerArgs
from models.vaes.ltx095_parallel import (
    LTX095_DECODE_OVERLAP_LATENT_UNITS,
    LTX095_ENCODE_OVERLAP_LATENT_UNITS,
    LTX095VAEParallelAdapter,
)
from nodes.schedule_batch import Req
from nodes.stages.base import PipelineStage
from pipelines.stages.ltx095_erase._common import (
    _distributed_context,
    _field_summary,
    _is_writer_rank,
    _maybe_register_pin_memory,
    _official_parallel_context,
    _official_vae_parallel_active,
    _onload_module,
    _offload_module,
    _record_official_parallel_event,
    record_ltx095_vae_execution_decision,
    record_ltx095_vae_parallel_metrics,
    record_ltx095_sp_writer_stage_operation,
    record_ltx095_sp_writer_stage_skip,
    ltx095_memory_phase_scope,
    resolve_ltx095_vae_binding,
    resolve_ltx095_vae_control_device,
    resolve_ltx095_vae_execution,
    should_skip_ltx095_sp_writer_stage,
)
from memory.policies.memory_phase_controller import MemoryPhase
from parallel.stage_policy import StageExecutionPolicy
from parallel.vae_parallel import (
    VAEOperation,
    VAEParallelBinding,
    VAEParallelEngine,
    vae_task_plan_metadata_size,
)
from utils.mask import encode_mask
from utils.latent import (
    normalize_latents,
    preprocess_video_tensor,
)
from memory.tensor_ops import (
    maybe_pin_tensor,
    module_device,
    module_dtype,
)


def record_and_return_condition_skip(batch: Req) -> Req:
    record_ltx095_sp_writer_stage_skip(
        batch,
        stage="LTX095EraseConditionEncodingStage",
    )
    return batch


def run_serial_condition_encoding(
    batch: Req,
    server_args: ServerArgs,
    *,
    stage: PipelineStage | None = None,
) -> Req:
    record_ltx095_sp_writer_stage_operation(
        batch,
        server_args,
        operation="condition_encode",
    )
    vae, policy = _onload_module(batch, server_args, "vae")
    if _official_vae_parallel_active(server_args):
        official_parallel_context = _official_parallel_context(server_args)
        _record_official_parallel_event(
            batch,
            "vae_parallel_stage_enter",
            stage="LTX095EraseConditionEncodingStage",
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
            stage="LTX095EraseConditionEncodingStage",
            rank=int(getattr(_distributed_context(server_args), "rank", 0) or 0),
        )
    try:
        vae_dtype = module_dtype(vae)
        vae_input = preprocess_video_tensor(batch.masked_video)
        if policy.pin_memory:
            vae_input = maybe_pin_tensor(vae_input, enable=True)
            _maybe_register_pin_memory(batch, "masked_video", "tensor", True)
        vae_input = vae_input.to(
            device=module_device(vae),
            dtype=vae_dtype,
            non_blocking=policy.pin_memory,
        )
        with torch.no_grad():
            posterior = vae.encode(vae_input)
            if hasattr(posterior, "latent_dist"):
                cond_latents = posterior.latent_dist.mode()
            else:
                cond_latents = posterior.latents
        batch.cond_latents = normalize_latents(
            latents=cond_latents.float(),
            latents_mean=vae.latents_mean,
            latents_std=vae.latents_std,
            scaling_factor=float(vae.scaling_factor),
        ).to(device=module_device(vae), dtype=vae_dtype)
        padded_mask = batch.padded_mask
        if policy.pin_memory:
            padded_mask = maybe_pin_tensor(padded_mask, enable=True)
            _maybe_register_pin_memory(batch, "padded_mask", "tensor", True)
        batch.cond_masks = encode_mask(
            padded_mask.to(
                device=batch.cond_latents.device,
                dtype=batch.cond_latents.dtype,
                non_blocking=policy.pin_memory,
            ),
            aim_shape=tuple(batch.cond_latents.shape),
            use_conv_3d=True,
            return_one_channel=batch.mask_one_channel,
        )
        batch.mask_values = batch.cond_masks
        batch.latents = batch.cond_latents.clone()
        batch.latent_shape = tuple(batch.cond_latents.shape)
        if stage is not None:
            stage.log_info(
                "%s | %s",
                _field_summary("cond_latents", batch.cond_latents),
                _field_summary("cond_masks", batch.cond_masks),
            )
        if _official_vae_parallel_active(server_args):
            _record_official_parallel_event(
                batch,
                "stage_complete",
                stage="LTX095EraseConditionEncodingStage",
                rank=int(getattr(_distributed_context(server_args), "rank", 0) or 0),
                latent_shape=tuple(batch.cond_latents.shape),
            )
        return batch
    finally:
        _offload_module(
            batch,
            server_args,
            "vae",
            policy,
            reason="condition_encoding_stage",
        )


def _clear_condition_outputs(batch: Req) -> None:
    batch.cond_latents = None
    batch.cond_masks = None
    batch.mask_values = None
    batch.latents = None
    batch.latent_shape = None


def run_parallel_condition_encoding(
    batch: Req,
    server_args: ServerArgs,
    binding: VAEParallelBinding,
    *,
    stage: PipelineStage | None = None,
) -> Req:
    record_ltx095_sp_writer_stage_operation(
        batch,
        server_args,
        operation="condition_encode",
    )
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
        vae_input = None
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
            if binding.is_owner:
                vae_input = preprocess_video_tensor(batch.masked_video)
                if policy.pin_memory:
                    vae_input = maybe_pin_tensor(vae_input, enable=True)
                    _maybe_register_pin_memory(
                        batch,
                        "masked_video",
                        "tensor",
                        True,
                    )
                vae_input = vae_input.to(
                    device=vae_device,
                    dtype=vae_dtype,
                    non_blocking=policy.pin_memory,
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
                operation=VAEOperation.ENCODE,
                owner_input=vae_input,
                adapter=adapter,
            )
            if execution.uses_parallel_tiles and binding.is_owner:
                assert vae_input is not None
                plan = adapter.build_encode_plan(
                    vae_input,
                    group_ranks=binding.vae_group.spec.ranks,
                    moments_channels=int(vae.latents_std.numel()) * 2,
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

        result = None
        fallback_cond_latents = None
        execute_error: BaseException | None = None
        try:
            with torch.no_grad():
                if execution.uses_parallel_tiles:
                    owner_input = [vae_input] if binding.is_owner else []
                    vae_input = None
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
                        execute_local=adapter.execute_local_encode,
                        merge_round=(
                            merger
                            if merger is not None
                            else lambda current, outputs: current
                        ),
                    )
                elif binding.is_owner:
                    assert vae_input is not None
                    posterior = vae.encode(vae_input)
                    fallback_cond_latents = (
                        posterior.latent_dist.mode()
                        if hasattr(posterior, "latent_dist")
                        else posterior.latents
                    )
        except BaseException as error:
            execute_error = error
        engine.synchronize_phase_error(execute_error, phase="execute")

        publish_error: BaseException | None = None
        try:
            if binding.is_owner:
                if execution.uses_parallel_tiles:
                    if result is None or result.output is None:
                        raise RuntimeError("owner VAE encode did not produce moments")
                    cond_latents = DiagonalGaussianDistribution(result.output).mode()
                else:
                    if fallback_cond_latents is None:
                        raise RuntimeError("owner VAE encode fallback produced no latents")
                    cond_latents = fallback_cond_latents
                batch.cond_latents = normalize_latents(
                    latents=cond_latents.float(),
                    latents_mean=vae.latents_mean,
                    latents_std=vae.latents_std,
                    scaling_factor=float(vae.scaling_factor),
                ).to(device=vae_device, dtype=vae_dtype)
                padded_mask = batch.padded_mask
                if policy.pin_memory:
                    padded_mask = maybe_pin_tensor(padded_mask, enable=True)
                    _maybe_register_pin_memory(
                        batch,
                        "padded_mask",
                        "tensor",
                        True,
                    )
                batch.cond_masks = encode_mask(
                    padded_mask.to(
                        device=batch.cond_latents.device,
                        dtype=batch.cond_latents.dtype,
                        non_blocking=policy.pin_memory,
                    ),
                    aim_shape=tuple(batch.cond_latents.shape),
                    use_conv_3d=True,
                    return_one_channel=batch.mask_one_channel,
                )
                batch.mask_values = batch.cond_masks
                batch.latents = batch.cond_latents.clone()
                batch.latent_shape = tuple(batch.cond_latents.shape)
                if stage is not None:
                    stage.log_info(
                        "%s | %s",
                        _field_summary("cond_latents", batch.cond_latents),
                        _field_summary("cond_masks", batch.cond_masks),
                    )
            else:
                _clear_condition_outputs(batch)
        except BaseException as error:
            publish_error = error
        engine.synchronize_phase_error(publish_error, phase="merge")
        if result is not None:
            record_ltx095_vae_parallel_metrics(
                batch,
                replace(
                    result.metrics,
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
                reason="condition_encoding_stage",
            )


class LTX095EraseConditionEncodingStage(PipelineStage):
    @property
    def execution_policy(self) -> StageExecutionPolicy:
        if resolve_ltx095_vae_binding(self.server_args) is not None:
            return StageExecutionPolicy.group("vae")
        return StageExecutionPolicy.replicated()

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        skipped = should_skip_ltx095_sp_writer_stage(batch, server_args)
        with ltx095_memory_phase_scope(
            batch,
            MemoryPhase.VAE_ENCODE,
            component_name="vae.encoder",
            skipped=skipped,
        ):
            return self._forward_impl(batch, server_args)

    def _forward_impl(self, batch: Req, server_args: ServerArgs) -> Req:
        binding = resolve_ltx095_vae_binding(server_args)
        if binding is not None:
            return run_parallel_condition_encoding(
                batch,
                server_args,
                binding,
                stage=self,
            )
        if should_skip_ltx095_sp_writer_stage(batch, server_args):
            return record_and_return_condition_skip(batch)
        return run_serial_condition_encoding(
            batch,
            server_args,
            stage=self,
        )
