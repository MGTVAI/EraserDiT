"""LTX095 erase - latent preparation stage."""

from __future__ import annotations

import torch

from config.server_args import ServerArgs
from models.dits.ltx095_parallel import LTX095SequenceParallelContract
from nodes.schedule_batch import Req
from nodes.stages.base import PipelineStage
from pipelines.stages.ltx095_erase._common import (
    _field_summary,
    _record_official_parallel_event,
    _retrieve_timesteps,
    _should_skip_writer_only_stage,
    _trim_timesteps_for_strength,
)
from distributed.parallel_state import ParallelContext

_P3_GLOBAL_NOISE_KEY = "ltx095_sequence_parallel_global_noise"


def _active_sequence_parallel_contract(
    server_args: ServerArgs,
) -> LTX095SequenceParallelContract | None:
    contract = getattr(server_args, "ltx095_sequence_parallel_contract", None)
    if contract is None:
        return None
    if not isinstance(contract, LTX095SequenceParallelContract):
        raise TypeError(
            "ltx095_sequence_parallel_contract must be a frozen "
            "LTX095SequenceParallelContract"
        )
    return contract if contract.active else None


def _validate_active_parallel_context(
    server_args: ServerArgs,
    contract: LTX095SequenceParallelContract,
) -> ParallelContext:
    context = getattr(server_args, "parallel_context", None)
    if not isinstance(context, ParallelContext) or not context.enabled:
        raise RuntimeError(
            "active LTX095 parallel requires an enabled ParallelContext"
        )
    plan = context.plan
    frozen_values = (
        ("world_size", plan.world_size, contract.world_size),
        ("sp_degree", plan.sp_degree, contract.sp_degree),
        ("cfg_degree", plan.cfg_degree, contract.cfg_degree),
        ("vae_degree", plan.vae_degree, contract.vae_degree),
        ("writer_rank", plan.writer_rank, contract.writer_rank),
    )
    for name, actual, expected in frozen_values:
        if actual != expected:
            raise RuntimeError(
                f"parallel context {name} changed after LTX095 P3 capability "
                f"freeze: expected {expected}, got {actual}"
            )
    if not 0 <= context.global_rank < contract.world_size:
        raise RuntimeError("parallel context global_rank is outside world_size")
    return context


class LTX095EraseLatentPreparationStage(PipelineStage):
    def __init__(self, scheduler):
        super().__init__()
        self._scheduler = scheduler

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if _should_skip_writer_only_stage(server_args, self.__class__.__name__):
            _record_official_parallel_event(
                batch,
                "stage_skip_non_writer",
                stage=self.__class__.__name__,
                reason="writer_only_stage",
            )
            return batch
        contract = _active_sequence_parallel_contract(server_args)
        parallel_context = (
            _validate_active_parallel_context(server_args, contract)
            if contract is not None
            else None
        )
        is_writer = (
            contract is None or parallel_context.global_rank == contract.writer_rank
        )
        if is_writer:
            timestep_device = batch.cond_latents.device
            latent_batch_size = batch.cond_latents.shape[0]
        else:
            latent_shape = batch.latent_shape
            if type(latent_shape) is not tuple or len(latent_shape) != 5:
                raise TypeError(
                    "batch.latent_shape must be an explicit rank-5 tuple on an "
                    "active LTX095 parallel peer"
                )
            if any(type(value) is not int or value <= 0 for value in latent_shape):
                raise ValueError(
                    "batch.latent_shape dimensions must be positive ints on an "
                    "active LTX095 parallel peer"
                )
            timestep_device = torch.device(server_args.device)
            latent_batch_size = latent_shape[0]
        timesteps = _retrieve_timesteps(
            scheduler=self._scheduler,
            num_inference_steps=batch.num_inference_steps,
            device=timestep_device,
        )
        timesteps, effective_steps = _trim_timesteps_for_strength(
            scheduler=self._scheduler,
            timesteps=timesteps,
            num_inference_steps=batch.num_inference_steps,
            strength=batch.strength,
        )
        latent_timestep = timesteps[:1].repeat(latent_batch_size)
        batch.timesteps = timesteps
        batch.latent_timestep = latent_timestep
        if is_writer:
            generator = (
                batch.generator
                if isinstance(batch.generator, torch.Generator)
                else None
            )
            noise = torch.randn(
                batch.cond_latents.shape,
                device=batch.cond_latents.device,
                dtype=torch.float32,
                generator=generator,
            )
            batch.noisy_latents = self._scheduler.scale_noise(
                sample=batch.cond_latents.to(torch.float32),
                timestep=latent_timestep,
                noise=noise,
            ).to(device=batch.cond_latents.device, dtype=batch.cond_latents.dtype)
            batch.latents = batch.noisy_latents
            if contract is not None:
                batch.extra[_P3_GLOBAL_NOISE_KEY] = noise
        else:
            batch.noisy_latents = None
            batch.latents = None
            batch.extra.pop(_P3_GLOBAL_NOISE_KEY, None)
        batch.extra["effective_inference_steps"] = effective_steps
        self.log_info(
            "%s | timesteps=%s",
            _field_summary("latents", batch.latents),
            tuple(float(x) for x in batch.timesteps.tolist()),
        )
        return batch
