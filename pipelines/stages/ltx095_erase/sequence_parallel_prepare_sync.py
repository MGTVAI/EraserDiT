"""Synchronize minimal writer-owned window metadata for LTX095 SP peers."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from math import isfinite, prod

import torch

from config.server_args import ServerArgs
from nodes.schedule_batch import Req
from nodes.stages.base import PipelineStage
from distributed.group_coordinator import GroupCoordinator
from distributed.parallel_state import ParallelContext
from pipelines.runtime.windowing.sp_dispatch import (
    LTX095ActiveSPWindowContext,
    resolve_active_ltx095_window_commit_context,
)
from parallel.stage_policy import synchronize_stage_error
from utils.latent import build_rope_interpolation_scale

_PROTOCOL_VERSION = 1
_STATUS_OK = 0
_STATUS_ERROR = 1
_SCALAR_HEADER_SIZE = 24
_TENSOR_HEADER_SIZE = 10
_MAX_TENSOR_NDIM = 4
_MAX_TENSOR_NUMEL = 2**31 - 1
_PROMPT_FIELDS = (
    "prompt_embeds",
    "prompt_attention_mask",
    "negative_prompt_embeds",
    "negative_attention_mask",
)

_DTYPE_TO_CODE = {
    dtype: code
    for name, code in (
        ("bool", 1),
        ("uint8", 2),
        ("int32", 3),
        ("int64", 4),
        ("float16", 5),
        ("bfloat16", 6),
        ("float32", 7),
        ("float64", 8),
    )
    if (dtype := getattr(torch, name, None)) is not None
}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}


@dataclass(frozen=True)
class _PrepareMetadata:
    latent_shape: tuple[int, int, int, int, int]
    mask_shape: tuple[int, int, int, int, int]
    max_sequence_length: int
    dynamic_cfg: bool
    cfg_step: int
    enable_dynamic_cfg_space: bool
    guidance_scale: float
    rope_interpolation_scale: tuple[float, float, float]
    fps: float
    num_inference_steps: int
    strength: float


@dataclass(frozen=True)
class _TensorMetadata:
    shape: tuple[int, ...]
    dtype: torch.dtype
    numel: int


def _float_to_int64(value: float) -> int:
    return struct.unpack("!q", struct.pack("!d", float(value)))[0]


def _int64_to_float(value: int) -> float:
    return struct.unpack("!d", struct.pack("!q", int(value)))[0]


def _validate_rank_five_shape(name: str, shape) -> tuple[int, int, int, int, int]:
    if not isinstance(shape, (tuple, list, torch.Size)) or len(shape) != 5:
        raise TypeError(f"{name} must be an explicit rank-5 shape")
    normalized = tuple(shape)
    if any(type(value) is not int or value <= 0 for value in normalized):
        raise ValueError(f"{name} must contain positive non-bool ints")
    return normalized


def _build_writer_metadata(batch: Req) -> _PrepareMetadata:
    latent_shape = _validate_rank_five_shape("latent_shape", batch.latent_shape)
    mask_values = batch.mask_values
    if not isinstance(mask_values, torch.Tensor):
        raise TypeError("writer mask_values must be a tensor")
    mask_shape = _validate_rank_five_shape("mask_values.shape", mask_values.shape)
    if mask_shape[0] != latent_shape[0] or mask_shape[2:] != latent_shape[2:]:
        raise ValueError("writer mask_values shape must match latent batch and grid")

    rope = batch.rope_interpolation_scale
    if rope is None:
        vae = batch.modules.get("vae")
        if vae is None:
            raise ValueError("writer VAE is required to build rope interpolation")
        rope = build_rope_interpolation_scale(
            temporal_ratio=int(vae.temporal_compression_ratio),
            frame_rate=float(batch.fps),
            spatial_ratio=int(vae.spatial_compression_ratio),
        )
        batch.rope_interpolation_scale = rope
    rope = tuple(float(value) for value in rope)
    if len(rope) != 3:
        raise ValueError("rope_interpolation_scale must contain three values")

    max_sequence_length = batch.max_sequence_length
    if type(max_sequence_length) is not int or max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be a positive non-bool int")
    num_inference_steps = batch.num_inference_steps
    if type(num_inference_steps) is not int or num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be a positive non-bool int")

    cfg_step = int(batch.cfg_step or 0)
    guidance_scale = float(batch.guidance_scale)
    fps = float(batch.fps)
    strength = float(batch.strength)
    if cfg_step < 0:
        raise ValueError("cfg_step must be non-negative")
    if not isfinite(guidance_scale):
        raise ValueError("guidance_scale must be finite")
    if not isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be finite and between zero and one")

    metadata = _PrepareMetadata(
        latent_shape=latent_shape,
        mask_shape=mask_shape,
        max_sequence_length=max_sequence_length,
        dynamic_cfg=bool(batch.dynamic_cfg),
        cfg_step=cfg_step,
        enable_dynamic_cfg_space=bool(getattr(batch, "enable_dynamic_cfg_space", True)),
        guidance_scale=guidance_scale,
        rope_interpolation_scale=rope,
        fps=fps,
        num_inference_steps=num_inference_steps,
        strength=strength,
    )
    batch.extra["ltx095_sp_mask_shape"] = mask_shape
    return metadata


def _encode_scalar_header(
    metadata: _PrepareMetadata,
    *,
    device: torch.device,
) -> torch.Tensor:
    header = torch.tensor(
        [
            _PROTOCOL_VERSION,
            _STATUS_OK,
            *metadata.latent_shape,
            *metadata.mask_shape,
            metadata.max_sequence_length,
            int(metadata.dynamic_cfg),
            metadata.cfg_step,
            int(metadata.enable_dynamic_cfg_space),
            _float_to_int64(metadata.guidance_scale),
            *(_float_to_int64(value) for value in metadata.rope_interpolation_scale),
            _float_to_int64(metadata.fps),
            metadata.num_inference_steps,
            _float_to_int64(metadata.strength),
            0,
        ],
        dtype=torch.int64,
        device=device,
    )
    if header.numel() != _SCALAR_HEADER_SIZE:
        raise AssertionError("LTX095 SP prepare scalar header width changed")
    return header


def _error_header(size: int, *, device: torch.device) -> torch.Tensor:
    header = torch.zeros(size, dtype=torch.int64, device=device)
    header[0] = _PROTOCOL_VERSION
    header[1] = _STATUS_ERROR
    return header


def _decode_scalar_header(header: torch.Tensor) -> _PrepareMetadata:
    if (
        header.dtype is not torch.int64
        or header.ndim != 1
        or header.numel() != _SCALAR_HEADER_SIZE
    ):
        raise TypeError("LTX095 SP prepare scalar header is invalid")
    values = header.detach().to(device="cpu").tolist()
    if values[0] != _PROTOCOL_VERSION:
        raise ValueError("unsupported LTX095 SP prepare protocol version")
    if values[1] == _STATUS_ERROR:
        raise RuntimeError("writer failed to prepare LTX095 SP window metadata")
    if values[1] != _STATUS_OK or values[23] != 0:
        raise ValueError("LTX095 SP prepare scalar status or reserved field is invalid")
    if values[13] not in (0, 1) or values[15] not in (0, 1):
        raise ValueError("LTX095 SP prepare boolean metadata is invalid")
    metadata = _PrepareMetadata(
        latent_shape=_validate_rank_five_shape("latent_shape", values[2:7]),
        mask_shape=_validate_rank_five_shape("mask_shape", values[7:12]),
        max_sequence_length=int(values[12]),
        dynamic_cfg=bool(values[13]),
        cfg_step=int(values[14]),
        enable_dynamic_cfg_space=bool(values[15]),
        guidance_scale=_int64_to_float(values[16]),
        rope_interpolation_scale=tuple(
            _int64_to_float(value) for value in values[17:20]
        ),
        fps=_int64_to_float(values[20]),
        num_inference_steps=int(values[21]),
        strength=_int64_to_float(values[22]),
    )
    if metadata.max_sequence_length <= 0:
        raise ValueError("synchronized max_sequence_length must be positive")
    if metadata.cfg_step < 0:
        raise ValueError("synchronized cfg_step must be non-negative")
    if metadata.num_inference_steps <= 0:
        raise ValueError("synchronized num_inference_steps must be positive")
    if not isfinite(metadata.guidance_scale):
        raise ValueError("synchronized guidance_scale must be finite")
    if not isfinite(metadata.fps) or metadata.fps <= 0:
        raise ValueError("synchronized fps must be finite and positive")
    if not isfinite(metadata.strength) or not 0.0 <= metadata.strength <= 1.0:
        raise ValueError("synchronized strength must be between zero and one")
    if (
        metadata.mask_shape[0] != metadata.latent_shape[0]
        or metadata.mask_shape[2:] != metadata.latent_shape[2:]
    ):
        raise ValueError("synchronized mask shape must match latent batch and grid")
    return metadata


def _validate_tensor_metadata(
    field_name: str,
    tensor: torch.Tensor,
) -> _TensorMetadata:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{field_name} must be a tensor")
    if not 1 <= tensor.ndim <= _MAX_TENSOR_NDIM:
        raise ValueError(f"{field_name} has an unsupported rank")
    shape = tuple(int(value) for value in tensor.shape)
    if any(value <= 0 for value in shape):
        raise ValueError(f"{field_name} shape values must be positive")
    numel = prod(shape)
    if tensor.numel() != numel or not 0 < numel <= _MAX_TENSOR_NUMEL:
        raise ValueError(f"{field_name} numel exceeds the bounded shape")
    if tensor.dtype not in _DTYPE_TO_CODE:
        raise TypeError(f"{field_name} has unsupported dtype {tensor.dtype}")
    return _TensorMetadata(shape=shape, dtype=tensor.dtype, numel=numel)


def _encode_tensor_header(
    metadata: _TensorMetadata,
    *,
    device: torch.device,
) -> torch.Tensor:
    padded_shape = metadata.shape + (0,) * (_MAX_TENSOR_NDIM - len(metadata.shape))
    return torch.tensor(
        [
            _PROTOCOL_VERSION,
            _STATUS_OK,
            _DTYPE_TO_CODE[metadata.dtype],
            len(metadata.shape),
            metadata.numel,
            *padded_shape,
            0,
        ],
        dtype=torch.int64,
        device=device,
    )


def _decode_tensor_header(header: torch.Tensor) -> _TensorMetadata:
    if (
        header.dtype is not torch.int64
        or header.ndim != 1
        or header.numel() != _TENSOR_HEADER_SIZE
    ):
        raise TypeError("LTX095 SP prepare tensor header is invalid")
    values = header.detach().to(device="cpu").tolist()
    if values[0] != _PROTOCOL_VERSION:
        raise ValueError("unsupported LTX095 SP prepare tensor protocol version")
    if values[1] == _STATUS_ERROR:
        raise RuntimeError("writer failed to prepare an LTX095 SP prompt tensor")
    ndim = int(values[3])
    if values[1] != _STATUS_OK or not 1 <= ndim <= _MAX_TENSOR_NDIM:
        raise ValueError("LTX095 SP prepare tensor status or rank is invalid")
    if values[9] != 0 or any(values[5 + ndim : 9]):
        raise ValueError("LTX095 SP prepare tensor reserved fields are invalid")
    dtype = _CODE_TO_DTYPE.get(values[2])
    if dtype is None:
        raise ValueError("LTX095 SP prepare tensor dtype code is invalid")
    shape = tuple(int(value) for value in values[5 : 5 + ndim])
    metadata = _TensorMetadata(shape=shape, dtype=dtype, numel=int(values[4]))
    if (
        any(value <= 0 for value in shape)
        or prod(shape) != metadata.numel
        or not 0 < metadata.numel <= _MAX_TENSOR_NUMEL
    ):
        raise ValueError("LTX095 SP prepare tensor shape and numel are invalid")
    return metadata


def _normalize_error(error: BaseException | None) -> Exception | None:
    if error is None or isinstance(error, Exception):
        return error
    return RuntimeError(
        "LTX095 SP prepare raised non-Exception " f"{type(error).__name__}: {error}"
    )


def _synchronize_error(
    error: BaseException | None,
    server_args: ServerArgs,
) -> None:
    normalized = _normalize_error(error)
    parallel_context = getattr(server_args, "parallel_context", None)
    synchronize_stage_error(
        normalized,
        parallel_context if isinstance(parallel_context, ParallelContext) else None,
    )


def _broadcast_prompt_tensor(
    *,
    batch: Req,
    field_name: str,
    active: LTX095ActiveSPWindowContext,
    coordinator: GroupCoordinator,
    server_args: ServerArgs,
) -> torch.Tensor:
    payload = None
    metadata = None
    preparation_error: BaseException | None = None
    if active.is_writer:
        try:
            source = getattr(batch, field_name)
            metadata = _validate_tensor_metadata(field_name, source)
            payload = source.detach().to(device=active.control_device).contiguous()
            header = _encode_tensor_header(
                metadata,
                device=active.control_device,
            )
        except BaseException as error:
            preparation_error = error
            header = _error_header(
                _TENSOR_HEADER_SIZE,
                device=active.control_device,
            )
    else:
        header = torch.zeros(
            _TENSOR_HEADER_SIZE,
            dtype=torch.int64,
            device=active.control_device,
        )
    coordinator.broadcast(header, src=active.contract.writer_rank)
    decoded = None
    local_error = preparation_error
    if local_error is None:
        try:
            decoded = _decode_tensor_header(header)
            if metadata is not None and metadata != decoded:
                raise ValueError(
                    f"writer {field_name} changed after header construction"
                )
            if payload is None:
                payload = torch.empty(
                    decoded.shape,
                    dtype=decoded.dtype,
                    device=active.control_device,
                )
        except BaseException as error:
            local_error = error
    _synchronize_error(local_error, server_args)
    assert decoded is not None and payload is not None
    coordinator.broadcast(payload, src=active.contract.writer_rank)
    return payload


def _validate_prompt_contract(batch: Req) -> None:
    prompt = batch.prompt_embeds
    prompt_mask = batch.prompt_attention_mask
    negative = batch.negative_prompt_embeds
    negative_mask = batch.negative_attention_mask
    if prompt.ndim != 3 or negative.ndim != 3:
        raise ValueError("prompt embeddings must be rank-3 tensors")
    if not prompt.is_floating_point() or not negative.is_floating_point():
        raise TypeError("prompt embeddings must use floating-point dtypes")
    if prompt_mask.ndim != 2 or negative_mask.ndim != 2:
        raise ValueError("prompt attention masks must be rank-2 tensors")
    if prompt_mask.is_floating_point() or negative_mask.is_floating_point():
        raise TypeError("prompt attention masks must use boolean or integer dtypes")
    if prompt.shape[:2] != prompt_mask.shape:
        raise ValueError("prompt embeddings and attention mask shapes disagree")
    if negative.shape[:2] != negative_mask.shape:
        raise ValueError("negative prompt embeddings and mask shapes disagree")
    if prompt.shape[0] != negative.shape[0] or prompt.shape[2] != negative.shape[2]:
        raise ValueError("positive and negative prompt embedding layouts disagree")


def _apply_metadata(batch: Req, metadata: _PrepareMetadata) -> None:
    batch.latent_shape = metadata.latent_shape
    batch.extra["ltx095_sp_mask_shape"] = metadata.mask_shape
    batch.max_sequence_length = metadata.max_sequence_length
    batch.dynamic_cfg = metadata.dynamic_cfg
    batch.cfg_step = metadata.cfg_step
    batch.enable_dynamic_cfg_space = metadata.enable_dynamic_cfg_space
    batch.guidance_scale = metadata.guidance_scale
    batch.rope_interpolation_scale = metadata.rope_interpolation_scale
    batch.fps = metadata.fps
    batch.num_inference_steps = metadata.num_inference_steps
    batch.strength = metadata.strength
    batch.extra["ltx095_sp_prepare_signature"] = {
        "latent_shape": metadata.latent_shape,
        "mask_shape": metadata.mask_shape,
        "max_sequence_length": metadata.max_sequence_length,
        "dynamic_cfg": metadata.dynamic_cfg,
        "cfg_step": metadata.cfg_step,
        "enable_dynamic_cfg_space": metadata.enable_dynamic_cfg_space,
        "guidance_scale": metadata.guidance_scale,
        "rope_interpolation_scale": metadata.rope_interpolation_scale,
        "fps": metadata.fps,
        "num_inference_steps": metadata.num_inference_steps,
        "strength": metadata.strength,
    }


def _select_cfg_prompt_branch(
    batch: Req,
    active: LTX095ActiveSPWindowContext,
) -> None:
    if not active.contract.cfg_parallel_active:
        return
    branch = getattr(active, "cfg_branch", None)
    if branch is None:
        branch = "positive" if active.cfg_group.group_rank == 0 else "negative"
    if branch == "positive":
        batch.negative_prompt_embeds = None
        batch.negative_attention_mask = None
    elif branch == "negative":
        batch.prompt_embeds = None
        batch.prompt_attention_mask = None
    else:
        raise ValueError(f"unsupported LTX095 CFG branch {branch!r}")
    batch.extra["ltx095_cfg_branch"] = branch


class LTX095EraseSequenceParallelPrepareSyncStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if not bool(batch.extra.get("ltx095_sp_writer_owned_runtime")):
            return batch
        if batch.extra.get("runtime_mode_effective") != "windowed_streaming":
            return batch
        active = resolve_active_ltx095_window_commit_context(server_args)
        if active is None:
            return batch
        coordinator = GroupCoordinator(active.world_data_group)

        writer_metadata = None
        preparation_error: BaseException | None = None
        if active.is_writer:
            try:
                writer_metadata = _build_writer_metadata(batch)
                header = _encode_scalar_header(
                    writer_metadata,
                    device=active.control_device,
                )
            except BaseException as error:
                preparation_error = error
                header = _error_header(
                    _SCALAR_HEADER_SIZE,
                    device=active.control_device,
                )
        else:
            header = torch.zeros(
                _SCALAR_HEADER_SIZE,
                dtype=torch.int64,
                device=active.control_device,
            )
        coordinator.broadcast(header, src=active.contract.writer_rank)
        metadata = None
        local_error = preparation_error
        if local_error is None:
            try:
                metadata = _decode_scalar_header(header)
                if writer_metadata is not None and metadata != writer_metadata:
                    raise ValueError(
                        "writer prepare metadata changed after header construction"
                    )
            except BaseException as error:
                local_error = error
        _synchronize_error(local_error, server_args)
        assert metadata is not None
        _apply_metadata(batch, metadata)

        for field_name in _PROMPT_FIELDS:
            setattr(
                batch,
                field_name,
                _broadcast_prompt_tensor(
                    batch=batch,
                    field_name=field_name,
                    active=active,
                    coordinator=coordinator,
                    server_args=server_args,
                ),
            )
        local_error = None
        try:
            _validate_prompt_contract(batch)
        except BaseException as error:
            local_error = error
        _synchronize_error(local_error, server_args)
        _select_cfg_prompt_branch(batch, active)
        return batch


__all__ = ("LTX095EraseSequenceParallelPrepareSyncStage",)
