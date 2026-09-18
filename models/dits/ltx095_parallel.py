"""LTX0.9.5 P3-P5 sequence, CFG, and VAE parallel capability contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import torch

from config.transformer_cache import validate_transformer_cache_request
from layers.attention.sequence_parallel import SequenceParallelMetadata
from distributed import communication_op
from parallel.planner import ResolvedAccelerationPlan

if TYPE_CHECKING:
    from distributed.group_coordinator import GroupCoordinator


@dataclass(frozen=True)
class LTX095SequenceParallelRuntimeOptions:
    parallel_mode: str
    distributed_compute_mode: str
    enable_torch_compile: bool = False
    transformer_cache_mode: str = "off"
    teacache_threshold: float = 0.0
    do_teacache_calibrate: bool = False
    global_rank: int = 0
    local_rank: int = 0
    device: str = "cuda:0"


@dataclass(frozen=True)
class LTX095SequenceParallelContract:
    active: bool
    world_size: int
    sp_degree: int
    cfg_degree: int
    vae_degree: int
    writer_rank: int
    num_attention_heads: int | None
    attention_head_dim: int | None
    hidden_size: int | None
    attention_backend: str
    distributed_compute_mode: str

    @property
    def sequence_parallel_active(self) -> bool:
        return self.active and self.sp_degree > 1

    @property
    def cfg_parallel_active(self) -> bool:
        return self.active and self.cfg_degree == 2

    @property
    def vae_parallel_active(self) -> bool:
        return self.active and self.vae_degree > 1


class LTX095SequenceParallelPreflightPeerError(RuntimeError):
    """Report a peer's failure before the first Ulysses collective."""

    def __init__(self, peer_ranks: tuple[int, ...]) -> None:
        self.peer_ranks = peer_ranks
        ranks = ", ".join(str(rank) for rank in peer_ranks)
        super().__init__(
            "LTX095 sequence-parallel preflight failed on peer rank(s): " + ranks
        )


class LTX095SequenceParallelPeerError(RuntimeError):
    """Report a peer failure at a fixed model-internal control phase."""

    def __init__(self, peer_ranks: tuple[int, ...], *, phase: str) -> None:
        self.peer_ranks = peer_ranks
        self.phase = phase
        ranks = ", ".join(str(rank) for rank in peer_ranks)
        super().__init__(
            f"LTX095 sequence-parallel {phase} failed on peer rank(s): {ranks}"
        )


@dataclass(frozen=True)
class LTX095SequenceParallelControlBinding:
    """Trusted, process-local control plane frozen after model placement."""

    coordinator: GroupCoordinator
    group_slot: int
    group_ranks: tuple[int, ...]
    control_device: torch.device
    control_dtype: torch.dtype

    @property
    def local_flag_shape(self) -> tuple[int]:
        return (1,)

    @property
    def gathered_flag_shape(self) -> tuple[int]:
        return (len(self.group_ranks),)


def create_ltx095_sequence_parallel_control_binding(
    coordinator: GroupCoordinator,
    *,
    control_device: torch.device | str,
) -> LTX095SequenceParallelControlBinding:
    """Validate and freeze the coordinator fields used by preflight control."""
    if not callable(getattr(coordinator, "all_gather_into_tensor", None)):
        raise TypeError(
            "sequence-parallel control coordinator must provide "
            "all_gather_into_tensor"
        )
    group = getattr(coordinator, "group", None)
    group_slot = getattr(group, "group_rank", None)
    group_ranks = getattr(getattr(group, "spec", None), "ranks", None)
    if type(group_ranks) is not tuple:
        raise TypeError("sequence-parallel control group ranks must be a tuple")
    if not group_ranks:
        raise ValueError("sequence-parallel control group ranks must not be empty")
    if any(type(rank) is not int or rank < 0 for rank in group_ranks):
        raise ValueError(
            "sequence-parallel control group ranks must be non-negative ints"
        )
    if len(set(group_ranks)) != len(group_ranks):
        raise ValueError(
            "sequence-parallel control group ranks must not contain duplicates"
        )
    if type(group_slot) is not int or not 0 <= group_slot < len(group_ranks):
        raise ValueError("sequence-parallel control group slot must fit group ranks")
    if getattr(coordinator, "rank", None) != group_slot:
        raise ValueError(
            "sequence-parallel control coordinator rank must match group slot"
        )
    if getattr(coordinator, "world_size", None) != len(group_ranks):
        raise ValueError(
            "sequence-parallel control coordinator world_size must match group ranks"
        )
    return LTX095SequenceParallelControlBinding(
        coordinator=coordinator,
        group_slot=group_slot,
        group_ranks=tuple(group_ranks),
        control_device=torch.device(control_device),
        control_dtype=torch.int32,
    )


def validate_ltx095_sequence_parallel_execution(
    metadata: SequenceParallelMetadata | None,
    coordinator: GroupCoordinator | None,
) -> tuple[SequenceParallelMetadata | None, GroupCoordinator | None]:
    """Validate one explicit model-call sequence-parallel binding."""
    if (metadata is None) != (coordinator is None):
        raise ValueError(
            "sequence_parallel_metadata and sequence_parallel_coordinator "
            "must be provided together"
        )
    if metadata is None:
        return None, None
    if not isinstance(metadata, SequenceParallelMetadata):
        raise TypeError(
            "sequence_parallel_metadata must be SequenceParallelMetadata or None"
        )

    coordinator_world_size = getattr(coordinator, "world_size", None)
    if coordinator_world_size != metadata.sp_degree:
        raise ValueError(
            "sequence_parallel_coordinator world_size must match metadata "
            f"sp_degree, got world_size={coordinator_world_size!r}, "
            f"sp_degree={metadata.sp_degree}"
        )
    coordinator_rank = getattr(coordinator, "rank", None)
    if coordinator_rank != metadata.rank:
        raise ValueError(
            "sequence_parallel_coordinator rank must match metadata rank, "
            f"got rank={coordinator_rank!r}, metadata.rank={metadata.rank}"
        )
    return metadata, coordinator


def validate_ltx095_sequence_parallel_group(
    metadata: SequenceParallelMetadata,
    coordinator: GroupCoordinator,
) -> None:
    """Validate the coordinator group fields consumed by Ulysses."""
    validate_ltx095_sequence_parallel_execution(metadata, coordinator)
    group = getattr(coordinator, "group", None)
    spec = getattr(group, "spec", None)
    group_ranks = getattr(spec, "ranks", None)
    if type(group_ranks) is not tuple:
        raise TypeError("sequence_parallel_coordinator group ranks must be a tuple")
    if len(group_ranks) != metadata.sp_degree:
        raise ValueError(
            "sequence_parallel_coordinator group ranks must match metadata "
            f"sp_degree, got ranks={group_ranks!r}, sp_degree={metadata.sp_degree}"
        )
    if any(type(rank) is not int or rank < 0 for rank in group_ranks):
        raise ValueError(
            "sequence_parallel_coordinator group ranks must be non-negative ints"
        )
    if len(set(group_ranks)) != len(group_ranks):
        raise ValueError(
            "sequence_parallel_coordinator group ranks must not contain duplicates"
        )
    group_rank = getattr(group, "group_rank", metadata.rank)
    if group_rank != metadata.rank:
        raise ValueError(
            "sequence_parallel_coordinator group rank must match metadata rank, "
            f"got group_rank={group_rank!r}, metadata.rank={metadata.rank}"
        )


@communication_op._disable_torch_compile
def synchronize_ltx095_sequence_parallel_preflight(
    local_error: BaseException | None,
    *,
    binding: LTX095SequenceParallelControlBinding,
) -> None:
    """Gather one fixed flag per trusted group slot before Ulysses all-to-all."""
    if not isinstance(binding, LTX095SequenceParallelControlBinding):
        raise TypeError("binding must be LTX095SequenceParallelControlBinding")
    local_flag = torch.full(
        binding.local_flag_shape,
        1 if local_error is not None else 0,
        dtype=binding.control_dtype,
        device=binding.control_device,
    )
    gathered_flags = torch.empty(
        binding.gathered_flag_shape,
        dtype=binding.control_dtype,
        device=binding.control_device,
    )
    binding.coordinator.all_gather_into_tensor(gathered_flags, local_flag)
    failed_slots = gathered_flags.ne(0).nonzero().flatten().tolist()

    if local_error is not None:
        raise local_error
    if not failed_slots:
        return

    peer_ranks = tuple(binding.group_ranks[slot] for slot in failed_slots)
    raise LTX095SequenceParallelPreflightPeerError(peer_ranks)


@communication_op._disable_torch_compile
def synchronize_ltx095_sequence_parallel_phase(
    local_error: BaseException | None,
    *,
    binding: LTX095SequenceParallelControlBinding,
    phase: str,
) -> None:
    """Synchronize one fixed-width model-internal error flag across SP ranks."""
    if not isinstance(binding, LTX095SequenceParallelControlBinding):
        raise TypeError("binding must be LTX095SequenceParallelControlBinding")
    if not isinstance(phase, str) or not phase:
        raise ValueError("phase must be a non-empty string")
    local_flag = torch.full(
        binding.local_flag_shape,
        1 if local_error is not None else 0,
        dtype=binding.control_dtype,
        device=binding.control_device,
    )
    gathered_flags = torch.empty(
        binding.gathered_flag_shape,
        dtype=binding.control_dtype,
        device=binding.control_device,
    )
    binding.coordinator.all_gather_into_tensor(gathered_flags, local_flag)
    failed_slots = gathered_flags.ne(0).nonzero().flatten().tolist()

    if local_error is not None:
        raise local_error
    if not failed_slots:
        return

    peer_ranks = tuple(binding.group_ranks[slot] for slot in failed_slots)
    raise LTX095SequenceParallelPeerError(peer_ranks, phase=phase)


def _build_contract(
    *,
    active: bool,
    plan: ResolvedAccelerationPlan,
    num_attention_heads: int | None,
    attention_head_dim: int | None,
    hidden_size: int | None,
    attention_backend: str,
    distributed_compute_mode: str,
) -> LTX095SequenceParallelContract:
    return LTX095SequenceParallelContract(
        active=active,
        world_size=plan.world_size,
        sp_degree=plan.sp_degree,
        cfg_degree=plan.cfg_degree,
        vae_degree=plan.vae_degree,
        writer_rank=plan.writer_rank,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        hidden_size=hidden_size,
        attention_backend=attention_backend,
        distributed_compute_mode=distributed_compute_mode,
    )


def _optional_positive_int(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


def _validate_rank(field: str, value: Any, world_size: int) -> None:
    if type(value) is not int or not 0 <= value < world_size:
        raise ValueError(
            f"LTX095 P5 requires {field} to be a non-bool int in "
            f"[0, {world_size}), got {field}={value!r}"
        )


def validate_ltx095_sequence_parallel_capability(
    *,
    plan: ResolvedAccelerationPlan,
    attention_backend: str,
    transformer_config: Mapping[str, Any],
    runtime_options: LTX095SequenceParallelRuntimeOptions,
) -> LTX095SequenceParallelContract:
    """Validate and freeze the executable P5 LTX095 parallel capability."""
    num_attention_heads = transformer_config.get("num_attention_heads")
    attention_head_dim = transformer_config.get("attention_head_dim")
    validate_transformer_cache_request(
        mode=runtime_options.transformer_cache_mode,
        enable_torch_compile=runtime_options.enable_torch_compile,
    )
    if runtime_options.enable_torch_compile:
        compile_topology = (
            plan.world_size,
            plan.sp_degree,
            plan.cfg_degree,
            plan.vae_degree,
        )
        supported_compile_topologies = {
            (1, 1, 1, 1),
            (2, 2, 1, 2),
            (4, 4, 1, 4),
        }
        if compile_topology not in supported_compile_topologies:
            raise ValueError(
                "LTX095 torch compile requires one of the recommended "
                "(world_size, sp_degree, cfg_degree, vae_degree) topologies "
                "(1, 1, 1, 1), (2, 2, 1, 2), or (4, 4, 1, 4), got "
                f"{compile_topology}"
            )
    if runtime_options.parallel_mode == "disabled" or (
        runtime_options.parallel_mode == "auto" and not plan.enabled
    ):
        num_attention_heads = _optional_positive_int(num_attention_heads)
        attention_head_dim = _optional_positive_int(attention_head_dim)
        return _build_contract(
            active=False,
            plan=plan,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            hidden_size=(
                num_attention_heads * attention_head_dim
                if num_attention_heads is not None and attention_head_dim is not None
                else None
            ),
            attention_backend=attention_backend,
            distributed_compute_mode=runtime_options.distributed_compute_mode,
        )

    if runtime_options.parallel_mode not in {"auto", "manual"}:
        raise ValueError(
            "LTX095 P5 requires parallel_mode='auto' or 'manual', "
            f"got parallel_mode={runtime_options.parallel_mode!r}"
        )
    if plan.enabled is not True:
        raise ValueError(
            "LTX095 P5 requires plan.enabled=True, "
            f"got plan.enabled={plan.enabled!r}"
        )
    supported_topologies = {
        (2, 2, 1, 1),
        (2, 2, 1, 2),
        (2, 1, 2, 1),
        (4, 4, 1, 1),
        (4, 4, 1, 4),
        (4, 2, 2, 1),
        (4, 2, 2, 4),
    }
    topology = (
        plan.world_size,
        plan.sp_degree,
        plan.cfg_degree,
        plan.vae_degree,
    )
    if topology not in supported_topologies:
        raise ValueError(
            "LTX095 P5 requires (world_size, sp_degree, cfg_degree, vae_degree) "
            "to be one of (2, 2, 1, 1), (2, 2, 1, 2), (2, 1, 2, 1), "
            "(4, 4, 1, 1), (4, 4, 1, 4), (4, 2, 2, 1), "
            "(4, 2, 2, 4), got "
            f"world_size={plan.world_size}, sp_degree={plan.sp_degree}, "
            f"cfg_degree={plan.cfg_degree}, vae_degree={plan.vae_degree}"
        )
    if plan.writer_rank != 0:
        raise ValueError(
            f"LTX095 P5 requires writer_rank=0, got writer_rank={plan.writer_rank}"
        )
    if runtime_options.distributed_compute_mode != "entry_only":
        raise ValueError(
            "LTX095 P5 requires distributed_compute_mode='entry_only', "
            "got distributed_compute_mode="
            f"{runtime_options.distributed_compute_mode!r}"
        )
    _validate_rank("global_rank", runtime_options.global_rank, plan.world_size)
    _validate_rank("local_rank", runtime_options.local_rank, plan.world_size)
    if runtime_options.global_rank != runtime_options.local_rank:
        raise ValueError(
            "LTX095 P5 single-machine execution requires global_rank=local_rank, "
            f"got global_rank={runtime_options.global_rank}, "
            f"local_rank={runtime_options.local_rank}"
        )
    expected_device = f"cuda:{runtime_options.local_rank}"
    if runtime_options.device != expected_device:
        raise ValueError(
            "LTX095 P5 requires device to exactly match local_rank, "
            f"got device={runtime_options.device!r}, "
            f"local_rank={runtime_options.local_rank}, expected_device={expected_device!r}"
        )
    if attention_backend not in {
        "sdpa",
        "flash_attn",
        "sage_attn",
        "sage_fp8",
    }:
        raise ValueError(
            "LTX095 P5 requires attention_backend in "
            "{'sdpa', 'flash_attn', 'sage_attn', 'sage_fp8'}, "
            f"got attention_backend={attention_backend!r}"
        )
    if type(num_attention_heads) is not int or num_attention_heads <= 0:
        raise ValueError(
            "LTX095 P5 requires num_attention_heads to be a positive int, "
            f"got num_attention_heads={num_attention_heads!r}"
        )
    if num_attention_heads % plan.sp_degree != 0:
        raise ValueError(
            "LTX095 P5 requires num_attention_heads to be divisible by sp_degree, "
            f"got num_attention_heads={num_attention_heads}, "
            f"sp_degree={plan.sp_degree}"
        )
    if type(attention_head_dim) is not int or attention_head_dim <= 0:
        raise ValueError(
            "LTX095 P5 requires attention_head_dim to be a positive int, "
            f"got attention_head_dim={attention_head_dim!r}"
        )
    hidden_size = num_attention_heads * attention_head_dim
    configured_hidden_size = transformer_config.get("hidden_size")
    if "hidden_size" in transformer_config and (
        type(configured_hidden_size) is not int
        or configured_hidden_size <= 0
        or configured_hidden_size != hidden_size
    ):
        raise ValueError(
            "LTX095 P5 requires optional hidden_size to be a positive int equal "
            "to num_attention_heads * attention_head_dim, "
            f"got hidden_size={configured_hidden_size!r}, "
            f"computed_hidden_size={hidden_size}"
        )
    return _build_contract(
        active=True,
        plan=plan,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        hidden_size=hidden_size,
        attention_backend=attention_backend,
        distributed_compute_mode=runtime_options.distributed_compute_mode,
    )
