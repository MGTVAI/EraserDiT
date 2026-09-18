"""LTX0.9.5 adapter for the local single-device attention backends."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional

import torch
from diffusers.models.attention_processor import Attention

from config.server_args import get_global_server_args
from layers.attention import (
    AttentionBackendEnum,
    AttentionCapability,
    AttentionImpl,
    AttentionMetadata,
    AttentionSelection,
    FlashAttentionBackend,
    SageAttentionBackend,
    SageFP8AttentionBackend,
    SDPABackend,
    SequenceParallelAttention,
    SequenceParallelBackend,
    SequenceParallelMetadata,
    resolve_attention_backend,
)
from layers.operator_fusion.qk_rmsnorm_rope import (
    apply_fused_qk_rmsnorm_rope,
)
from layers.operator_fusion.config import QK_RMSNORM_ROPE_OP
from layers.operator_fusion.registry import get_operator_fusion_decision
from models.dits.ltx095_parallel import (
    LTX095SequenceParallelControlBinding,
    synchronize_ltx095_sequence_parallel_phase,
    validate_ltx095_sequence_parallel_group,
)
from utils.logging_utils import init_logger

if TYPE_CHECKING:
    from distributed.group_coordinator import GroupCoordinator

logger = init_logger(__name__)

_BACKEND_CLASSES = {
    AttentionBackendEnum.TORCH_SDPA: SDPABackend,
    AttentionBackendEnum.FLASH_ATTN: FlashAttentionBackend,
    AttentionBackendEnum.SAGE_ATTN: SageAttentionBackend,
    AttentionBackendEnum.SAGE_FP8: SageFP8AttentionBackend,
}

# SageAttention is the preferred automatic backend for the 1080p LTX095
# runtime.  FlashAttention and SDPA remain ordered fallbacks when Sage is not
# available for the actual device/dtype/shape capability.
_LTX095_AUTO_BACKENDS = (
    AttentionBackendEnum.SAGE_ATTN,
    AttentionBackendEnum.FLASH_ATTN,
    AttentionBackendEnum.TORCH_SDPA,
)

_SelectionCacheKey = tuple[
    str,
    str,
    int | None,
    torch.dtype,
    int,
    bool,
    bool,
]
_ImplCacheKey = tuple[
    AttentionBackendEnum,
    str,
    int | None,
    torch.dtype,
    int,
    bool,
    bool,
    int,
    int,
]


@dataclass(frozen=True)
class LTX095SequenceParallelAttentionContext:
    metadata: SequenceParallelMetadata
    coordinator: GroupCoordinator
    backend: SequenceParallelBackend
    control_binding: LTX095SequenceParallelControlBinding | None = None


@dataclass
class LTX095SequenceParallelBlockControl:
    block_index: int
    before_first_collective_joined: bool = False
    before_second_collective_joined: bool = False


def _normalize_attention_backend(value: str | None) -> str:
    backend = str(value or "sdpa").strip().lower()
    if backend not in {"auto", "sdpa", "flash_attn", "sage_attn", "sage_fp8"}:
        raise ValueError(f"Unsupported attention backend: {backend}")
    return backend


def apply_rotary_emb(
    x: torch.Tensor,
    freqs: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


class LTXVideo2VideoAttentionProcessor2_0:
    """Project LTX Q/K/V and route BSHD tensors to local backends."""

    def __init__(self, attention_backend: str | None = None) -> None:
        server_args = get_global_server_args()
        if attention_backend is None:
            attention_backend = getattr(
                server_args,
                "attention_backend",
                "sdpa",
            )
        self.attention_backend = _normalize_attention_backend(attention_backend)
        self.operator_fusion_decision = get_operator_fusion_decision(server_args)
        self._self_attention_selection_cache: dict[
            _SelectionCacheKey, AttentionSelection
        ] = {}
        self._attention_impl_cache: dict[_ImplCacheKey, AttentionImpl] = {}
        self._latest_self_attention_selection: AttentionSelection | None = None
        self._initialize_sequence_parallel_context()

    def _initialize_sequence_parallel_context(self) -> None:
        self._sequence_parallel_context: ContextVar[
            LTX095SequenceParallelAttentionContext | None
        ] = ContextVar(
            f"ltx095_sequence_parallel_context_{id(self)}",
            default=None,
        )
        self._sequence_parallel_block_control: ContextVar[
            LTX095SequenceParallelBlockControl | None
        ] = ContextVar(
            f"ltx095_sequence_parallel_block_control_{id(self)}",
            default=None,
        )

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state.pop("_sequence_parallel_context", None)
        state.pop("_sequence_parallel_block_control", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._initialize_sequence_parallel_context()

    @contextmanager
    def sequence_parallel_scope(
        self,
        context: LTX095SequenceParallelAttentionContext,
    ) -> Iterator[None]:
        """Bind explicit per-call SP state without leaking shared processor state."""
        if not isinstance(context, LTX095SequenceParallelAttentionContext):
            raise TypeError("context must be LTX095SequenceParallelAttentionContext")
        token = self._sequence_parallel_context.set(context)
        try:
            yield
        finally:
            self._sequence_parallel_context.reset(token)

    @contextmanager
    def sequence_parallel_block_scope(
        self,
        block_index: int,
    ) -> Iterator[LTX095SequenceParallelBlockControl]:
        """Track the fixed pre-collective control phases for one block."""
        context = self._sequence_parallel_context.get()
        if context is None or context.control_binding is None:
            raise RuntimeError(
                "sequence-parallel block scope requires a bound control context"
            )
        if type(block_index) is not int or block_index < 0:
            raise ValueError("block_index must be a non-negative int")
        control = LTX095SequenceParallelBlockControl(block_index=block_index)
        token = self._sequence_parallel_block_control.set(control)
        try:
            yield control
        finally:
            self._sequence_parallel_block_control.reset(token)

    def synchronize_before_collective(
        self,
        collective_index: int,
        local_error: BaseException | None,
    ) -> None:
        """Join a block's indexed pre-A2A phase once on the trusted group."""
        context = self._sequence_parallel_context.get()
        control = self._sequence_parallel_block_control.get()
        if context is None or context.control_binding is None or control is None:
            raise RuntimeError(
                "pre-A2A synchronization requires an active block control scope"
            )
        if collective_index == 1:
            joined_attribute = "before_first_collective_joined"
            phase_name = "pre-first-A2A preparation"
        elif collective_index == 2:
            joined_attribute = "before_second_collective_joined"
            phase_name = "pre-second-A2A preparation"
        else:
            raise ValueError("collective_index must be 1 or 2")
        if getattr(control, joined_attribute):
            raise RuntimeError(
                f"pre-A2A {collective_index} synchronization already joined "
                "for this block"
            )
        setattr(control, joined_attribute, True)
        synchronize_ltx095_sequence_parallel_phase(
            local_error,
            binding=context.control_binding,
            phase=f"block {control.block_index} {phase_name}",
        )

    def prepare_sequence_parallel_call(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        metadata: SequenceParallelMetadata,
        coordinator: GroupCoordinator,
        control_binding: LTX095SequenceParallelControlBinding | None = None,
    ) -> LTX095SequenceParallelAttentionContext:
        """Validate the per-call data plane and prepare its frozen backend."""
        validate_ltx095_sequence_parallel_group(metadata, coordinator)
        backend = self._prepare_sequence_parallel_backend(
            attn,
            hidden_states,
            image_rotary_emb,
            metadata,
        )
        return LTX095SequenceParallelAttentionContext(
            metadata=metadata,
            coordinator=coordinator,
            backend=backend,
            control_binding=control_binding,
        )

    def _prepare_sequence_parallel_backend(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        metadata: SequenceParallelMetadata,
    ) -> SequenceParallelBackend:
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a torch.Tensor")
        if hidden_states.ndim != 3:
            raise ValueError("LTX095 SP hidden_states must use BSH rank-3 layout")
        if hidden_states.shape[1] != metadata.local_length:
            raise ValueError(
                "LTX095 SP local sequence length must match metadata local length"
            )

        heads = getattr(attn, "heads", None)
        if type(heads) is not int or heads <= 0:
            raise ValueError("LTX095 SP attention heads must be a positive int")
        if heads % metadata.sp_degree != 0:
            raise ValueError("LTX095 SP attention heads must be divisible by sp_degree")

        projections = {
            name: getattr(attn, name, None) for name in ("to_q", "to_k", "to_v")
        }
        widths: dict[str, int] = {}
        for name, projection in projections.items():
            input_width = getattr(projection, "in_features", None)
            output_width = getattr(projection, "out_features", None)
            if type(input_width) is not int or input_width != hidden_states.shape[-1]:
                raise ValueError(
                    f"LTX095 SP {name} input width must match hidden width"
                )
            if type(output_width) is not int or output_width <= 0:
                raise ValueError(f"LTX095 SP {name} output width must be positive")
            widths[name] = output_width
        if len(set(widths.values())) != 1:
            raise ValueError("LTX095 SP self-attention Q/K/V widths must match exactly")

        projection_width = widths["to_q"]
        if projection_width % heads != 0:
            raise ValueError(
                "LTX095 SP query projection width must be divisible by heads"
            )
        head_dim = projection_width // heads
        if projection_width % 2 != 0:
            raise ValueError("LTX095 SP rotary projection width must be even")

        if image_rotary_emb is not None:
            if type(image_rotary_emb) is not tuple or len(image_rotary_emb) != 2:
                raise TypeError("image_rotary_emb must be a (cos, sin) tuple")
            target_shape = (
                hidden_states.shape[0],
                hidden_states.shape[1],
                projection_width,
            )
            cos, sin = image_rotary_emb
            for name, frequency in (("cos", cos), ("sin", sin)):
                if not isinstance(frequency, torch.Tensor):
                    raise TypeError(f"rotary {name} must be a torch.Tensor")
                if frequency.device != hidden_states.device:
                    raise ValueError(
                        f"rotary {name} device must match hidden_states device"
                    )
                try:
                    broadcast_shape = torch.broadcast_shapes(
                        target_shape,
                        tuple(frequency.shape),
                    )
                except RuntimeError as error:
                    raise ValueError(
                        f"rotary {name} must be broadcastable to projected Q/K"
                    ) from error
                if broadcast_shape != target_shape:
                    raise ValueError(
                        f"rotary {name} must not expand projected Q/K layout"
                    )

        query = hidden_states.new_empty(
            hidden_states.shape[0],
            hidden_states.shape[1],
            heads,
            head_dim,
        )
        selection, impl = self._resolve_self_attention(
            query,
            query,
            has_attn_mask=False,
        )
        backend = SequenceParallelBackend(selection.selected, impl)
        if backend.selected is not metadata.backend:
            raise ValueError(
                "LTX095 SP backend selection must match sequence-parallel metadata backend"
            )
        if metadata.backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.SAGE_ATTN,
        } and metadata.padded_length != metadata.global_length:
            raise AttentionBackendUnavailableError(
                metadata.backend,
                "sequence padding requires an explicit attention mask",
            )
        return backend

    @property
    def effective_self_attention_backend(self) -> str | None:
        selection = self._latest_self_attention_selection
        return selection.selected.value if selection is not None else None

    @property
    def self_attention_fallback_reasons(self) -> tuple[str, ...]:
        selection = self._latest_self_attention_selection
        return selection.fallback_reasons if selection is not None else ()

    def preflight_self_attention_backend(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        head_size: int = 64,
        num_heads: int = 32,
    ) -> dict[str, object]:
        capability = AttentionCapability(
            device=device,
            dtype=dtype,
            head_size=head_size,
            causal=False,
            has_attn_mask=False,
        )
        selection = resolve_attention_backend(self.attention_backend, capability)
        self._latest_self_attention_selection = selection
        impl = self._get_attention_impl(
            selection.selected,
            capability,
            num_heads=num_heads,
            num_kv_heads=num_heads,
        )
        return self.attention_backend_report(impl=impl)

    def attention_backend_report(
        self,
        *,
        impl: AttentionImpl | None = None,
    ) -> dict[str, object]:
        selection = self._latest_self_attention_selection
        fallback_reasons = selection.fallback_reasons if selection is not None else ()
        report: dict[str, object] = {
            "requested": self.attention_backend,
            "effective": selection.selected.value if selection is not None else None,
            "cross_attention_backend": "torch_sdpa",
            "fallback_reasons": list(fallback_reasons),
            "fallback_count": len(fallback_reasons),
        }
        candidates = [impl] if impl is not None else list(self._attention_impl_cache.values())
        strict_reports = [
            candidate.report()
            for candidate in candidates
            if candidate is not None and callable(getattr(candidate, "report", None))
        ]
        if strict_reports:
            report.update(strict_reports[0])
            report["call_count"] = sum(int(item["call_count"]) for item in strict_reports)
            report["failure_count"] = sum(
                int(item["failure_count"]) for item in strict_reports
            )
        return report

    @staticmethod
    def _selection_cache_key(
        requested: str,
        capability: AttentionCapability,
    ) -> _SelectionCacheKey:
        return (
            requested,
            capability.device.type,
            capability.device.index,
            capability.dtype,
            capability.head_size,
            capability.causal,
            capability.has_attn_mask,
        )

    @staticmethod
    def _impl_cache_key(
        selected_backend: AttentionBackendEnum,
        capability: AttentionCapability,
        *,
        num_heads: int,
        num_kv_heads: int,
    ) -> _ImplCacheKey:
        return (
            selected_backend,
            capability.device.type,
            capability.device.index,
            capability.dtype,
            capability.head_size,
            capability.causal,
            capability.has_attn_mask,
            num_heads,
            num_kv_heads,
        )

    def _get_attention_impl(
        self,
        selected_backend: AttentionBackendEnum,
        capability: AttentionCapability,
        *,
        num_heads: int,
        num_kv_heads: int,
    ) -> AttentionImpl:
        cache_key = self._impl_cache_key(
            selected_backend,
            capability,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        impl = self._attention_impl_cache.get(cache_key)
        if impl is not None:
            return impl

        backend_cls = _BACKEND_CLASSES[selected_backend]
        impl = backend_cls.get_impl_cls()(
            num_heads=num_heads,
            head_size=capability.head_size,
            softmax_scale=capability.head_size**-0.5,
            causal=capability.causal,
            num_kv_heads=num_kv_heads,
            prefix="ltx095_attention.impl",
            dropout_p=0.0,
        )
        self._attention_impl_cache[cache_key] = impl
        return impl

    def _resolve_self_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        has_attn_mask: bool,
    ) -> tuple[AttentionSelection, AttentionImpl]:
        capability = AttentionCapability(
            device=query.device,
            dtype=query.dtype,
            head_size=query.shape[-1],
            causal=False,
            has_attn_mask=has_attn_mask,
        )
        selection_key = self._selection_cache_key(
            self.attention_backend,
            capability,
        )
        selection = self._self_attention_selection_cache.get(selection_key)
        if selection is None:
            resolve_kwargs = (
                {"auto_backends": _LTX095_AUTO_BACKENDS}
                if self.attention_backend == "auto"
                else {}
            )
            selection = resolve_attention_backend(
                self.attention_backend,
                capability,
                **resolve_kwargs,
            )
            self._self_attention_selection_cache[selection_key] = selection
            logger.info(
                "Resolved LTX095 self-attention backend: "
                "requested_self_attention_backend=%s "
                "effective_self_attention_backend=%s fallback_reasons=%s",
                selection.requested,
                selection.selected.value,
                "; ".join(selection.fallback_reasons) or "none",
            )
        self._latest_self_attention_selection = selection
        return (
            selection,
            self._get_attention_impl(
                selection.selected,
                capability,
                num_heads=query.shape[2],
                num_kv_heads=key.shape[2],
            ),
        )

    def _get_cross_attention_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        has_attn_mask: bool,
    ) -> AttentionImpl:
        capability = AttentionCapability(
            device=query.device,
            dtype=query.dtype,
            head_size=query.shape[-1],
            causal=False,
            has_attn_mask=has_attn_mask,
        )
        return self._get_attention_impl(
            AttentionBackendEnum.TORCH_SDPA,
            capability,
            num_heads=query.shape[2],
            num_kv_heads=key.shape[2],
        )

    @staticmethod
    def _project_output(
        attn: Attention,
        hidden_states: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        hidden_states = hidden_states.flatten(2, 3).to(output_dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    @staticmethod
    def _prepare_attention_mask(
        attn: Attention,
        attention_mask: torch.Tensor | None,
        *,
        sequence_length: int,
        batch_size: int,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        attention_mask = attn.prepare_attention_mask(
            attention_mask,
            sequence_length,
            batch_size,
        )
        return attention_mask.view(
            batch_size,
            attn.heads,
            -1,
            attention_mask.shape[-1],
        )

    @staticmethod
    def _reshape_qkv(
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if query.shape[-1] % attn.heads != 0:
            raise ValueError(
                "LTX query projection width must be divisible by query heads"
            )
        head_dim = query.shape[-1] // attn.heads
        if head_dim <= 0:
            raise ValueError("LTX attention head_dim must be positive")
        if key.shape[-1] != value.shape[-1]:
            raise ValueError("LTX key/value projection widths must match")
        if key.shape[-1] % head_dim != 0:
            raise ValueError(
                "LTX key/value projection width must be divisible by head_dim"
            )

        num_kv_heads = key.shape[-1] // head_dim
        if num_kv_heads <= 0 or attn.heads % num_kv_heads != 0:
            raise ValueError(
                "LTX query heads must be a positive integer multiple of KV heads"
            )
        return (
            query.unflatten(2, (attn.heads, head_dim)),
            key.unflatten(2, (num_kv_heads, head_dim)),
            value.unflatten(2, (num_kv_heads, head_dim)),
        )

    def self_attn(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = self._prepare_attention_mask(
            attn,
            attention_mask,
            sequence_length=sequence_length,
            batch_size=batch_size,
        )
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        if QK_RMSNORM_ROPE_OP in self.operator_fusion_decision.effective_ops:

            def reference_qk_norm_rope() -> tuple[torch.Tensor, torch.Tensor]:
                reference_query = attn.norm_q(query)
                reference_key = attn.norm_k(key)
                if image_rotary_emb is not None:
                    reference_query = apply_rotary_emb(
                        reference_query, image_rotary_emb
                    )
                    reference_key = apply_rotary_emb(
                        reference_key, image_rotary_emb
                    )
                return reference_query, reference_key

            query, key = apply_fused_qk_rmsnorm_rope(
                query,
                key,
                attn.norm_q,
                attn.norm_k,
                image_rotary_emb,
                decision=self.operator_fusion_decision,
                reference=reference_qk_norm_rope,
            )
        else:
            query = attn.norm_q(query)
            key = attn.norm_k(key)
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        query, key, value = self._reshape_qkv(attn, query, key, value)
        attn_metadata = AttentionMetadata(attn_mask=attention_mask)
        sequence_parallel_context = self._sequence_parallel_context.get()
        if sequence_parallel_context is None:
            _selection, impl = self._resolve_self_attention(
                query,
                key,
                has_attn_mask=attention_mask is not None,
            )
            hidden_states = impl.forward(
                query,
                key,
                value,
                attn_metadata,
            )
        else:
            sequence_parallel_attention = SequenceParallelAttention(
                sequence_parallel_context.coordinator
            )
            sequence_parallel_kwargs = {}
            if sequence_parallel_context.control_binding is not None:
                sequence_parallel_kwargs["before_collective"] = (
                    self.synchronize_before_collective
                )
            hidden_states = sequence_parallel_attention.forward(
                query,
                key,
                value,
                backend=sequence_parallel_context.backend,
                metadata=sequence_parallel_context.metadata,
                attn_metadata=attn_metadata,
                **sequence_parallel_kwargs,
            )
        return self._project_output(attn, hidden_states, query.dtype)

    def cross_attn(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        attention_mask = self._prepare_attention_mask(
            attn,
            attention_mask,
            sequence_length=sequence_length,
            batch_size=batch_size,
        )

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        query, key, value = self._reshape_qkv(attn, query, key, value)
        impl = self._get_cross_attention_impl(
            query,
            key,
            has_attn_mask=attention_mask is not None,
        )
        hidden_states = impl.forward(
            query,
            key,
            value,
            AttentionMetadata(attn_mask=attention_mask),
        )
        return self._project_output(attn, hidden_states, query.dtype)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            return self.self_attn(
                attn=attn,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )
        return self.cross_attn(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
