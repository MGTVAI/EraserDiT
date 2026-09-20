"""EraserDiT binding for budgeted pinned-weight extents.

The budget bounds managed extents, not activations or the small unwrapped
parameters. Their sizes are reported separately. VAE normalization buffers
remain on CPU; the model's normalization helpers copy them to the latent device.
"""

from torch import nn

from memory.adapters.model_memory_adapter import (
    ModelMemoryAdapter,
    _candidate_modules,
    _unique_module_bytes,
)
from memory.backends.federated_storage import FederatedStorage
from memory.backends.flexible_module_extent_base import FlexibleModuleExtentBase


def _plan_extents(module, *, min_size, max_size, prefix, seen):
    if id(module) in seen:
        return []
    seen.add(id(module))
    if hasattr(module, "flexible_extent"):
        raise ValueError(f"{prefix} already has a flexible extent")
    size = FlexibleModuleExtentBase.get_module_memory_require(
        module, contain_grad=False
    )
    if not isinstance(module, nn.ModuleList):
        if min_size <= size <= max_size:
            return [(prefix, module)]
        if size < min_size:
            return []
    children = list(module.named_children())
    if not children and size > max_size:
        raise ValueError(
            f"{prefix} requires {size} bytes, exceeding max_weight_usage "
            f"or extent limit {max_size}"
        )
    result = []
    for name, child in children:
        result.extend(_plan_extents(
            child, min_size=min_size, max_size=max_size,
            prefix=f"{prefix}.{name}", seen=seen,
        ))
    return result


class EraserDiTMemoryAdapter(ModelMemoryAdapter):
    def _coalesce_t5_embedding(self, modules):
        from transformers import T5EncoderModel

        text = modules.get("text_encoder")
        if not isinstance(text, T5EncoderModel):
            return
        shared, embedded = text.shared, text.encoder.embed_tokens
        if shared is embedded:
            return
        # Recent Transformers uses two Embedding objects with a tied Parameter.
        # One physical embedding must have one extent; otherwise rewriting the
        # Parameter views would sever the tie. Use T5's supported setter.
        if shared.weight is not embedded.weight or any(
            getattr(shared, key) != getattr(embedded, key)
            for key in ("padding_idx", "max_norm", "norm_type", "scale_grad_by_freq", "sparse")
        ):
            raise ValueError("dynamic offload requires equivalent tied T5 embeddings")
        self._t5_embedding_alias = (text, embedded)
        text.set_input_embeddings(shared)

    def _restore_t5_embedding(self):
        alias = getattr(self, "_t5_embedding_alias", None)
        if alias is not None:
            text, embedded = alias
            embedded.weight = text.shared.weight
            text.encoder.set_input_embeddings(embedded)
            self._t5_embedding_alias = None

    def register(self, **kwargs):
        if kwargs["dynamic_offload"]:
            self._coalesce_t5_embedding(kwargs["modules"])
        try:
            return self._register(**kwargs)
        except BaseException:
            self._restore_t5_embedding()
            raise

    def _register(self, **kwargs):
        candidates = _candidate_modules(kwargs["modules"])
        self._non_extent_bytes = {}
        if kwargs["dynamic_offload"]:
            self.extent_max_size = min(500 * 1024**2, int(kwargs["max_weight_usage"]))
            if self.extent_max_size <= 0:
                raise ValueError("max_weight_usage must be positive")
            planned = []
            seen = set()
            for name, module in candidates:
                planned.extend(_plan_extents(
                    module, min_size=min(15 * 1024**2, self.extent_max_size),
                    max_size=self.extent_max_size, prefix=name, seen=seen,
                ))
            # Validate before mutating any parameter storage. Shared modules
            # (T5 embedding) are one extent; aliases across distinct owners are
            # rejected rather than silently detached by registration.
            groups = {
                name: dict(module.named_parameters(remove_duplicate=False))
                | dict(module.named_buffers(remove_duplicate=False))
                for name, module in planned
            }
            owned = {id(child) for _, module in planned for child in module.modules()}
            remainder = {}
            for prefix, module in candidates:
                for name, child in module.named_modules():
                    if id(child) not in owned:
                        for key, value in list(child._parameters.items()) + list(child._buffers.items()):
                            if value is not None:
                                remainder[f"{prefix}.{name}.{key}"] = value
            groups["unwrapped"] = remainder
            FederatedStorage.validate_no_cross_extent_aliases(groups)
        try:
            summary = super().register(**kwargs)
        except BaseException:
            # Registration can fail half way through a component. Include
            # extents not yet appended to the base adapter's registry.
            found = {}
            for _, module in candidates:
                for child in module.modules():
                    extent = getattr(child, "flexible_extent", None)
                    if extent is not None:
                        found[id(extent)] = extent
            self._extents = list(found.values())
            self.shutdown(terminal=False)
            raise
        self._non_extent_bytes = {
            name: _unique_module_bytes(module, skip_flexible=True)
            for name, module in candidates
        }
        return summary

    def shutdown(self, *, terminal=False):
        was_closed = self._closed
        snapshot = super().shutdown(terminal=terminal)
        if (
            not was_closed and self._dynamic_offload and not self._extents
            and self._device is not None and self._device.type == "cuda"
        ):
            self.reset_device_fn(self._device)
        self._restore_t5_embedding()
        return snapshot

    def onload_remainder(self, component_name):
        self.onload_module_fn(
            self._component_modules[component_name], self._device,
            contain_sub=True, skip_flexible=True, check_device=False,
        )

    def offload_remainder(self, component_name):
        # Complete asynchronous layer offloads before the next phase/request.
        self.settle_component_transfers()
        self.offload_module_fn(
            self._component_modules[component_name],
            contain_sub=True, skip_flexible=True, check_device=False,
        )

    def _live_snapshot(self):
        result = super()._live_snapshot()
        result["non_extent_bytes"] = dict(getattr(self, "_non_extent_bytes", {}))
        result["weight_budget_scope"] = "managed_extents_only"
        result["extent_storage"] = "pinned_cpu_mirror" if self._dynamic_offload else None
        return result
