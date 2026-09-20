"""Phase-scoped component or dynamic transfers for EraserDiT inference.

The simple policy uses blocking Module.to at stage boundaries. The dynamic
policy delegates residency to the adapter and drains pending layer transfers
before moving to the next phase, including on exceptional exits.
"""

from functools import wraps

import torch

from memory.policies.memory_phase_controller import MemoryPhase


_FLAGS = {
    "text_encoder": "text_encoder_cpu_offload",
    "transformer": "dit_cpu_offload",
    "vae": "vae_cpu_offload",
}


def offload_component(component_name: str, *, phase: MemoryPhase | None = None):
    flag = _FLAGS[component_name]
    if phase is None:
        phase = {
            "text_encoder": MemoryPhase.TEXT_ENCODE,
            "transformer": MemoryPhase.DENOISE,
        }.get(component_name)

    def decorate(forward):
        @wraps(forward)
        def wrapped(self, batch, server_args):
            policy = server_args.resolve_resource_policy()
            if policy.dynamic_offload:
                controller = batch.extra.get("memory_phase_controller")
                if controller is None or phase is None:
                    raise RuntimeError("dynamic offload requires a memory phase controller")
                name = component_name
                if name == "vae":
                    name = "vae.encoder" if phase is MemoryPhase.VAE_ENCODE else "vae.decoder"
                adapter = controller.adapter
                try:
                    controller.enter(
                        phase, component_name=name,
                        window_key=(int(batch.extra.get("object_index", -1)),
                                    int(batch.extra.get("window_index", -1))),
                    )
                    if adapter.active_component_name != name:
                        adapter.onload_remainder(name)
                    return forward(self, batch, server_args)
                finally:
                    try:
                        if controller.active_phase is phase:
                            controller.exit(phase)
                    finally:
                        adapter.offload_remainder(name)
            if not getattr(policy, flag):
                return forward(self, batch, server_args)
            module = batch.modules[component_name]
            # Include onload in the try block: a partially failed transfer must
            # also return weights to CPU before the next request.
            try:
                module.to(device=torch.device(server_args.device))
                return forward(self, batch, server_args)
            finally:
                module.to(device=torch.device("cpu"))

        return wrapped

    return decorate
