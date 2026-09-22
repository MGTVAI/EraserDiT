"""Memory management package.

Three responsibility boundaries:

* ``adapters/``  – per-model residency adapters (what may be offloaded, when).
* ``backends/``  – the actual move/offload machinery (extents, offload tools, storage).
* ``policies/``  – residency and stage-coordination policies.

Adapted from EraserDiT_origin ``utils_inference/memory/`` and ``runtime/resource/``.
"""

from memory.adapters.model_memory_adapter import (  # noqa: F401
    ComponentResidencyDecision,
    MemoryRegistrationSummary,
    ModelMemoryAdapter,
)
from memory.backends.flexible_memory_device_state import FlexibleMemoryDeviceState  # noqa: F401
from memory.backends.flexible_memory_states import FlexibleMemoryState  # noqa: F401
from memory.backends.flexible_module_extent import FlexibleModuleExtent  # noqa: F401
from memory.backends.flexible_module_extent_cuda_async import (  # noqa: F401
    FlexibleModuleExtentCudaAsync,
)
from memory.policies.memory_phase_controller import (  # noqa: F401
    MemoryPhase,
    MemoryPhaseController,
)

__all__ = (
    "ComponentResidencyDecision",
    "FlexibleMemoryDeviceState",
    "FlexibleMemoryState",
    "FlexibleModuleExtent",
    "FlexibleModuleExtentCudaAsync",
    "MemoryPhase",
    "MemoryPhaseController",
    "MemoryRegistrationSummary",
    "ModelMemoryAdapter",
)
