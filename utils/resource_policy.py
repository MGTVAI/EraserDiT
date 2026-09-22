"""Compatibility exports for resource configuration and tensor operations."""

from config.resource_policy import (
    RuntimeResourcePolicy,
    normalize_resource_policy_name,
    resolve_runtime_resource_policy,
)
from memory.tensor_ops import (
    maybe_pin_tensor,
    module_device,
    module_dtype,
    move_module_to_device,
    pin_module_cpu_memory,
)
