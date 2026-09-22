"""Video-level runtime for the erase pipeline."""

from pipelines.runtime.context import prepare_runtime_context
from pipelines.runtime.contracts import EraseRuntimeContext

__all__ = ("EraseRuntimeContext", "prepare_runtime_context")
