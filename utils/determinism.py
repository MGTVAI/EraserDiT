"""Deterministic numerical profile.

Mirrors the frozen baseline's third external modification (``vibe/plan.md`` M0):
it pins kernel selection and reduction order only -- no algorithmic or sampling
parameter is touched.  Enabled by default; performance measurement runs with
``ERASERDIT_DETERMINISTIC=0``.

M1b equivalence runs must use this profile on **both** sides; the baseline is
byte-reproducible under it, so any kernel difference in the new architecture
shows up directly in the non-erase-region metrics.
"""

from __future__ import annotations

import os
from typing import Any

import torch

DETERMINISTIC = os.environ.get("ERASERDIT_DETERMINISTIC", "1") != "0"

# cuBLAS reads the workspace configuration when the CUDA context is created, so
# it has to be set before any CUDA work happens.
if DETERMINISTIC:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

_APPLIED = False


def enable_deterministic_mode() -> dict[str, Any]:
    """Apply the deterministic switches and return what actually took effect."""
    global _APPLIED
    settings: dict[str, Any] = {
        "enabled": DETERMINISTIC,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if not DETERMINISTIC:
        return settings
    if not _APPLIED:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        _APPLIED = True
    settings.update(
        use_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
    )
    return settings


__all__ = ["DETERMINISTIC", "enable_deterministic_mode"]
