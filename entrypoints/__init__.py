"""Local entrypoints for the minimal EraserDiT runtime."""

import os

# Set this before torch (which also imports NumPy). On shared hosts, NumPy's
# MADV_HUGEPAGE allocations can stall in synchronous memory compaction while
# loading full-resolution video/mask arrays. Respect an explicit user override.
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

from utils.logging_utils import globally_suppress_loggers

globally_suppress_loggers(
    "transformers.modeling_utils",
    "diffusers.configuration_utils",
)
