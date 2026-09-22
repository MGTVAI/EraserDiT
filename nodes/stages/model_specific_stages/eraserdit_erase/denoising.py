"""Compatibility alias for pipelines.stages.eraserdit_erase.denoising; no implementation lives here."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("pipelines.stages.eraserdit_erase.denoising")
