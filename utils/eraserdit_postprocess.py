"""Compatibility alias for models.adapters.eraserdit.postprocess; no implementation lives here."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("models.adapters.eraserdit.postprocess")
