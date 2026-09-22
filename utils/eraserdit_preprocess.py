"""Compatibility alias for models.adapters.eraserdit.preprocess; no implementation lives here."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("models.adapters.eraserdit.preprocess")
