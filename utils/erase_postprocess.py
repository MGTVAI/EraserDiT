"""Compatibility alias for models.adapters.ltx095.postprocess; no implementation lives here."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("models.adapters.ltx095.postprocess")
