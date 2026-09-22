"""Compatibility alias for models.adapters.ltx095.preprocess; no implementation lives here."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("models.adapters.ltx095.preprocess")
