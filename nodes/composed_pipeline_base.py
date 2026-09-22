"""Compatibility alias for pipelines.base; no implementation lives here."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("pipelines.base")
