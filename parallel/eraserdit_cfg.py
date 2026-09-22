"""Compatibility alias for models.adapters.eraserdit.cfg; no implementation lives here."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("models.adapters.eraserdit.cfg")
