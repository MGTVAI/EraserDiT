"""Compatibility alias for parallel.runtime; no implementation lives here."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("parallel.runtime")
