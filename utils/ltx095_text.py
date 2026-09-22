"""Compatibility alias for models.text_encoders.ltx095_text; no implementation lives here."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("models.text_encoders.ltx095_text")
