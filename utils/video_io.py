"""Compatibility alias for media.video_io; no implementation lives here."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("media.video_io")
