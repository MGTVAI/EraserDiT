"""Parallel primitives stay independent of model-specific adapters."""

import importlib
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class ParallelCompatibilityTests(unittest.TestCase):
    def test_shared_parallel_and_layers_do_not_load_models(self):
        result = subprocess.run(
            [sys.executable, "-c", """
import sys
import parallel
assert 'layers' not in sys.modules
import layers.attention.sequence_parallel
for name in ('models', 'loader', 'nodes', 'pipelines', 'entrypoints'):
    assert name not in sys.modules, name
"""], cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_adapter_aliases_in_both_import_orders(self):
        script = """
import importlib
import sys
for name in ('cfg', 'mesh', 'vae', 'vae_spatial'):
    old = 'parallel.eraserdit_' + name
    new = 'models.adapters.eraserdit.' + name
    order = (old, new) if sys.argv[1] == 'legacy' else (new, old)
    for module in order:
        importlib.import_module(module)
    assert sys.modules[old] is sys.modules[new], name
"""
        for order in ("legacy", "canonical"):
            with self.subTest(order=order):
                result = subprocess.run(
                    [sys.executable, "-c", script, order], cwd=ROOT,
                    capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_legacy_patch_reaches_vae_dispatch(self):
        old = importlib.import_module("parallel.eraserdit_vae")
        new = importlib.import_module("models.adapters.eraserdit.vae")
        marker = object()
        with patch.object(old, "resolve_mesh", marker):
            self.assertIs(new.tiled_vae.__globals__["resolve_mesh"], marker)


if __name__ == "__main__":
    unittest.main()
