"""Legacy stage paths resolve to one implementation, including module globals."""

import importlib
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class StageCompatibilityTests(unittest.TestCase):
    def test_stage_import_identity_in_both_orders(self):
        # Fresh processes expose import-order bugs hidden by unittest discovery.
        script = """
import importlib
from pathlib import Path
import sys

legacy_root = Path('nodes/stages/model_specific_stages')
for path in sorted(legacy_root.rglob('*.py')):
    legacy = '.'.join(path.with_suffix('').parts).removesuffix('.__init__')
    canonical = legacy.replace('nodes.stages.model_specific_stages', 'pipelines.stages')
    order = (legacy, canonical) if sys.argv[1] == 'legacy' else (canonical, legacy)
    for name in order:
        importlib.import_module(name)
    old, new = sys.modules[legacy], sys.modules[canonical]
    if path.name == '__init__.py':
        assert old.__all__ == new.__all__, legacy
        for name in new.__all__:
            assert getattr(old, name) is getattr(new, name), (legacy, name)
    else:
        assert old is new, legacy
"""
        for order in ("legacy", "canonical"):
            with self.subTest(order=order):
                result = subprocess.run(
                    [sys.executable, "-c", script, order], cwd=ROOT,
                    capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_legacy_module_patch_reaches_stage_function_globals(self):
        legacy = importlib.import_module(
            "nodes.stages.model_specific_stages.eraserdit_erase.denoising"
        )
        canonical = importlib.import_module("pipelines.stages.eraserdit_erase.denoising")
        # A forwarding class export alone would leave patches in the wrong module.
        marker = object()
        with patch.object(legacy, "validate_cfg_parallel", marker):
            self.assertIs(canonical.EraserDiTEraseDenoisingStage.__init__.__globals__[
                "validate_cfg_parallel"
            ], marker)

    def test_pipeline_base_compatibility(self):
        from nodes import ComposedPipelineBase as exported
        from pipelines.base import ComposedPipelineBase
        from pipelines.eraserdit_erase_pipeline import EraserDiTErasePipeline
        from pipelines.ltx_095_erase_pipeline import LTX095ErasePipeline

        old = importlib.import_module("nodes.composed_pipeline_base")
        new = importlib.import_module("pipelines.base")
        self.assertIs(old, new)
        self.assertIs(exported, ComposedPipelineBase)
        self.assertTrue(issubclass(EraserDiTErasePipeline, exported))
        self.assertTrue(issubclass(LTX095ErasePipeline, exported))


if __name__ == "__main__":
    unittest.main()
