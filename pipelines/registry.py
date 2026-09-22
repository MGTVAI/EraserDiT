"""Pipeline registry.

Mirrors ``models/registry.py``: modules under ``pipelines/`` declare
``EntryClass = <PipelineClass>`` and are discovered by walking the package, so
adding a model never requires editing the service or session layer.
"""

from __future__ import annotations

import ast
import importlib
import os
from functools import lru_cache
from typing import Any

from utils.logging_utils import init_logger

logger = init_logger(__name__)

PIPELINES_PATH = os.path.dirname(__file__)
DEFAULT_PIPELINE = "EraserDiTErasePipeline"


@lru_cache(maxsize=None)
def _discover_pipelines() -> dict[str, tuple[str, str]]:
    discovered: dict[str, tuple[str, str]] = {}
    for root, _dirs, files in os.walk(PIPELINES_PATH):
        for filename in sorted(files):
            if not filename.endswith(".py") or filename.startswith("__"):
                continue
            path = os.path.join(root, filename)
            rel_path = os.path.relpath(path, PIPELINES_PATH)
            module_name = "pipelines." + rel_path[:-3].replace(os.sep, ".")
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    tree = ast.parse(fh.read(), filename=filename)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Skipping pipeline parse for %s: %s", path, exc)
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "EntryClass":
                        value = node.value
                        if isinstance(value, ast.Name):
                            discovered[value.id] = (module_name, value.id)
                        elif isinstance(value, ast.Constant) and isinstance(value.value, str):
                            discovered[value.value] = (module_name, value.value)
    return discovered


class _PipelineRegistry:
    def resolve(self, name: str | None) -> tuple[type[Any], str]:
        requested = name or DEFAULT_PIPELINE
        discovered = _discover_pipelines()
        if requested not in discovered:
            raise ValueError(
                f"Unsupported pipeline: {requested}. "
                f"Known pipelines: {sorted(discovered)}"
            )
        module_name, class_name = discovered[requested]
        module = importlib.import_module(module_name)
        return getattr(module, class_name), requested

    def names(self) -> list[str]:
        return sorted(_discover_pipelines())


PipelineRegistry = _PipelineRegistry()

__all__ = ["PipelineRegistry", "DEFAULT_PIPELINE"]
