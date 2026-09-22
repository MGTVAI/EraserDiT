"""Enforce the dependency boundaries already migrated by the refactor."""

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "cache", "config", "distributed", "entrypoints", "layers", "loader",
    "memory", "models", "nodes", "parallel", "pipelines", "utils", "media",
)

RESOURCE_POLICY_EXPORTS = """
from config.resource_policy import (
    RuntimeResourcePolicy, normalize_resource_policy_name, resolve_runtime_resource_policy,
)
from memory.tensor_ops import (
    maybe_pin_tensor, module_device, module_dtype, move_module_to_device, pin_module_cpu_memory,
)
"""

UTILITY_ALIASES = {
    "utils/erase_preprocess.py": "models.adapters.ltx095.preprocess",
    "utils/erase_postprocess.py": "models.adapters.ltx095.postprocess",
    "utils/eraserdit_preprocess.py": "models.adapters.eraserdit.preprocess",
    "utils/eraserdit_postprocess.py": "models.adapters.eraserdit.postprocess",

    "utils/video_io.py": "media.video_io",
    "utils/distributed_runtime.py": "parallel.runtime",
    "utils/ltx095_text.py": "models.text_encoders.ltx095_text",
}


def compatibility_target(path):
    relative = path.relative_to(ROOT).as_posix()
    if relative in UTILITY_ALIASES:
        return UTILITY_ALIASES[relative]
    if relative == "nodes/composed_pipeline_base.py":
        return "pipelines.base"
    if relative in {
        f"parallel/eraserdit_{name}.py" for name in ("cfg", "mesh", "vae", "vae_spatial")
    }:
        return "models.adapters.eraserdit." + path.stem.removeprefix("eraserdit_")
    prefix = "nodes/stages/model_specific_stages/"
    if relative.startswith(prefix):
        suffix = relative[len(prefix):-3].replace("/", ".")
        return ("pipelines.stages." + suffix).removesuffix(".__init__")
    return None


def imported_modules(path):
    """Include local imports, relative imports and literal dynamic imports."""
    parts = path.relative_to(ROOT).with_suffix("").parts
    package = ".".join(parts[:-1])
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = importlib.util.resolve_name("." * node.level + module, package)
            yield node.lineno, module
        elif isinstance(node, ast.Call) and node.args:
            fn = node.func
            dynamic_import = (
                isinstance(fn, ast.Name) and fn.id in {"__import__", "import_module"}
            ) or (
                isinstance(fn, ast.Attribute) and fn.attr == "import_module"
            )
            value = node.args[0]
            if dynamic_import and isinstance(value, ast.Constant) and isinstance(value.value, str):
                module = value.value
                if module.startswith("."):
                    module = importlib.util.resolve_name(module, package)
                yield node.lineno, module


class ArchitectureTests(unittest.TestCase):
    def test_migrated_dependency_boundaries(self):
        violations = []
        for source in PACKAGES:
            forbidden = {"entrypoints"} if source != "entrypoints" else set()
            if source == "config":
                forbidden |= set(PACKAGES) - {"config", "utils"}
            if source == "distributed":
                forbidden |= {"parallel", "models", "layers", "loader", "nodes", "pipelines"}
            if source == "nodes":
                forbidden |= {"pipelines", "models", "loader", "cache", "memory"}
            if source == "parallel":
                forbidden |= {"models", "layers", "loader", "nodes", "pipelines", "cache", "memory"}
            if source == "layers":
                forbidden |= {"models", "loader", "nodes", "pipelines", "cache", "memory"}
            if source == "models":
                forbidden |= {"loader", "nodes", "pipelines"}
            if source == "utils":
                forbidden |= set(PACKAGES) - {"utils", "media"}
            if source == "media":
                forbidden |= set(PACKAGES) - {"media"}
            if source == "memory":
                forbidden |= set(PACKAGES) - {"memory", "config", "utils"}
            for path in sorted((ROOT / source).rglob("*.py")):
                compatibility = compatibility_target(path)
                allowed = {compatibility}
                if path == ROOT / "utils/resource_policy.py":
                    allowed = {"config.resource_policy", "memory.tensor_ops"}
                for line, target in imported_modules(path):
                    if target.split(".")[0] in forbidden and target not in allowed:
                        violations.append(f"{path.relative_to(ROOT)}:{line}: {target}")
                    legacy = target == "nodes.composed_pipeline_base" or target.startswith(
                        "nodes.stages.model_specific_stages"
                    ) or target.startswith("parallel.eraserdit_") or target in {
                        "utils.resource_policy", "utils.video_io", "utils.distributed_runtime",
                        "utils.ltx095_text", "utils.erase_preprocess", "utils.erase_postprocess",
                        "utils.eraserdit_preprocess", "utils.eraserdit_postprocess",
                    }
                    if legacy and path != ROOT / "nodes/__init__.py":
                        violations.append(f"{path.relative_to(ROOT)}:{line}: legacy import {target}")
                    if path.is_relative_to(ROOT / "pipelines/runtime") and (
                        target.startswith("pipelines.")
                        and not (target == "pipelines.runtime" or target.startswith("pipelines.runtime."))
                    ):
                        violations.append(f"{path.relative_to(ROOT)}:{line}: runtime imports assembly {target}")
        self.assertEqual(violations, [], "Forbidden dependencies:\n" + "\n".join(violations))

    def test_compatibility_modules_only_forward(self):
        for path in (
            *sorted((ROOT / "nodes").rglob("*.py")),
            *sorted((ROOT / "parallel").rglob("*.py")),
            *sorted((ROOT / "utils").rglob("*.py")),
        ):
            target = compatibility_target(path)
            resource_policy = path == ROOT / "utils/resource_policy.py"
            if target is None and not resource_policy:
                continue
            if resource_policy:
                expected = RESOURCE_POLICY_EXPORTS
            elif path.name == "__init__.py":
                expected = f"from {target} import *\nfrom {target} import __all__"
            else:
                expected = (
                    "import importlib\nimport sys\n"
                    f"sys.modules[__name__] = importlib.import_module('{target}')"
                )
            tree = ast.parse(path.read_text())
            # The only permitted extra statement is the module docstring.
            if ast.get_docstring(tree) is not None:
                tree.body.pop(0)
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertEqual(ast.dump(tree), ast.dump(ast.parse(expected)))

    def test_implementation_package_graph_is_acyclic(self):
        dependencies = {name: set() for name in PACKAGES}
        for source in PACKAGES:
            for path in (ROOT / source).rglob("*.py"):
                # Compatibility facades are checked separately and forbidden to callers.
                if compatibility_target(path) or path == ROOT / "utils/resource_policy.py":
                    continue
                for _, target in imported_modules(path):
                    package = target.split(".")[0]
                    if package in dependencies and package != source:
                        dependencies[source].add(package)
        remaining = dict(dependencies)
        while remaining:
            leaves = {name for name, targets in remaining.items() if not targets & remaining.keys()}
            self.assertTrue(leaves, f"Cyclic implementation dependencies: {remaining}")
            for name in leaves:
                del remaining[name]

    def test_control_import_does_not_load_application_or_models(self):
        result = subprocess.run(
            [sys.executable, "-c", """
import sys
import nodes.control
for name in ('entrypoints', 'pipelines', 'loader', 'models'):
    assert name not in sys.modules, name
"""],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
