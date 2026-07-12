"""Dependency-direction guards for the runtime application boundary."""

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "ghostwheel"

CANONICAL_RUNTIME_MODULES = (
    "runtime_contracts",
    "history_config",
    "history",
    "controller",
    "event_dispatcher",
    "presentation",
    "pydantic_runner",
    "compaction",
    "review",
    "terminal_composer",
    "terminal_ui",
)

COMPATIBILITY_MODULES = {"agent", "models"}


def _module_tree(module: str) -> ast.Module:
    path = PACKAGE_ROOT / f"{module}.py"
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_modules(tree: ast.Module) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return imported


@pytest.mark.parametrize("module", CANONICAL_RUNTIME_MODULES)
def test_runtime_modules_do_not_depend_on_session_compatibility_facade(
    module: str,
) -> None:
    assert "ghostwheel.session" not in _imported_modules(_module_tree(module))


def test_session_defines_orchestration_and_reexports_canonical_boundaries() -> None:
    tree = _module_tree("session")
    imported = _imported_modules(tree)
    defined_classes = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }

    assert "ghostwheel.runtime_contracts" in imported
    assert "ghostwheel.history" in imported
    assert "ChatSession" in defined_classes
    assert defined_classes.isdisjoint(
        {
            "AgentRunner",
            "ContextCompactor",
            "TurnSucceeded",
            "TurnNoResult",
            "TurnFailed",
            "FailureKind",
            "HistoryPolicy",
            "HistoryState",
            "CompactionPlan",
            "CompactionStats",
        }
    )


def test_internal_modules_do_not_depend_on_compatibility_facades() -> None:
    forbidden = {f"ghostwheel.{module}" for module in COMPATIBILITY_MODULES}
    offenders: dict[str, set[str]] = {}

    for path in PACKAGE_ROOT.glob("*.py"):
        if path.stem in COMPATIBILITY_MODULES:
            continue
        imported = _imported_modules(
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        )
        violations = imported & forbidden
        if violations:
            offenders[path.stem] = violations

    assert offenders == {}


def test_terminal_ui_does_not_depend_on_entrypoint_or_composition_root() -> None:
    imported = _imported_modules(_module_tree("terminal_ui"))

    assert "ghostwheel.cli" not in imported
    assert "ghostwheel.bootstrap" not in imported


def test_legacy_terminal_adapters_are_removed() -> None:
    assert not any(
        (PACKAGE_ROOT / filename).exists()
        for filename in (
            "input_ui.py",
            "keyboard.py",
            "rich_ui.py",
            "textual_composer.py",
            "textual_ui.py",
        )
    )


def test_bootstrap_uses_owned_agent_definitions_not_sdk_private_state() -> None:
    tree = _module_tree("bootstrap")
    private_sdk_attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in {"_function_toolset", "_instructions"}
    }

    assert private_sdk_attributes == set()
    assert "ghostwheel.agent_factory" in _imported_modules(tree)
    assert "ghostwheel.agent" not in _imported_modules(tree)


def test_configuration_does_not_import_the_tool_implementation_graph() -> None:
    imported = _imported_modules(_module_tree("config"))

    assert not any(module.startswith("ghostwheel.tools") for module in imported)
    assert "ghostwheel.tool_config" in imported
