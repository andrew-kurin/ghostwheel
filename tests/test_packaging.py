import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]


def test_local_logfire_state_is_excluded_from_source_distributions() -> None:
    configuration = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    build_excludes = configuration["tool"]["hatch"]["build"]["exclude"]
    git_ignores = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "/.logfire" in build_excludes
    assert ".logfire/" in git_ignores


def test_project_uses_one_terminal_ui_stack() -> None:
    configuration = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = configuration["project"]["dependencies"]

    assert any(dependency.startswith("prompt-toolkit") for dependency in dependencies)
    assert any(dependency.startswith("rich") for dependency in dependencies)
    assert not any(dependency.startswith("textual") for dependency in dependencies)
