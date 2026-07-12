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
