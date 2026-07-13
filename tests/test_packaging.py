import shutil
import subprocess
import tarfile
import tomllib
from pathlib import Path
from uuid import uuid4

import pytest


PROJECT_ROOT = Path(__file__).parents[1]


def test_local_logfire_state_is_excluded_from_source_distributions() -> None:
    configuration = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    build_excludes = configuration["tool"]["hatch"]["build"]["exclude"]
    git_ignores = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "/.logfire" in build_excludes
    assert ".logfire/" in git_ignores


def test_source_distribution_excludes_nested_ignored_local_state(
    tmp_path: Path,
) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("source-distribution regression requires uv")

    local_root = PROJECT_ROOT / f"relative-home-sdist-test-{uuid4().hex}"
    local_state = local_root / ".cache" / "uv"
    sentinel = local_state / "ghostwheel-sdist-sentinel"

    try:
        local_state.mkdir(parents=True)
        (local_state / ".gitignore").write_text("*\n", encoding="utf-8")
        sentinel.write_text("must not be packaged\n", encoding="utf-8")
        subprocess.run(
            [
                uv,
                "build",
                "--sdist",
                "--out-dir",
                str(tmp_path),
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        shutil.rmtree(local_root, ignore_errors=True)

    archives = list(tmp_path.glob("*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0], "r:gz") as archive:
        archived_paths = [member.name for member in archive.getmembers()]

    assert not any(local_root.name in path for path in archived_paths)


def test_source_distribution_has_an_explicit_content_boundary() -> None:
    configuration = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert configuration["tool"]["hatch"]["build"]["targets"]["sdist"][
        "only-include"
    ] == [
        ".env.example",
        ".github",
        ".gitignore",
        ".python-version",
        "AGENTS.md",
        "ARCHITECTURE.md",
        "README.md",
        "pyproject.toml",
        "src",
        "tests",
        "uv.lock",
    ]


def test_project_uses_one_terminal_ui_stack() -> None:
    configuration = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = configuration["project"]["dependencies"]

    assert any(dependency.startswith("prompt-toolkit") for dependency in dependencies)
    assert any(dependency.startswith("rich") for dependency in dependencies)
    assert not any(dependency.startswith("textual") for dependency in dependencies)
