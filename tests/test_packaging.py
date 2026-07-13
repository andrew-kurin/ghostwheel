import shutil
import subprocess
import tarfile
import tomllib
import zipfile
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


def test_release_artifacts_exclude_nested_ignored_local_state(
    tmp_path: Path,
) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("release-artifact regression requires uv")

    local_root = (
        PROJECT_ROOT / "src" / "ghostwheel" / (f"local-state-build-test-{uuid4().hex}")
    )
    local_state = local_root / ".cache"
    sentinel = local_state / "ghostwheel-sdist-sentinel"
    sdist_directory = tmp_path / "sdist"
    wheel_directory = tmp_path / "wheel"

    try:
        local_state.mkdir(parents=True)
        (local_state / ".gitignore").write_text("*\n", encoding="utf-8")
        sentinel.write_text("must not be packaged\n", encoding="utf-8")
        for artifact, output_directory in (
            ("--sdist", sdist_directory),
            ("--wheel", wheel_directory),
        ):
            subprocess.run(
                [uv, "build", artifact, "--out-dir", str(output_directory)],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
    finally:
        shutil.rmtree(local_root, ignore_errors=True)

    source_archives = list(sdist_directory.glob("*.tar.gz"))
    wheels = list(wheel_directory.glob("*.whl"))
    assert len(source_archives) == 1
    assert len(wheels) == 1
    with tarfile.open(source_archives[0], "r:gz") as archive:
        source_paths = [member.name for member in archive.getmembers()]
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_paths = archive.namelist()

    assert not any(local_root.name in path for path in source_paths)
    assert not any(local_root.name in path for path in wheel_paths)


def test_release_artifacts_have_explicit_content_boundaries() -> None:
    configuration = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    targets = configuration["tool"]["hatch"]["build"]["targets"]
    assert targets["sdist"]["include"] == [
        "/.env.example",
        "/.github/workflows/*.yml",
        "/.gitignore",
        "/.python-version",
        "/AGENTS.md",
        "/ARCHITECTURE.md",
        "/README.md",
        "/pyproject.toml",
        "/src/ghostwheel/**/*.py",
        "/tests/**/*.py",
        "/uv.lock",
    ]
    assert targets["wheel"] == {
        "include": ["/src/ghostwheel/**/*.py"],
        "sources": ["src"],
    }


def test_project_uses_one_terminal_ui_stack() -> None:
    configuration = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = configuration["project"]["dependencies"]

    assert any(dependency.startswith("prompt-toolkit") for dependency in dependencies)
    assert any(dependency.startswith("rich") for dependency in dependencies)
    assert not any(dependency.startswith("textual") for dependency in dependencies)
