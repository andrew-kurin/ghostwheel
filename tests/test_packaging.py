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
    ignored_python = local_state / "ignored-state.py"
    external_target = tmp_path / "external-secret.py"
    external_symlink = local_state / "external-secret.py"
    ignored_marker = uuid4().hex.encode()
    external_marker = uuid4().hex.encode()
    sdist_directory = tmp_path / "sdist"
    wheel_directory = tmp_path / "wheel"

    try:
        local_state.mkdir(parents=True)
        (local_state / ".gitignore").write_text("*\n", encoding="utf-8")
        ignored_python.write_bytes(ignored_marker)
        external_target.write_bytes(external_marker)
        external_symlink.symlink_to(external_target)
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
        source_members = archive.getmembers()
        source_paths = [member.name for member in source_members]
        source_contents = b"".join(
            extracted.read()
            for member in source_members
            if member.isfile()
            and (extracted := archive.extractfile(member)) is not None
        )
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_paths = archive.namelist()
        wheel_contents = b"".join(
            archive.read(path) for path in wheel_paths if not path.endswith("/")
        )

    assert not any(local_root.name in path for path in source_paths)
    assert not any(local_root.name in path for path in wheel_paths)
    assert ignored_marker not in source_contents
    assert ignored_marker not in wheel_contents
    assert external_marker not in source_contents
    assert external_marker not in wheel_contents
    assert all(
        path.startswith("ghostwheel/") or ".dist-info/" in path for path in wheel_paths
    )


def test_release_manifest_uses_only_exact_regular_files() -> None:
    configuration = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    build = configuration["tool"]["hatch"]["build"]
    manifest = build["include"]
    assert manifest
    for entry in manifest:
        assert entry.startswith("/")
        assert not set("*?[") & set(entry)
        source = PROJECT_ROOT / entry.removeprefix("/")
        assert source.is_file()
        assert not source.is_symlink()

    wheel = build["targets"]["wheel"]
    assert wheel["sources"] == ["src"]
    assert wheel["only-packages"] is True
    assert wheel["exclude"] == ["/tests/"]


def test_release_manifest_matches_tracked_files() -> None:
    if not (PROJECT_ROOT / ".git").exists():
        pytest.skip("release-manifest regression requires a Git worktree")
    git = shutil.which("git")
    if git is None:
        pytest.skip("release-manifest regression requires git")

    tracked = subprocess.run(
        [git, "-C", str(PROJECT_ROOT), "ls-files"],
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        pytest.skip("release-manifest regression requires a Git worktree")

    configuration = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    manifest = {
        entry.removeprefix("/")
        for entry in configuration["tool"]["hatch"]["build"]["include"]
    }
    tracked_files = set(tracked.stdout.splitlines())

    assert manifest == tracked_files


def test_project_uses_one_terminal_ui_stack() -> None:
    configuration = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = configuration["project"]["dependencies"]

    assert any(dependency.startswith("prompt-toolkit") for dependency in dependencies)
    assert any(dependency.startswith("rich") for dependency in dependencies)
    assert not any(dependency.startswith("textual") for dependency in dependencies)
