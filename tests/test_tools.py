from pathlib import Path
from types import SimpleNamespace

import pytest

from ghostwheel.tools.bash import bash
from ghostwheel.tools.deps import ToolDeps
from ghostwheel.tools.filesystem import ls, read
from ghostwheel.tools.search import grep


def tool_ctx(root: Path, **overrides: object) -> SimpleNamespace:
    deps_kwargs = {
        "cwd": root,
        "allowed_roots": [root],
        "max_output_bytes": 100_000,
        "bash_timeout_seconds": 30,
        "dry_run": False,
        **overrides,
    }
    return SimpleNamespace(deps=ToolDeps(**deps_kwargs))


def test_read_resolves_relative_paths_against_tool_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / "README.md").write_text("hello\nworld\n", encoding="utf-8")

    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    result = read(tool_ctx(root), "README.md")

    assert result.path == str(root / "README.md")
    assert result.content == "   1 | hello\n   2 | world"
    assert result.line_count == 2
    assert result.truncated is False


def test_filesystem_tools_reject_paths_outside_allowed_roots(tmp_path: Path) -> None:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    ctx = tool_ctx(root)

    with pytest.raises(ValueError, match="outside allowed roots"):
        read(ctx, "../outside.txt")

    with pytest.raises(ValueError, match="outside allowed roots"):
        ls(ctx, "..")


def test_grep_finds_matches_and_skips_noise_directories(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("needle\n", encoding="utf-8")
    (root / ".venv").mkdir()
    (root / ".venv" / "ignored.py").write_text("needle\n", encoding="utf-8")

    result = grep(tool_ctx(root), "needle", file_glob="*.py")

    assert result.files_searched == 1
    assert result.truncated is False
    assert [(match.file, match.line, match.text) for match in result.matches] == [
        ("src/app.py", 1, "needle")
    ]


def test_grep_returns_paths_relative_to_tool_cwd_for_scoped_search(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("needle\n", encoding="utf-8")

    result = grep(tool_ctx(root), "needle", path="src", file_glob="*.py")

    assert [(match.file, match.line, match.text) for match in result.matches] == [
        ("src/app.py", 1, "needle")
    ]


def test_ls_does_not_count_filtered_hidden_entries_toward_limit(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    for index in range(205):
        (root / f".hidden-{index:03d}").write_text("", encoding="utf-8")
    (root / "visible.txt").write_text("", encoding="utf-8")

    result = ls(tool_ctx(root), show_hidden=False)

    assert [entry.name for entry in result.entries] == ["visible.txt"]
    assert result.truncated is False


def test_bash_dry_run_does_not_execute_command(tmp_path: Path) -> None:
    target = tmp_path / "created.txt"

    result = bash(tool_ctx(tmp_path, dry_run=True), f"touch {target}")

    assert result.exit_code is None
    assert result.stderr == "Dry run: command was not executed."
    assert not target.exists()


def test_bash_truncates_combined_output_to_max_bytes(tmp_path: Path) -> None:
    result = bash(
        tool_ctx(tmp_path, max_output_bytes=12),
        "printf 'abcdefghij'; printf 'klmnopqrst' >&2",
    )

    assert result.truncated is True
    assert len((result.stdout + result.stderr).encode()) <= 12
    assert result.stdout == "abcdefghij"
    assert result.stderr == "kl"


def test_bash_truncation_preserves_stderr_when_stdout_exhausts_budget(
    tmp_path: Path,
) -> None:
    result = bash(
        tool_ctx(tmp_path, max_output_bytes=10),
        "printf 'abcdefghijklmnopqrst'; printf 'uvwxyz' >&2",
    )

    assert result.truncated is True
    assert len((result.stdout + result.stderr).encode()) <= 10
    assert result.stdout == "abcde"
    assert result.stderr == "uvwxy"
