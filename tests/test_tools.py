import asyncio
import copy
import os
from pathlib import Path
import shlex
import subprocess
import time
from types import SimpleNamespace

import pytest
import ghostwheel.tools.workspace as workspace_module

from ghostwheel.tools.bash import bash
from ghostwheel.tools.catalog import DEFAULT_TOOL_CATALOG, ToolProfile
from ghostwheel.tools.command import CommandResult
from ghostwheel.tools.deps import ToolDeps, ToolLimits
from ghostwheel.tools.filesystem import FileKind, ls, read
from ghostwheel.tools.output import OutputBudget, truncate_utf8
from ghostwheel.tools.search import grep
from ghostwheel.tools.workspace import Workspace


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


def run_bash(ctx: SimpleNamespace, command: str):
    return asyncio.run(bash(ctx, command))


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

    assert result.path == "README.md"
    assert result.content == "   1 | hello\n   2 | world"
    assert result.line_count == 2
    assert result.truncated is False


def test_relative_filesystem_roots_are_resolved_from_tool_cwd(tmp_path: Path) -> None:
    cwd = (tmp_path / "repo").resolve()
    shared = cwd / "shared"
    shared.mkdir(parents=True)
    (shared / "value.txt").write_text("value", encoding="utf-8")
    ctx = SimpleNamespace(deps=ToolDeps(cwd=cwd, filesystem_roots=["shared"]))

    result = read(ctx, str(shared / "value.txt"))

    assert result.path == "shared/value.txt"


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


def test_filesystem_tools_never_traverse_symlinked_parent_directories(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "repo").resolve()
    outside = (tmp_path / "outside").resolve()
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET", encoding="utf-8")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    ctx = tool_ctx(root)

    with pytest.raises(OSError):
        read(ctx, "linked/secret.txt")
    with pytest.raises(OSError):
        ls(ctx, "linked")

    assert grep(ctx, "SECRET").matches == []


def test_read_is_safe_when_parent_is_swapped_after_descriptor_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "repo").resolve()
    safe = root / "safe"
    outside = (tmp_path / "outside").resolve()
    safe.mkdir(parents=True)
    outside.mkdir()
    (safe / "value.txt").write_text("SAFE", encoding="utf-8")
    (outside / "value.txt").write_text("SECRET", encoding="utf-8")
    original_open = os.open
    swapped = False

    def racing_open(path: object, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if path == "value.txt" and not swapped:
            swapped = True
            safe.rename(root / "original-safe")
            safe.symlink_to(outside, target_is_directory=True)
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(workspace_module.os, "open", racing_open)

    result = read(tool_ctx(root), "safe/value.txt")

    assert result.content.endswith("SAFE")
    assert "SECRET" not in result.content


def test_workspace_pins_allowed_root_across_ancestor_swap(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    root = parent / "repo"
    outside_parent = tmp_path / "outside-parent"
    outside_root = outside_parent / "repo"
    root.mkdir(parents=True)
    outside_root.mkdir(parents=True)
    (root / "value.txt").write_text("SAFE", encoding="utf-8")
    (outside_root / "value.txt").write_text("SECRET", encoding="utf-8")
    ctx = tool_ctx(root)
    parent.rename(tmp_path / "original-parent")
    parent.symlink_to(outside_parent, target_is_directory=True)

    result = read(ctx, "value.txt")

    assert result.content.endswith("SAFE")
    assert "SECRET" not in result.content


def test_workspace_initialization_rejects_ancestor_symlink_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "allowed-parent-race"
    root = parent / "repo"
    outside_parent = tmp_path / "outside-parent-race"
    (outside_parent / "repo").mkdir(parents=True)
    root.mkdir(parents=True)
    original_open = os.open
    swapped = False

    def racing_open(path: object, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if not swapped and path == parent.name:
            swapped = True
            parent.rename(tmp_path / "original-allowed-parent")
            parent.symlink_to(outside_parent, target_is_directory=True)
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(workspace_module.os, "open", racing_open)

    with pytest.raises(OSError):
        tool_ctx(root)


def test_closed_workspace_never_reuses_or_recloses_descriptor_numbers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "value.txt").write_text("SAFE", encoding="utf-8")
    workspace = Workspace(root)
    workspace.close()
    outside_fd = os.open(outside, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(RuntimeError, match="Workspace is closed"):
            with workspace.open_file("value.txt"):
                pass
        workspace.close()
        os.fstat(outside_fd)
    finally:
        os.close(outside_fd)


def test_workspace_copies_share_one_descriptor_owner(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    workspace = Workspace(root)

    assert copy.copy(workspace) is workspace
    assert copy.deepcopy(workspace) is workspace


def test_read_rejects_non_positive_max_output_bytes(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    (root / "README.md").write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="max_output_bytes must be positive"):
        read(tool_ctx(root, max_output_bytes=0), "README.md")


@pytest.mark.parametrize("field", ["regex_timeout_seconds", "bash_timeout_seconds"])
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_runtime_timeouts_must_be_finite(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        ToolLimits(**{field: value})


def test_output_budget_normalizes_surrogateescaped_names() -> None:
    value = "bad\udcff.py"
    normalized, changed = truncate_utf8(value, 100)
    budget = OutputBudget(100)

    assert normalized == "bad\\udcff.py"
    assert changed is True
    assert budget.consume(value) is True


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


def test_grep_returns_readable_absolute_paths_for_allowed_roots_outside_cwd(
    tmp_path: Path,
) -> None:
    cwd = (tmp_path / "repo").resolve()
    outside_root = (tmp_path / "shared").resolve()
    cwd.mkdir()
    outside_root.mkdir()
    (outside_root / "shared.py").write_text("needle\n", encoding="utf-8")
    ctx = tool_ctx(cwd, allowed_roots=[cwd, outside_root])

    result = grep(ctx, "needle", path=str(outside_root), file_glob="*.py")

    assert [(match.file, match.line, match.text) for match in result.matches] == [
        (str(outside_root / "shared.py"), 1, "needle")
    ]
    assert read(ctx, result.matches[0].file).path == str(outside_root / "shared.py")


def test_grep_skips_symlinked_files_outside_allowed_roots(tmp_path: Path) -> None:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    outside = (tmp_path / "secret.txt").resolve()
    outside.write_text("needle\n", encoding="utf-8")
    (root / "link.txt").symlink_to(outside)

    result = grep(tool_ctx(root), "needle", file_glob="*.txt")

    assert result.matches == []
    assert result.files_searched == 0


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


def test_ls_bounds_scanning_of_filtered_hidden_entries(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f".hidden-{index}").write_text("", encoding="utf-8")

    result = ls(tool_ctx(tmp_path, max_directory_scan_entries=2))

    assert result.entries == []
    assert result.truncated is True


def test_ls_uses_typed_kinds_and_configured_entry_limit(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")

    result = ls(tool_ctx(tmp_path, max_entries=1))

    assert len(result.entries) == 1
    assert result.entries[0].type is FileKind.FILE
    assert result.truncated is True


def test_read_caps_numbered_content_to_output_budget(tmp_path: Path) -> None:
    (tmp_path / "many-lines.txt").write_text("a\nb\nc\n", encoding="utf-8")

    result = read(tool_ctx(tmp_path, max_output_bytes=10), "many-lines.txt")

    assert len(result.content.encode("utf-8")) <= 10
    assert result.truncated is True


def test_grep_caps_match_text_to_output_budget(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text("needle " + "x" * 1_000, encoding="utf-8")

    result = grep(
        tool_ctx(tmp_path, max_output_bytes=40),
        "needle",
        file_glob="*.txt",
    )

    assert result.truncated is True
    assert result.matches
    model_text = (
        f"{result.matches[0].file}:{result.matches[0].line}:{result.matches[0].text}"
    )
    assert len(model_text.encode("utf-8")) <= 40


def test_grep_skips_files_over_the_input_limit(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text("needle", encoding="utf-8")

    result = grep(
        tool_ctx(tmp_path, max_search_file_bytes=5),
        "needle",
        file_glob="*.txt",
    )

    assert result.matches == []
    assert result.files_skipped == 1
    assert result.truncated is True


def test_grep_reports_inaccessible_candidates_as_incomplete(tmp_path: Path) -> None:
    blocked = tmp_path / "blocked.txt"
    blocked.write_text("needle", encoding="utf-8")
    blocked.chmod(0)
    try:
        result = grep(tool_ctx(tmp_path), "needle", file_glob="*.txt")
        direct_result = grep(tool_ctx(tmp_path), "needle", path="blocked.txt")
    finally:
        blocked.chmod(0o600)

    assert result.matches == []
    assert result.files_skipped == 1
    assert result.truncated is True
    assert direct_result.files_skipped == 1
    assert direct_result.truncated is True


def test_grep_replacement_decoding_does_not_fabricate_matches(tmp_path: Path) -> None:
    (tmp_path / "invalid.txt").write_bytes(b"need\xffle\n")

    result = grep(tool_ctx(tmp_path), "needle", file_glob="*.txt")

    assert result.matches == []


def test_grep_preserves_recursive_glob_semantics(tmp_path: Path) -> None:
    (tmp_path / "src" / "nested").mkdir(parents=True)
    (tmp_path / "a.py").write_text("needle", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("needle", encoding="utf-8")
    (tmp_path / "src" / "nested" / "c.py").write_text("needle", encoding="utf-8")

    all_python = grep(tool_ctx(tmp_path), "needle", file_glob="**/*.py")
    src_python = grep(tool_ctx(tmp_path), "needle", file_glob="src/**/*.py")

    assert {match.file for match in all_python.matches} == {
        "a.py",
        "src/b.py",
        "src/nested/c.py",
    }
    assert {match.file for match in src_python.matches} == {
        "src/b.py",
        "src/nested/c.py",
    }


def test_grep_bounds_traversal_even_when_files_do_not_match(tmp_path: Path) -> None:
    for index in range(4):
        (tmp_path / f"directory-{index}").mkdir()

    result = grep(
        tool_ctx(tmp_path, max_search_files=2),
        "needle",
        file_glob="*.nomatch",
    )

    assert result.truncated is True


def test_grep_does_not_treat_workspace_ancestors_as_noise(tmp_path: Path) -> None:
    root = tmp_path / ".venv" / "project"
    root.mkdir(parents=True)
    (root / "app.py").write_text("needle", encoding="utf-8")

    result = grep(tool_ctx(root), "needle", file_glob="*.py")

    assert [match.file for match in result.matches] == ["app.py"]


def test_grep_times_out_pathological_regular_expressions(tmp_path: Path) -> None:
    (tmp_path / "pathological.txt").write_text("a" * 50_000 + "!", encoding="utf-8")

    with pytest.raises(TimeoutError, match="per-line timeout"):
        grep(
            tool_ctx(tmp_path, regex_timeout_seconds=0.000001),
            "(a+)+$",
            file_glob="*.txt",
        )


def test_bash_dry_run_does_not_execute_command(tmp_path: Path) -> None:
    target = tmp_path / "created.txt"

    result = run_bash(tool_ctx(tmp_path, dry_run=True), f"touch {target}")

    assert result.exit_code is None
    assert result.stderr == "Dry run: command was not executed."
    assert not target.exists()


def test_bash_truncates_combined_output_to_max_bytes(tmp_path: Path) -> None:
    result = run_bash(
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
    result = run_bash(
        tool_ctx(tmp_path, max_output_bytes=10),
        "printf 'abcdefghijklmnopqrst'; printf 'uvwxyz' >&2",
    )

    assert result.truncated is True
    assert len((result.stdout + result.stderr).encode()) <= 10
    assert result.stdout == "abcde"
    assert result.stderr == "uvwxy"


def test_bash_decodes_invalid_utf8_without_failing(tmp_path: Path) -> None:
    result = run_bash(tool_ctx(tmp_path), r"printf '\377'")

    assert result.exit_code == 0
    assert result.stdout == "�"


def test_bash_timeout_terminates_the_process_group(tmp_path: Path) -> None:
    result = run_bash(
        tool_ctx(tmp_path, bash_timeout_seconds=0.1),
        "sleep 10 & child=$!; echo $child; wait $child",
    )

    assert result.timed_out is True
    child_pid = int(result.stdout.strip())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(child_pid)],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not state or state.startswith("Z"):
            break
        time.sleep(0.02)
    else:
        os.kill(child_pid, 9)
        pytest.fail(f"child process {child_pid} survived command timeout")


def test_detached_child_cannot_stall_timeout_cleanup(tmp_path: Path) -> None:
    started = time.monotonic()
    result = run_bash(
        tool_ctx(tmp_path, bash_timeout_seconds=0.1),
        (
            "python -c 'import os,time; print(os.getpid(), flush=True); "
            "os.setsid(); time.sleep(2)' & wait"
        ),
    )
    elapsed = time.monotonic() - started

    assert result.timed_out is True
    assert elapsed < 1.0
    # A process that deliberately leaves the owned process group is outside the
    # default runner's guarantee; clean up the adversarial fixture explicitly.
    if result.stdout.strip():
        child_pid = int(result.stdout.strip())
        try:
            os.kill(child_pid, 9)
        except ProcessLookupError:
            pass


def test_bash_cancellation_terminates_owned_process_group(tmp_path: Path) -> None:
    pid_file = tmp_path / "shell.pid"

    async def cancel_running_command() -> None:
        task = asyncio.create_task(
            bash(
                tool_ctx(tmp_path),
                f"echo $$ > {shlex.quote(str(pid_file))}; sleep 10",
            )
        )
        deadline = asyncio.get_running_loop().time() + 2
        while not pid_file.exists():
            if asyncio.get_running_loop().time() >= deadline:
                task.cancel()
                raise AssertionError("command did not start")
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_running_command())

    shell_pid = int(pid_file.read_text(encoding="utf-8"))
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(shell_pid)],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert not state or state.startswith("Z")


def test_repeated_cancellation_cannot_interrupt_process_cleanup(tmp_path: Path) -> None:
    shell_file = tmp_path / "shell.pid"
    child_file = tmp_path / "child.pid"

    async def cancel_repeatedly() -> None:
        command = (
            f"echo $$ > {shlex.quote(str(shell_file))}; trap '' TERM; "
            f"sleep 10 & echo $! > {shlex.quote(str(child_file))}; wait"
        )
        task = asyncio.create_task(bash(tool_ctx(tmp_path), command))
        deadline = asyncio.get_running_loop().time() + 2
        while not shell_file.exists() or not child_file.exists():
            if asyncio.get_running_loop().time() >= deadline:
                task.cancel()
                raise AssertionError("command did not start")
            await asyncio.sleep(0.01)
        loop = asyncio.get_running_loop()
        task.cancel()
        loop.call_later(0.01, task.cancel)
        loop.call_later(0.02, task.cancel)
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_repeatedly())

    for pid_file in (shell_file, child_file):
        pid = int(pid_file.read_text(encoding="utf-8"))
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert not state or state.startswith("Z")


def test_bash_uses_injected_command_runner(tmp_path: Path) -> None:
    class FakeCommandRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Path, int, float]] = []

        async def run(
            self,
            command: str,
            *,
            cwd: Path,
            max_output_bytes: int,
            timeout_seconds: float,
        ) -> CommandResult:
            self.calls.append((command, cwd, max_output_bytes, timeout_seconds))
            return CommandResult(0, "injected", "", False, False)

    runner = FakeCommandRunner()
    ctx = SimpleNamespace(
        deps=ToolDeps(cwd=tmp_path, command_runner=runner, max_output_bytes=123)
    )

    result = run_bash(ctx, "status")

    assert result.stdout == "injected"
    assert runner.calls == [("status", tmp_path.resolve(), 123, 30)]


def test_falsey_command_runner_is_not_replaced(tmp_path: Path) -> None:
    class FalseyRunner:
        def __bool__(self) -> bool:
            return False

        async def run(self, *args: object, **kwargs: object) -> CommandResult:
            return CommandResult(0, "injected", "", False, False)

    runner = FalseyRunner()

    deps = ToolDeps(cwd=tmp_path, command_runner=runner)

    assert deps.command_runner is runner


def test_tool_catalog_exposes_immutable_capability_profiles() -> None:
    assert DEFAULT_TOOL_CATALOG.for_profile(ToolProfile.READ_ONLY) == (
        read,
        ls,
        grep,
    )
    assert DEFAULT_TOOL_CATALOG.for_profile(ToolProfile.FULL)[-1] is bash
