import asyncio
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ghostwheel.tools.bash import bash
from ghostwheel.tools.command import CommandResult
from ghostwheel.tools.deps import ToolDeps

from .support import run_bash, tool_ctx


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
        # Allow a loaded CI runner to fork the child and emit its PID before
        # timeout cleanup; the 10-second sleep remains well beyond this limit.
        tool_ctx(tmp_path, bash_timeout_seconds=1.0),
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
    python = shlex.quote(sys.executable)
    result = run_bash(
        tool_ctx(tmp_path, bash_timeout_seconds=0.1),
        (
            f"{python} -c 'import os,time; print(os.getpid(), flush=True); "
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
