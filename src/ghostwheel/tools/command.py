"""Cancellation-aware local command execution adapter."""

import asyncio
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CommandResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    truncated: bool


class CommandRunner(Protocol):
    async def run(
        self,
        command: str,
        *,
        cwd: Path,
        max_output_bytes: int,
        timeout_seconds: float,
    ) -> CommandResult: ...


class _BoundedBuffer:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.data = bytearray()
        self.truncated = False

    def append(self, chunk: bytes) -> None:
        remaining = self.limit - len(self.data)
        if remaining > 0:
            self.data.extend(chunk[:remaining])
        if len(chunk) > remaining:
            self.truncated = True


async def _drain(
    stream: asyncio.StreamReader,
    buffer: _BoundedBuffer,
) -> None:
    while chunk := await stream.read(64 * 1024):
        buffer.append(chunk)


async def _wait_briefly(
    process: asyncio.subprocess.Process,
    timeout: float,
) -> None:
    try:
        await asyncio.wait_for(process.wait(), timeout)
    except TimeoutError:
        pass


def _signal_group(process: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    if hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, sig)
            return
        except ProcessLookupError, PermissionError:
            return
        except OSError:
            pass
    if process.returncode is not None:
        return
    try:
        process.send_signal(sig)
    except ProcessLookupError:
        pass


async def _terminate_group(process: asyncio.subprocess.Process) -> None:
    _signal_group(process, signal.SIGTERM)
    await _wait_briefly(process, 0.2)
    # The shell may already be reaped while children remain in its group.
    _signal_group(process, signal.SIGKILL)
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    await _wait_briefly(process, 0.2)


async def _finish_readers(tasks: tuple[asyncio.Task[None], ...]) -> None:
    if not tasks:
        return
    done, pending = await asyncio.wait(tasks, timeout=0.2)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    for task in done:
        # Surface unexpected reader failures while ignoring ordinary cancellation.
        if not task.cancelled():
            task.result()


async def _await_uninterruptibly(task: asyncio.Task[None]) -> None:
    """Finish cleanup even if the parent task receives repeated cancellation."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    task.result()


def _close_transport(process: asyncio.subprocess.Process) -> None:
    # asyncio does not expose a public close method on Process. Closing its
    # transport releases captured pipe descriptors even if a deliberately
    # detached descendant kept the write ends open.
    transport = getattr(process, "_transport", None)
    if transport is not None:
        transport.close()


def _truncate_streams(
    stdout: str,
    stderr: str,
    max_bytes: int,
) -> tuple[str, str, bool]:
    stdout_bytes = stdout.encode("utf-8")
    stderr_bytes = stderr.encode("utf-8")
    if len(stdout_bytes) + len(stderr_bytes) <= max_bytes:
        return stdout, stderr, False

    stdout_budget = min(len(stdout_bytes), max_bytes)
    stderr_budget = max_bytes - stdout_budget
    if stderr_bytes and stderr_budget == 0:
        stderr_budget = min(len(stderr_bytes), max_bytes // 2)
        stdout_budget = max_bytes - stderr_budget

    return (
        stdout_bytes[:stdout_budget].decode("utf-8", errors="ignore"),
        stderr_bytes[:stderr_budget].decode("utf-8", errors="ignore"),
        True,
    )


class LocalCommandRunner:
    """Run one shell command inside an owned process group.

    The external sandbox remains the security boundary. Process-group cleanup is
    best effort for descendants that deliberately create a new session, but pipe
    readers are always cancelled so such a child cannot stall the caller.
    """

    async def run(
        self,
        command: str,
        *,
        cwd: Path,
        max_output_bytes: int,
        timeout_seconds: float,
    ) -> CommandResult:
        if os.name != "posix":
            raise RuntimeError("The local command runner requires a POSIX host")
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-lc",
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        assert process.stdout is not None
        assert process.stderr is not None

        stdout_buffer = _BoundedBuffer(max_output_bytes)
        stderr_buffer = _BoundedBuffer(max_output_bytes)
        readers = (
            asyncio.create_task(_drain(process.stdout, stdout_buffer)),
            asyncio.create_task(_drain(process.stderr, stderr_buffer)),
        )
        timed_out = False
        cancelled = False

        try:
            try:
                await asyncio.wait_for(process.wait(), timeout_seconds)
            except TimeoutError:
                timed_out = True
            if timed_out:
                await _terminate_group(process)
            elif process.returncode is not None:
                # Clean up ordinary background children still in the command's
                # process group after the shell exits.
                _signal_group(process, signal.SIGTERM)
                _signal_group(process, signal.SIGKILL)
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:

            async def cleanup() -> None:
                try:
                    if cancelled or process.returncode is None:
                        await _terminate_group(process)
                    await _finish_readers(readers)
                finally:
                    _close_transport(process)

            await _await_uninterruptibly(asyncio.create_task(cleanup()))

        stdout = bytes(stdout_buffer.data).decode("utf-8", errors="replace")
        stderr = bytes(stderr_buffer.data).decode("utf-8", errors="replace")
        stdout, stderr, combined_truncated = _truncate_streams(
            stdout,
            stderr,
            max_output_bytes,
        )
        return CommandResult(
            exit_code=None if timed_out else process.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            truncated=(
                stdout_buffer.truncated or stderr_buffer.truncated or combined_truncated
            ),
        )
