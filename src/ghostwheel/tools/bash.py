import subprocess
from pydantic import BaseModel
from pydantic_ai import RunContext

from ghostwheel.tools.deps import ToolDeps


class BashResult(BaseModel):
    command: str
    cwd: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False


def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _truncate_to_bytes(value: str, max_bytes: int) -> str:
    return value.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def _truncate_streams(stdout: str, stderr: str, max_bytes: int) -> tuple[str, str, bool]:
    stdout_bytes = stdout.encode("utf-8")
    stderr_bytes = stderr.encode("utf-8")

    if len(stdout_bytes) + len(stderr_bytes) <= max_bytes:
        return stdout, stderr, False
    if max_bytes <= 0:
        return "", "", True

    stdout_budget = min(len(stdout_bytes), max_bytes)
    stderr_budget = max_bytes - stdout_budget
    if stderr_bytes and stderr_budget == 0:
        stderr_budget = min(len(stderr_bytes), max_bytes // 2)
        stdout_budget = max_bytes - stderr_budget

    return (
        _truncate_to_bytes(stdout, stdout_budget),
        _truncate_to_bytes(stderr, stderr_budget),
        True,
    )


def bash(ctx: RunContext[ToolDeps], command: str) -> BashResult:
    """Run a shell command in the project working directory.

    Use for inspection commands like rg, git status, python -m pytest, etc.
    Do not use destructive commands unless explicitly requested by the user.
    """
    if ctx.deps.dry_run:
        return BashResult(
            command=command,
            cwd=str(ctx.deps.cwd),
            exit_code=None,
            stdout="",
            stderr="Dry run: command was not executed.",
        )

    try:
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=ctx.deps.cwd,
            text=True,
            capture_output=True,
            timeout=ctx.deps.bash_timeout_seconds,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        exit_code = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = _to_text(exc.stdout)
        stderr = _to_text(exc.stderr)
        exit_code = None
        timed_out = True

    stdout, stderr, truncated = _truncate_streams(
        stdout,
        stderr,
        ctx.deps.max_output_bytes,
    )

    return BashResult(
        command=command,
        cwd=str(ctx.deps.cwd),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        truncated=truncated,
    )
