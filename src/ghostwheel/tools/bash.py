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

    max_bytes = ctx.deps.max_output_bytes
    combined = stdout + stderr
    truncated = len(combined.encode()) > max_bytes

    if truncated:
        stdout = stdout[:max_bytes]
        stderr = stderr[:max_bytes]

    return BashResult(
        command=command,
        cwd=str(ctx.deps.cwd),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        truncated=truncated,
    )
