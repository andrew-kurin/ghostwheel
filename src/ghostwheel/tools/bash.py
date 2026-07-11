from pydantic import BaseModel
from pydantic_ai import RunContext

from ghostwheel.tools.deps import ToolDeps
from ghostwheel.tools.output import normalize_utf8


class BashResult(BaseModel):
    command: str
    cwd: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False


async def bash(ctx: RunContext[ToolDeps], command: str) -> BashResult:
    """Run a shell command in the project working directory.

    The command runner owns timeout, cancellation, process-group cleanup, and
    bounded output capture. It does not restrict shell syntax or filesystem access.
    """
    if ctx.deps.dry_run:
        return BashResult(
            command=normalize_utf8(command),
            cwd=normalize_utf8(str(ctx.deps.cwd)),
            exit_code=None,
            stdout="",
            stderr="Dry run: command was not executed.",
        )

    result = await ctx.deps.command_runner.run(
        command,
        cwd=ctx.deps.cwd,
        max_output_bytes=ctx.deps.limits.max_output_bytes,
        timeout_seconds=ctx.deps.limits.bash_timeout_seconds,
    )
    return BashResult(
        command=normalize_utf8(command),
        cwd=normalize_utf8(str(ctx.deps.cwd)),
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        timed_out=result.timed_out,
        truncated=result.truncated,
    )
