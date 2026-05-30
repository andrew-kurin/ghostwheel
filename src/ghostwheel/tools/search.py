import re
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import RunContext
from .deps import ToolDeps


class GrepMatch(BaseModel):
    file: str
    line: int
    text: str


class GrepResult(BaseModel):
    pattern: str
    matches: list[GrepMatch]
    truncated: bool
    files_searched: int  # useful signal: "I searched 50 files and found 0" is different from "I searched 0 files"


_NOISE_DIRS = {".git", "node_modules", "__pycache__", ".venv"}


def _is_allowed(ctx: RunContext[ToolDeps], path: Path) -> bool:
    return any(path.is_relative_to(root.resolve()) for root in ctx.deps.allowed_roots)


def _is_noise_path(path: Path) -> bool:
    return any(part in _NOISE_DIRS for part in path.parts)


def _match_path(ctx: RunContext[ToolDeps], file_path: Path) -> str:
    try:
        return str(file_path.relative_to(ctx.deps.cwd))
    except ValueError:
        return str(file_path)


def grep(
    ctx: RunContext[ToolDeps],
    pattern: str,
    path: str = ".",
    file_glob: str = "*",
    case_sensitive: bool = True,
) -> GrepResult:
    """Search for a regex pattern across files.

    Args:
        pattern: Regular expression to search for.
        path: Directory to search in, relative to the working directory.
        file_glob: Glob pattern for files to search (e.g., "*.py", "*.md"). Defaults to all files.
        case_sensitive: Whether the pattern match is case-sensitive.

    Returns matches with file path, line number, and matching line text.
    Capped at 200 matches; if more exist, 'truncated' will be True.
    """
    target = (ctx.deps.cwd / path).expanduser().resolve()

    if not _is_allowed(ctx, target):
        raise ValueError(f"Path {target} is outside allowed roots")
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {target}")

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        raise ValueError(f"Invalid regex: {e}")

    MAX_MATCHES = 200
    matches: list[GrepMatch] = []
    files_searched = 0
    truncated = False

    files = [target] if target.is_file() else target.rglob(file_glob)

    for file_path in files:
        if _is_noise_path(file_path):
            continue

        try:
            resolved_file_path = file_path.resolve()
        except (OSError, RuntimeError):
            continue

        if not resolved_file_path.is_file():
            continue
        if not _is_allowed(ctx, resolved_file_path):
            continue
        if _is_noise_path(resolved_file_path):
            continue

        files_searched += 1

        try:
            with resolved_file_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    if regex.search(line):
                        if len(matches) >= MAX_MATCHES:
                            truncated = True
                            return GrepResult(
                                pattern=pattern,
                                matches=matches,
                                files_searched=files_searched,
                                truncated=truncated,
                            )
                        matches.append(
                            GrepMatch(
                                file=_match_path(ctx, resolved_file_path),
                                line=line_num,
                                text=line.rstrip("\n"),
                            )
                        )
        except OSError:
            continue

    return GrepResult(
        pattern=pattern,
        matches=matches,
        files_searched=files_searched,
        truncated=truncated,
    )
