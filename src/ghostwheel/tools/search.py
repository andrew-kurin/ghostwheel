import re
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

    if not any(target.is_relative_to(root) for root in ctx.deps.allowed_roots):
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
        if not file_path.is_file():
            continue

        # Skip common noise directories
        if any(
            part in {".git", "node_modules", "__pycache__", ".venv"}
            for part in file_path.parts
        ):
            continue

        files_searched += 1

        try:
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
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
                                file=str(
                                    file_path.relative_to(
                                        target if target.is_dir() else target.parent
                                    )
                                ),
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
