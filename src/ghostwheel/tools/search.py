from collections import deque
from collections.abc import Iterator
import os
from pathlib import Path
import stat

import regex
from pydantic import BaseModel
from pydantic_ai import RunContext

from .deps import ToolDeps
from .output import OutputBudget, normalize_utf8, truncate_utf8
from .path_filters import NOISE_DIRECTORY_NAMES, glob_matches
from .workspace import Workspace


class GrepMatch(BaseModel):
    file: str
    line: int
    text: str


class GrepResult(BaseModel):
    pattern: str
    matches: list[GrepMatch]
    truncated: bool
    files_searched: int
    files_skipped: int = 0


def _walk_candidates(
    workspace: Workspace,
    root: Path,
    file_glob: str,
    max_entries: int,
) -> Iterator[Path | None]:
    pending_directories = deque([Path()])
    entries_seen = 0
    incomplete = False
    while pending_directories:
        relative_directory = pending_directories.popleft()
        directory_candidates: list[Path] = []
        limit_reached = False
        try:
            with workspace.open_directory(root / relative_directory) as opened:
                with os.scandir(opened.fd) as iterator:
                    for entry in iterator:
                        entries_seen += 1
                        if entries_seen > max_entries:
                            limit_reached = True
                            break
                        try:
                            entry_stat = os.stat(
                                entry.name,
                                dir_fd=opened.fd,
                                follow_symlinks=False,
                            )
                        except OSError:
                            incomplete = True
                            continue
                        relative = relative_directory / entry.name
                        if stat.S_ISDIR(entry_stat.st_mode):
                            if entry.name in NOISE_DIRECTORY_NAMES:
                                continue
                            pending_directories.append(relative)
                            continue
                        if not stat.S_ISREG(entry_stat.st_mode):
                            continue
                        if glob_matches(relative, file_glob):
                            directory_candidates.append(root / relative)
        except OSError:
            # A subtree disappeared or became inaccessible while walking. The
            # result is incomplete, so signal truncation rather than silently
            # claiming a complete search.
            yield None
            return
        yield from directory_candidates
        if limit_reached:
            yield None
            return
    if incomplete:
        yield None


def grep(
    ctx: RunContext[ToolDeps],
    pattern: str,
    path: str = ".",
    file_glob: str = "*",
    case_sensitive: bool = True,
) -> GrepResult:
    """Search for a regular expression across bounded workspace files.

    Files, retained matches, payload bytes, and each regular-expression search
    are bounded by ``ToolLimits``. Symlinks are never traversed.
    """
    flags = 0 if case_sensitive else regex.IGNORECASE
    try:
        compiled = regex.compile(pattern, flags)
    except regex.error as error:
        raise ValueError(f"Invalid regex: {error}") from error

    workspace = ctx.deps.workspace
    target = workspace.locate(path)
    if workspace.is_file(target.absolute):
        candidates = iter((target.absolute,))
    elif workspace.is_directory(target.absolute):
        candidates = _walk_candidates(
            workspace,
            target.absolute,
            file_glob,
            ctx.deps.limits.max_search_files,
        )
    else:
        raise FileNotFoundError(f"Path does not exist: {target.absolute}")

    display_pattern, pattern_truncated = truncate_utf8(
        pattern,
        max(1, ctx.deps.limits.max_output_bytes // 4),
    )
    output_budget = OutputBudget(ctx.deps.limits.max_output_bytes)
    output_budget.consume(display_pattern)
    matches: list[GrepMatch] = []
    files_searched = 0
    files_skipped = 0
    truncated = pattern_truncated

    for candidate in candidates:
        if candidate is None:
            truncated = True
            break

        try:
            with workspace.open_file(candidate) as opened:
                if opened.stat.st_size > ctx.deps.limits.max_search_file_bytes:
                    files_skipped += 1
                    truncated = True
                    continue
                raw = opened.file.read(ctx.deps.limits.max_search_file_bytes + 1)
                if len(raw) > ctx.deps.limits.max_search_file_bytes:
                    files_skipped += 1
                    truncated = True
                    continue
                display_path = normalize_utf8(
                    workspace.display_path(opened.path.absolute)
                )
        except OSError, ValueError:
            files_skipped += 1
            truncated = True
            continue

        files_searched += 1
        for line_number, line in enumerate(
            raw.decode("utf-8", errors="replace").splitlines(),
            1,
        ):
            try:
                matched = compiled.search(
                    line,
                    timeout=ctx.deps.limits.regex_timeout_seconds,
                )
            except TimeoutError as error:
                raise TimeoutError(
                    "Regular expression exceeded the configured per-line timeout"
                ) from error
            if matched is None:
                continue
            if len(matches) >= ctx.deps.limits.max_matches:
                return GrepResult(
                    pattern=display_pattern,
                    matches=matches,
                    files_searched=files_searched,
                    files_skipped=files_skipped,
                    truncated=True,
                )

            metadata = f"{display_path}:{line_number}:"
            if not output_budget.consume(metadata):
                return GrepResult(
                    pattern=display_pattern,
                    matches=matches,
                    files_searched=files_searched,
                    files_skipped=files_skipped,
                    truncated=True,
                )
            fitted_text, text_truncated = truncate_utf8(
                line,
                output_budget.remaining_bytes,
            )
            output_budget.consume(fitted_text)
            matches.append(
                GrepMatch(
                    file=display_path,
                    line=line_number,
                    text=fitted_text,
                )
            )
            if text_truncated:
                return GrepResult(
                    pattern=display_pattern,
                    matches=matches,
                    files_searched=files_searched,
                    files_skipped=files_skipped,
                    truncated=True,
                )

    return GrepResult(
        pattern=display_pattern,
        matches=matches,
        files_searched=files_searched,
        files_skipped=files_skipped,
        truncated=truncated,
    )
