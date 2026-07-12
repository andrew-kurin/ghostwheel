import base64
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import time
from typing import Annotated, Any, BinaryIO

from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_ai.messages import ToolReturn
import regex

from .deps import ToolDeps
from .output import normalize_utf8
from .path_filters import CompiledPathGlob, NOISE_DIRECTORY_NAMES
from .workspace import Workspace


class GrepIncompleteReason(str, Enum):
    SCAN_LIMIT = "scan_limit"
    DEPTH_LIMIT = "depth_limit"
    ENTRY_ERROR = "entry_error"
    FILE_LIMIT = "file_limit"
    FILE_ERROR = "file_error"
    FILE_TOO_LARGE = "file_too_large"
    TOTAL_BYTES = "total_bytes"
    BINARY_FILE = "binary_file"
    ENCODING_ERROR = "encoding_error"
    TIMEOUT = "timeout"
    MATCH_LIMIT = "match_limit"
    PAGE_LIMIT = "page_limit"
    OUTPUT_BUDGET = "output_budget"


class GrepMatch(BaseModel):
    file: str
    line: int
    column: int = 1
    text: str
    text_truncated: bool = False


class GrepResult(BaseModel):
    pattern: str
    path: str
    matches: list[GrepMatch]
    entries_scanned: int
    files_searched: int
    files_skipped: int = 0
    bytes_inspected: int = 0
    output_skipped: int = 0
    complete: bool
    reasons: list[GrepIncompleteReason]
    next_cursor: str | None = None

    @property
    def truncated(self) -> bool:
        """Compatibility view of the former truncated field."""

        return not self.complete


_MAX_SEARCH_DEPTH = 64
_MAX_SNIPPET_CHARACTERS = 600
_MAX_CURSOR_LENGTH = 4_096
_MAX_PATTERN_CHARACTERS = 10_000
_MAX_GLOB_CHARACTERS = 4_096
_REGEX_META_CHARACTERS = frozenset(r"\.^$*+?{}[]|()")
_REASON_ORDER = (
    GrepIncompleteReason.SCAN_LIMIT,
    GrepIncompleteReason.DEPTH_LIMIT,
    GrepIncompleteReason.ENTRY_ERROR,
    GrepIncompleteReason.FILE_LIMIT,
    GrepIncompleteReason.FILE_ERROR,
    GrepIncompleteReason.FILE_TOO_LARGE,
    GrepIncompleteReason.TOTAL_BYTES,
    GrepIncompleteReason.BINARY_FILE,
    GrepIncompleteReason.ENCODING_ERROR,
    GrepIncompleteReason.TIMEOUT,
    GrepIncompleteReason.MATCH_LIMIT,
    GrepIncompleteReason.PAGE_LIMIT,
    GrepIncompleteReason.OUTPUT_BUDGET,
)


@dataclass(frozen=True, slots=True)
class _MatchCandidate:
    file: str
    raw_key: bytes
    line: int
    column: int
    text: str
    text_truncated: bool

    def as_match(self) -> GrepMatch:
        return GrepMatch(
            file=self.file,
            line=self.line,
            column=self.column,
            text=self.text,
            text_truncated=self.text_truncated,
        )


@dataclass(frozen=True, slots=True)
class _LineMatcher:
    substring: str | None
    compiled: Any | None
    timeout_required: bool

    @classmethod
    def create(
        cls,
        pattern: str,
        *,
        case_sensitive: bool,
        literal: bool,
    ) -> "_LineMatcher":
        flags = 0 if case_sensitive else regex.IGNORECASE
        syntax_free = not any(char in _REGEX_META_CHARACTERS for char in pattern)
        effective_literal = literal or syntax_free
        if effective_literal and case_sensitive:
            return cls(pattern, None, False)
        expression = regex.escape(pattern) if effective_literal else pattern
        try:
            compiled = regex.compile(expression, flags)
        except regex.error as error:
            raise ValueError(f"Invalid regex: {error}") from error
        return cls(None, compiled, not effective_literal)

    def search(self, line: str, *, timeout: float) -> tuple[int, int] | None:
        if self.substring is not None:
            start = line.find(self.substring)
            return None if start < 0 else (start, start + len(self.substring))
        assert self.compiled is not None
        match = (
            self.compiled.search(line, timeout=timeout)
            if self.timeout_required
            else self.compiled.search(line)
        )
        if match is None:
            return None
        return match.start(), match.end()


@dataclass(slots=True)
class _SearchState:
    workspace: Workspace
    path_glob: CompiledPathGlob
    line_matcher: _LineMatcher
    show_hidden: bool
    include_noise: bool
    scan_limit: int
    file_limit: int
    file_byte_limit: int
    total_byte_limit: int
    match_limit: int
    regex_timeout_seconds: float
    deadline: float
    entries_scanned: int = 0
    files_considered: int = 0
    files_searched: int = 0
    files_skipped: int = 0
    bytes_inspected: int = 0
    matches: list[_MatchCandidate] = field(default_factory=list)
    reasons: set[GrepIncompleteReason] = field(default_factory=set)
    visited_directories: set[tuple[int, int]] = field(default_factory=set)
    stop: bool = False


def _remaining_seconds(state: _SearchState) -> float:
    return state.deadline - time.monotonic()


def _check_deadline(state: _SearchState) -> bool:
    if _remaining_seconds(state) > 0:
        return False
    state.reasons.add(GrepIncompleteReason.TIMEOUT)
    state.stop = True
    return True


def _claim_file(state: _SearchState) -> bool:
    if state.files_considered >= state.file_limit:
        state.reasons.add(GrepIncompleteReason.FILE_LIMIT)
        state.stop = True
        return False
    state.files_considered += 1
    return True


def _iter_lf_lines(text: str) -> Iterator[tuple[int, str]]:
    if not text:
        return
    start = 0
    line_number = 1
    while start < len(text):
        end = text.find("\n", start)
        if end < 0:
            line = text[start:]
            if line.endswith("\r"):
                line = line[:-1]
            yield line_number, line
            return
        line = text[start:end]
        if line.endswith("\r"):
            line = line[:-1]
        yield line_number, line
        line_number += 1
        start = end + 1


def _centered_snippet(
    line: str,
    start: int,
    end: int,
) -> tuple[str, bool]:
    if len(line) <= _MAX_SNIPPET_CHARACTERS:
        return normalize_utf8(line), False

    match_length = max(1, end - start)
    if match_length >= _MAX_SNIPPET_CHARACTERS - 2:
        left = start
        right = min(len(line), start + _MAX_SNIPPET_CHARACTERS - 2)
    else:
        context = _MAX_SNIPPET_CHARACTERS - match_length - 2
        left = max(0, start - context // 2)
        right = min(len(line), end + context - (start - left))
        if right == len(line):
            left = max(0, right - (_MAX_SNIPPET_CHARACTERS - 2))

    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(line) else ""
    snippet = f"{prefix}{line[left:right]}{suffix}"
    return normalize_utf8(snippet), True


def _search_open_file(
    state: _SearchState,
    file: BinaryIO,
    file_stat: os.stat_result,
    *,
    absolute_path: Path,
    raw_key: bytes,
) -> None:
    if state.stop or _check_deadline(state):
        return
    if file_stat.st_size > state.file_byte_limit:
        state.files_skipped += 1
        state.reasons.add(GrepIncompleteReason.FILE_TOO_LARGE)
        return
    if state.bytes_inspected + file_stat.st_size > state.total_byte_limit:
        state.files_skipped += 1
        state.reasons.add(GrepIncompleteReason.TOTAL_BYTES)
        state.stop = True
        return

    remaining_total_bytes = state.total_byte_limit - state.bytes_inspected
    retained_byte_limit = min(state.file_byte_limit, remaining_total_bytes)
    try:
        raw = file.read(retained_byte_limit + 1)
    except OSError:
        state.files_skipped += 1
        state.reasons.add(GrepIncompleteReason.FILE_ERROR)
        return
    state.bytes_inspected += min(len(raw), retained_byte_limit)
    if (
        len(raw) > retained_byte_limit
        and remaining_total_bytes <= state.file_byte_limit
    ):
        state.files_skipped += 1
        state.reasons.add(GrepIncompleteReason.TOTAL_BYTES)
        state.stop = True
        return
    if len(raw) > retained_byte_limit:
        state.files_skipped += 1
        state.reasons.add(GrepIncompleteReason.FILE_TOO_LARGE)
        return
    if b"\0" in raw:
        state.files_skipped += 1
        state.reasons.add(GrepIncompleteReason.BINARY_FILE)
        return
    try:
        decoded = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        state.files_skipped += 1
        state.reasons.add(GrepIncompleteReason.ENCODING_ERROR)
        return
    if _check_deadline(state):
        return

    state.files_searched += 1
    display_path: str | None = None
    for line_number, line in _iter_lf_lines(decoded):
        if state.stop:
            return
        if state.line_matcher.timeout_required:
            remaining = _remaining_seconds(state)
            if remaining <= 0:
                state.reasons.add(GrepIncompleteReason.TIMEOUT)
                state.stop = True
                return
            timeout = min(state.regex_timeout_seconds, remaining)
        else:
            timeout = state.regex_timeout_seconds
        try:
            matched = state.line_matcher.search(line, timeout=timeout)
        except TimeoutError:
            state.reasons.add(GrepIncompleteReason.TIMEOUT)
            state.stop = True
            return
        if (
            not state.line_matcher.timeout_required or matched is not None
        ) and _check_deadline(state):
            return
        if matched is None:
            continue

        if display_path is None:
            display_path = normalize_utf8(state.workspace.display_path(absolute_path))
        start, end = matched
        snippet, text_truncated = _centered_snippet(line, start, end)
        state.matches.append(
            _MatchCandidate(
                file=display_path,
                raw_key=raw_key,
                line=line_number,
                column=start + 1,
                text=snippet,
                text_truncated=text_truncated,
            )
        )
        if len(state.matches) >= state.match_limit:
            state.reasons.add(GrepIncompleteReason.MATCH_LIMIT)
            state.stop = True
            return


def _open_and_search_child_file(
    state: _SearchState,
    *,
    parent_fd: int,
    child_name: str,
    absolute_path: Path,
    raw_key: bytes,
) -> None:
    if not _claim_file(state):
        return
    descriptor = -1
    try:
        descriptor = os.open(
            child_name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent_fd,
        )
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            state.files_skipped += 1
            state.reasons.add(GrepIncompleteReason.FILE_ERROR)
            return
        file = os.fdopen(descriptor, "rb")
        descriptor = -1
        with file:
            _search_open_file(
                state,
                file,
                file_stat,
                absolute_path=absolute_path,
                raw_key=raw_key,
            )
    except OSError:
        state.files_skipped += 1
        state.reasons.add(GrepIncompleteReason.FILE_ERROR)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _walk_directory(
    state: _SearchState,
    *,
    directory_fd: int,
    absolute_path: Path,
    relative_prefix: str,
    depth: int,
) -> None:
    batch: list[tuple[str, str, bytes, bool]] = []
    try:
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                if state.stop or _check_deadline(state):
                    break
                if state.entries_scanned >= state.scan_limit:
                    state.reasons.add(GrepIncompleteReason.SCAN_LIMIT)
                    break
                state.entries_scanned += 1
                if not state.show_hidden and entry.name.startswith("."):
                    continue
                try:
                    if entry.is_symlink():
                        continue
                    is_directory = entry.is_dir(follow_symlinks=False)
                    is_file = (
                        False if is_directory else entry.is_file(follow_symlinks=False)
                    )
                except OSError:
                    state.reasons.add(GrepIncompleteReason.ENTRY_ERROR)
                    continue
                if is_directory:
                    if not state.include_noise and entry.name in NOISE_DIRECTORY_NAMES:
                        continue
                elif not is_file:
                    continue
                relative = (
                    f"{relative_prefix}/{entry.name}" if relative_prefix else entry.name
                )
                batch.append(
                    (entry.name, relative, os.fsencode(relative), is_directory)
                )
    except OSError:
        state.reasons.add(GrepIncompleteReason.ENTRY_ERROR)
        return

    batch.sort(key=lambda item: item[2])
    for child_name, relative, raw_key, is_directory in batch:
        if state.stop:
            return
        child_absolute = absolute_path / child_name
        if not is_directory:
            if state.path_glob.matches(relative):
                _open_and_search_child_file(
                    state,
                    parent_fd=directory_fd,
                    child_name=child_name,
                    absolute_path=child_absolute,
                    raw_key=raw_key,
                )
            continue

        if GrepIncompleteReason.SCAN_LIMIT in state.reasons:
            continue
        if depth >= _MAX_SEARCH_DEPTH:
            state.reasons.add(GrepIncompleteReason.DEPTH_LIMIT)
            continue
        descriptor = -1
        try:
            descriptor = os.open(
                child_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            directory_stat = os.fstat(descriptor)
            if not stat.S_ISDIR(directory_stat.st_mode):
                state.reasons.add(GrepIncompleteReason.ENTRY_ERROR)
                continue
            identity = (directory_stat.st_dev, directory_stat.st_ino)
            if identity in state.visited_directories:
                state.reasons.add(GrepIncompleteReason.ENTRY_ERROR)
                continue
            state.visited_directories.add(identity)
            _walk_directory(
                state,
                directory_fd=descriptor,
                absolute_path=child_absolute,
                relative_prefix=relative,
                depth=depth + 1,
            )
        except OSError:
            state.reasons.add(GrepIncompleteReason.ENTRY_ERROR)
        finally:
            if descriptor >= 0:
                os.close(descriptor)


def _urlsafe_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _urlsafe_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid grep cursor") from error


def _query_fingerprint(
    absolute_path: Path,
    root_stat: os.stat_result,
    *,
    display_path: str,
    pattern: str,
    file_glob: str,
    case_sensitive: bool,
    literal: bool,
    show_hidden: bool,
    include_noise: bool,
) -> str:
    digest = hashlib.sha256()
    components = (
        os.fsencode(absolute_path),
        str(root_stat.st_dev).encode("ascii"),
        str(root_stat.st_ino).encode("ascii"),
        display_path.encode("utf-8", errors="surrogatepass"),
        pattern.encode("utf-8", errors="surrogatepass"),
        file_glob.encode("utf-8", errors="surrogatepass"),
        b"case:1" if case_sensitive else b"case:0",
        b"literal:1" if literal else b"literal:0",
        b"hidden:1" if show_hidden else b"hidden:0",
        b"noise:1" if include_noise else b"noise:0",
    )
    for component in components:
        digest.update(len(component).to_bytes(8, "big"))
        digest.update(component)
    return _urlsafe_encode(digest.digest()[:12])


def _snapshot_fingerprint(matches: list[_MatchCandidate]) -> str:
    digest = hashlib.sha256()
    for match in matches:
        digest.update(len(match.raw_key).to_bytes(4, "big"))
        digest.update(match.raw_key)
        digest.update(match.line.to_bytes(8, "big"))
        digest.update(match.column.to_bytes(8, "big"))
        text = match.text.encode("utf-8")
        digest.update(len(text).to_bytes(4, "big"))
        digest.update(text)
        digest.update(b"1" if match.text_truncated else b"0")
    return _urlsafe_encode(digest.digest()[:12])


def _encode_cursor(
    query_fingerprint: str,
    snapshot_fingerprint: str,
    offset: int,
) -> str:
    return f"g1.{query_fingerprint}.{snapshot_fingerprint}.{offset:x}"


def _decode_cursor(cursor: str | None, query_fingerprint: str) -> tuple[str, int]:
    if cursor is None:
        return "", 0
    if len(cursor) > _MAX_CURSOR_LENGTH:
        raise ValueError("Invalid grep cursor")
    parts = cursor.split(".")
    if len(parts) != 4 or parts[0] != "g1" or parts[1] != query_fingerprint:
        raise ValueError("Invalid or mismatched grep cursor")
    if len(_urlsafe_decode(parts[2])) != 12:
        raise ValueError("Invalid grep cursor")
    try:
        offset = int(parts[3], 16)
    except ValueError as error:
        raise ValueError("Invalid grep cursor") from error
    if offset <= 0:
        raise ValueError("Invalid grep cursor")
    return parts[2], offset


def _ordered_reasons(
    reasons: set[GrepIncompleteReason],
) -> list[GrepIncompleteReason]:
    return [reason for reason in _REASON_ORDER if reason in reasons]


def _quote(value: str) -> str:
    return json.dumps(normalize_utf8(value), ensure_ascii=True)


def _render_search(
    *,
    path: str,
    matches: list[_MatchCandidate],
    state: _SearchState,
    output_skipped: int,
    reasons: list[GrepIncompleteReason],
    next_cursor: str | None,
) -> str:
    reason_text = ",".join(reason.value for reason in reasons) or "-"
    omitted_text = f" omitted={output_skipped}" if output_skipped else ""
    lines = [
        f"grep {_quote(path)} returned={len(matches)}{omitted_text} "
        f"scanned={state.entries_scanned} searched={state.files_searched} "
        f"file_skipped={state.files_skipped} bytes={state.bytes_inspected} "
        f"complete={'false' if reasons else 'true'} reasons={reason_text}"
    ]
    previous_key: bytes | None = None
    for match in matches:
        if match.raw_key != previous_key:
            lines.append(f"f {_quote(match.file)}")
            previous_key = match.raw_key
        marker = "~" if match.text_truncated else ""
        lines.append(f"{match.line}:{match.column}{marker} {_quote(match.text)}")
    if next_cursor is not None:
        lines.append(f"next {_quote(next_cursor)}")
    return "\n".join(lines)


def _fit_search_page(
    *,
    path: str,
    state: _SearchState,
    matches: list[_MatchCandidate],
    limit: int,
    query_fingerprint: str,
    snapshot_fingerprint: str,
    start_offset: int,
    max_output_bytes: int,
) -> tuple[
    str,
    list[_MatchCandidate],
    list[GrepIncompleteReason],
    str | None,
    int,
]:
    page = matches[:limit]
    page_limit_reached = len(matches) > limit
    for count in range(len(page), -1, -1):
        emitted = page[:count]
        output_skipped = 0
        reasons = set(state.reasons)
        if page_limit_reached:
            reasons.add(GrepIncompleteReason.PAGE_LIMIT)
        if count < len(page):
            reasons.add(GrepIncompleteReason.OUTPUT_BUDGET)

        has_more = len(matches) > count
        next_cursor = None
        if has_more:
            cursor_offset = start_offset + count
            if not emitted and page:
                output_skipped = 1
                cursor_offset += 1
            if emitted or output_skipped:
                next_cursor = _encode_cursor(
                    query_fingerprint,
                    snapshot_fingerprint,
                    cursor_offset,
                )

        ordered_reasons = _ordered_reasons(reasons)
        rendered = _render_search(
            path=path,
            matches=emitted,
            state=state,
            output_skipped=output_skipped,
            reasons=ordered_reasons,
            next_cursor=next_cursor,
        )
        if len(rendered.encode("utf-8")) <= max_output_bytes:
            return rendered, emitted, ordered_reasons, next_cursor, output_skipped

    reasons = set(state.reasons)
    if page_limit_reached:
        reasons.add(GrepIncompleteReason.PAGE_LIMIT)
    reasons.add(GrepIncompleteReason.OUTPUT_BUDGET)
    ordered_reasons = _ordered_reasons(reasons)
    reason_text = ",".join(reason.value for reason in ordered_reasons)
    output_skipped = 1 if page else 0
    next_cursor = None
    if output_skipped:
        next_cursor = _encode_cursor(
            query_fingerprint,
            snapshot_fingerprint,
            start_offset + 1,
        )
        with_cursor = (
            f"grep omitted=1 reasons={reason_text}\nnext {_quote(next_cursor)}"
        )
        if len(with_cursor.encode("utf-8")) <= max_output_bytes:
            return (
                with_cursor,
                [],
                ordered_reasons,
                next_cursor,
                output_skipped,
            )
        next_cursor = None

    minimal = (
        f"grep incomplete{' omitted=1' if output_skipped else ''} reasons={reason_text}"
    )
    compact = f"grep{' omitted=1' if output_skipped else ''} {reason_text}"
    if len(minimal.encode("utf-8")) <= max_output_bytes:
        rendered = minimal
    elif len(compact.encode("utf-8")) <= max_output_bytes:
        rendered = compact
    else:
        rendered = ""
    return rendered, [], ordered_reasons, next_cursor, output_skipped


def _finish_search(
    *,
    pattern: str,
    path: str,
    state: _SearchState,
    query_fingerprint: str,
    cursor: str | None,
    cursor_snapshot: str,
    start_offset: int,
    limit: int,
    max_output_bytes: int,
) -> ToolReturn[str]:
    snapshot = _snapshot_fingerprint(state.matches)
    if cursor is not None and cursor_snapshot != snapshot:
        raise ValueError("The search results changed; restart grep pagination")
    if start_offset > len(state.matches):
        raise ValueError("Invalid grep cursor")
    remaining = state.matches[start_offset:]
    rendered, emitted, reasons, next_cursor, output_skipped = _fit_search_page(
        path=path,
        state=state,
        matches=remaining,
        limit=limit,
        query_fingerprint=query_fingerprint,
        snapshot_fingerprint=snapshot,
        start_offset=start_offset,
        max_output_bytes=max_output_bytes,
    )
    metadata = GrepResult(
        pattern=normalize_utf8(pattern),
        path=path,
        matches=[match.as_match() for match in emitted],
        entries_scanned=state.entries_scanned,
        files_searched=state.files_searched,
        files_skipped=state.files_skipped,
        bytes_inspected=state.bytes_inspected,
        output_skipped=output_skipped,
        complete=not reasons,
        reasons=reasons,
        next_cursor=next_cursor,
    )
    return ToolReturn[str](return_value=rendered, metadata=metadata)


def grep(
    ctx: RunContext[ToolDeps],
    pattern: Annotated[
        str,
        Field(min_length=1, max_length=_MAX_PATTERN_CHARACTERS),
    ],
    path: str = ".",
    *,
    file_glob: Annotated[
        str,
        Field(min_length=1, max_length=_MAX_GLOB_CHARACTERS),
    ] = "*",
    case_sensitive: bool = True,
    literal: bool = False,
    limit: Annotated[int | None, Field(gt=0)] = None,
    cursor: str | None = None,
    show_hidden: bool = False,
    include_noise: bool = False,
) -> ToolReturn[str]:
    """Search UTF-8 text files with a bounded line-oriented regular expression.

    Args:
        pattern: Non-empty Python regex-package expression, or exact text in literal mode.
        path: Workspace-relative file or directory to search. Defaults to '.'.
        file_glob: Non-empty relative glob applied to recursive file paths.
        case_sensitive: Apply case-sensitive matching. Defaults to True.
        literal: Treat pattern as exact text instead of regex syntax.
        limit: Maximum result rows for this page, capped by the configured match limit.
        cursor: Opaque continuation cursor returned by a previous matching call.
        show_hidden: Include entries whose names start with '.'. Defaults to False.
        include_noise: Include common dependency and cache directories.

    Each matching line is returned once at its first match. Rows are grouped by
    file as LINE:COLUMN JSON_SNIPPET; '~' after the column marks an elided line.
    Binary, invalid UTF-8, oversized, and inaccessible files are skipped explicitly.
    """
    if not pattern:
        raise ValueError("pattern must not be empty")
    if len(pattern) > _MAX_PATTERN_CHARACTERS:
        raise ValueError(
            f"pattern must be at most {_MAX_PATTERN_CHARACTERS} characters"
        )
    if not file_glob or PurePosixPath(file_glob).is_absolute():
        raise ValueError("file_glob must be a non-empty relative pattern")
    if len(file_glob) > _MAX_GLOB_CHARACTERS:
        raise ValueError(f"file_glob must be at most {_MAX_GLOB_CHARACTERS} characters")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if cursor is not None and len(cursor) > _MAX_CURSOR_LENGTH:
        raise ValueError("Invalid grep cursor")

    deadline = time.monotonic() + ctx.deps.limits.search_timeout_seconds
    line_matcher = _LineMatcher.create(
        pattern,
        case_sensitive=case_sensitive,
        literal=literal,
    )
    path_glob = CompiledPathGlob.compile(file_glob)
    workspace = ctx.deps.workspace
    target = workspace.locate(path)
    target_stat = workspace.stat_path(target.absolute)
    effective_limit = min(
        ctx.deps.limits.max_matches,
        ctx.deps.limits.max_matches if limit is None else limit,
    )

    def make_state() -> _SearchState:
        return _SearchState(
            workspace=workspace,
            path_glob=path_glob,
            line_matcher=line_matcher,
            show_hidden=show_hidden,
            include_noise=include_noise,
            scan_limit=ctx.deps.limits.max_directory_scan_entries,
            file_limit=ctx.deps.limits.max_search_files,
            file_byte_limit=ctx.deps.limits.max_search_file_bytes,
            total_byte_limit=ctx.deps.limits.max_search_total_bytes,
            match_limit=ctx.deps.limits.max_matches,
            regex_timeout_seconds=ctx.deps.limits.regex_timeout_seconds,
            deadline=deadline,
        )

    if stat.S_ISREG(target_stat.st_mode):
        display_path = normalize_utf8(workspace.display_path(target.absolute))
        state = make_state()
        with ExitStack() as stack:
            try:
                opened = stack.enter_context(workspace.open_file(target.absolute))
            except OSError, ValueError:
                query_fingerprint = _query_fingerprint(
                    target.absolute,
                    target_stat,
                    display_path=display_path,
                    pattern=pattern,
                    file_glob=file_glob,
                    case_sensitive=case_sensitive,
                    literal=literal,
                    show_hidden=show_hidden,
                    include_noise=include_noise,
                )
                cursor_snapshot, start_offset = _decode_cursor(
                    cursor,
                    query_fingerprint,
                )
                if _claim_file(state):
                    state.files_skipped += 1
                    state.reasons.add(GrepIncompleteReason.FILE_ERROR)
            else:
                display_path = normalize_utf8(
                    workspace.display_path(opened.path.absolute)
                )
                query_fingerprint = _query_fingerprint(
                    opened.path.absolute,
                    opened.stat,
                    display_path=display_path,
                    pattern=pattern,
                    file_glob=file_glob,
                    case_sensitive=case_sensitive,
                    literal=literal,
                    show_hidden=show_hidden,
                    include_noise=include_noise,
                )
                cursor_snapshot, start_offset = _decode_cursor(
                    cursor,
                    query_fingerprint,
                )
                if _claim_file(state):
                    _search_open_file(
                        state,
                        opened.file,
                        opened.stat,
                        absolute_path=opened.path.absolute,
                        raw_key=os.fsencode(opened.path.absolute),
                    )
    elif stat.S_ISDIR(target_stat.st_mode):
        display_path = normalize_utf8(workspace.display_path(target.absolute))
        state = make_state()
        with ExitStack() as stack:
            try:
                opened = stack.enter_context(workspace.open_directory(target.absolute))
            except OSError, ValueError:
                query_fingerprint = _query_fingerprint(
                    target.absolute,
                    target_stat,
                    display_path=display_path,
                    pattern=pattern,
                    file_glob=file_glob,
                    case_sensitive=case_sensitive,
                    literal=literal,
                    show_hidden=show_hidden,
                    include_noise=include_noise,
                )
                cursor_snapshot, start_offset = _decode_cursor(
                    cursor,
                    query_fingerprint,
                )
                state.reasons.add(GrepIncompleteReason.ENTRY_ERROR)
            else:
                display_path = normalize_utf8(
                    workspace.display_path(opened.path.absolute)
                )
                query_fingerprint = _query_fingerprint(
                    opened.path.absolute,
                    opened.stat,
                    display_path=display_path,
                    pattern=pattern,
                    file_glob=file_glob,
                    case_sensitive=case_sensitive,
                    literal=literal,
                    show_hidden=show_hidden,
                    include_noise=include_noise,
                )
                cursor_snapshot, start_offset = _decode_cursor(
                    cursor,
                    query_fingerprint,
                )
                state.visited_directories.add((opened.stat.st_dev, opened.stat.st_ino))
                _walk_directory(
                    state,
                    directory_fd=opened.fd,
                    absolute_path=opened.path.absolute,
                    relative_prefix="",
                    depth=1,
                )
    else:
        raise FileNotFoundError(
            f"Path is not a regular file or directory: {target.absolute}"
        )

    _check_deadline(state)
    return _finish_search(
        pattern=pattern,
        path=display_path,
        state=state,
        query_fingerprint=query_fingerprint,
        cursor=cursor,
        cursor_snapshot=cursor_snapshot,
        start_offset=start_offset,
        limit=effective_limit,
        max_output_bytes=ctx.deps.limits.max_output_bytes,
    )
