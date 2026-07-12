import base64
import codecs
from dataclasses import dataclass, field
import hashlib
import json
import os
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Annotated, BinaryIO

from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_ai.messages import ToolReturn

from .deps import ToolDeps
from .output import normalize_utf8
from .path_filters import NOISE_DIRECTORY_NAMES, glob_matches


class ReadIncompleteReason(str, Enum):
    PAGE_LIMIT = "page_limit"
    OUTPUT_BUDGET = "output_budget"
    LINE_TRUNCATED = "line_truncated"


class ReadResult(BaseModel):
    path: str
    start_line: int
    end_line: int | None
    lines_returned: int
    total_bytes: int
    eof: bool
    complete: bool
    reasons: list[ReadIncompleteReason]
    truncated_lines: list[int]
    next_cursor: str | None = None

    @property
    def line_count(self) -> int:
        """Compatibility view of the former line_count field."""

        return self.lines_returned

    @property
    def truncated(self) -> bool:
        """Compatibility view of the former truncated field."""

        return not self.complete


_MAX_READ_CURSOR_LENGTH = 4_096
_READ_CHUNK_BYTES = 64 * 1_024
_READ_CURSOR_DIGEST_BYTES = 12
_READ_TRUNCATION_MARKER = "… [line truncated]"
_READ_REASON_ORDER = (
    ReadIncompleteReason.PAGE_LIMIT,
    ReadIncompleteReason.OUTPUT_BUDGET,
    ReadIncompleteReason.LINE_TRUNCATED,
)
_READ_ESCAPE_TRANSLATION = {
    **{
        codepoint: f"\\x{codepoint:02x}"
        for codepoint in (*range(0x20), *range(0x7F, 0xA0))
        if codepoint != 0x09
    },
    0x0D: "\\r",
    0x5C: "\\\\",
    0x2028: "\\u2028",
    0x2029: "\\u2029",
}


class _BinaryFileError(ValueError):
    pass


class _ReadScanLimitError(ValueError):
    pass


@dataclass(slots=True)
class _ReadScanBudget:
    limit: int
    file_size: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def charge(self, amount: int) -> None:
        if amount > self.remaining:
            raise _ReadScanLimitError("Read scan byte limit exceeded")
        self.used += amount

    def readline(self, file: BinaryIO) -> bytes:
        if not self.remaining:
            if file.tell() < self.file_size:
                raise _ReadScanLimitError("Read scan byte limit exceeded")
            return b""
        raw = file.readline(min(_READ_CHUNK_BYTES, self.remaining))
        self.charge(len(raw))
        if not raw and file.tell() < self.file_size:
            raise _ReadScanLimitError("Read scan byte limit exceeded")
        return raw


@dataclass(frozen=True, slots=True)
class _ReadLine:
    number: int
    text: str
    rendered: str
    end_offset: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class _FittedReadPage:
    rendered: str
    emitted: tuple[_ReadLine, ...]
    reasons: tuple[ReadIncompleteReason, ...]
    truncated_lines: tuple[int, ...]
    eof: bool
    next_cursor: str | None


def _read_digest(*components: bytes) -> str:
    digest = hashlib.sha256()
    for component in components:
        digest.update(len(component).to_bytes(8, "big"))
        digest.update(component)
    return digest.digest()[:_READ_CURSOR_DIGEST_BYTES].hex()


def _read_path_fingerprint(path: Path) -> str:
    return _read_digest(os.fsencode(path))


def _read_stat_components(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _read_stat_fingerprint(file_stat: os.stat_result) -> str:
    return _read_digest(
        *(str(value).encode("ascii") for value in _read_stat_components(file_stat))
    )


def _encode_read_cursor(
    path_fingerprint: str,
    stat_fingerprint: str,
    next_line: int,
    byte_offset: int,
) -> str:
    line_text = f"{next_line:x}"
    offset_text = f"{byte_offset:x}"
    checksum = _read_digest(
        path_fingerprint.encode("ascii"),
        stat_fingerprint.encode("ascii"),
        line_text.encode("ascii"),
        offset_text.encode("ascii"),
    )
    return (
        f"r1.{path_fingerprint}.{stat_fingerprint}.{line_text}.{offset_text}.{checksum}"
    )


def _decode_read_cursor(
    cursor: str,
    *,
    expected_path_fingerprint: str,
    expected_stat_fingerprint: str,
    file: BinaryIO,
    file_size: int,
    scan_budget: _ReadScanBudget,
) -> tuple[int, int]:
    if len(cursor) > _MAX_READ_CURSOR_LENGTH:
        raise ValueError("Invalid read cursor")
    parts = cursor.split(".")
    if len(parts) != 6 or parts[0] != "r1":
        raise ValueError("Invalid read cursor")
    (
        _version,
        path_fingerprint,
        stat_fingerprint,
        line_text,
        offset_text,
        checksum,
    ) = parts
    if (
        len(path_fingerprint) != _READ_CURSOR_DIGEST_BYTES * 2
        or len(stat_fingerprint) != _READ_CURSOR_DIGEST_BYTES * 2
        or len(checksum) != _READ_CURSOR_DIGEST_BYTES * 2
    ):
        raise ValueError("Invalid read cursor")
    try:
        int(path_fingerprint, 16)
        int(stat_fingerprint, 16)
        next_line = int(line_text, 16)
        byte_offset = int(offset_text, 16)
    except ValueError as error:
        raise ValueError("Invalid read cursor") from error
    expected_checksum = _read_digest(
        path_fingerprint.encode("ascii"),
        stat_fingerprint.encode("ascii"),
        line_text.encode("ascii"),
        offset_text.encode("ascii"),
    )
    if checksum != expected_checksum:
        raise ValueError("Invalid read cursor")
    if path_fingerprint != expected_path_fingerprint:
        raise ValueError("Mismatched read cursor for requested path")
    if stat_fingerprint != expected_stat_fingerprint:
        raise ValueError("The file changed; restart read pagination")
    if next_line < 1 or byte_offset < 0 or byte_offset >= file_size:
        raise ValueError("Invalid read cursor")
    if byte_offset == 0:
        if next_line != 1:
            raise ValueError("Invalid read cursor")
    else:
        scan_budget.charge(1)
        file.seek(byte_offset - 1)
        if file.read(1) != b"\n":
            raise ValueError("Invalid read cursor")
    file.seek(byte_offset)
    return next_line, byte_offset


def _append_retained_text(
    retained: list[str],
    decoded: str,
    *,
    retained_bytes: int,
    used_bytes: int,
) -> tuple[int, bool]:
    if not decoded:
        return used_bytes, False
    remaining = max(0, retained_bytes - used_bytes)
    encoded = decoded.encode("utf-8")
    if len(encoded) <= remaining:
        retained.append(decoded)
        return used_bytes + len(encoded), False
    if remaining:
        retained.append(encoded[:remaining].decode("utf-8", errors="ignore"))
    return retained_bytes, True


def _read_utf8_line(
    file: BinaryIO,
    *,
    retained_bytes: int,
    scan_budget: _ReadScanBudget,
) -> tuple[str, int, bool] | None:
    first = scan_budget.readline(file)
    if not first:
        return None
    if b"\0" in first:
        raise _BinaryFileError("File appears to be binary")

    ended_with_lf = first.endswith(b"\n")
    if ended_with_lf or file.tell() >= scan_budget.file_size:
        content = first[:-1] if ended_with_lf else first
        if ended_with_lf and content.endswith(b"\r"):
            content = content[:-1]
        decoded = content.decode("utf-8", errors="strict")
        if len(content) <= retained_bytes:
            return decoded, file.tell(), False
        retained = content[:retained_bytes].decode("utf-8", errors="ignore")
        return retained, file.tell(), True

    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    retained: list[str] = []
    used_bytes = 0
    truncated = False
    raw = first

    while True:
        ended_with_lf = raw.endswith(b"\n")
        decoded = decoder.decode(raw, final=ended_with_lf)
        used_bytes, clipped = _append_retained_text(
            retained,
            decoded,
            retained_bytes=retained_bytes,
            used_bytes=used_bytes,
        )
        truncated = truncated or clipped
        if ended_with_lf:
            break

        raw = scan_budget.readline(file)
        if not raw:
            decoded = decoder.decode(b"", final=True)
            used_bytes, clipped = _append_retained_text(
                retained,
                decoded,
                retained_bytes=retained_bytes,
                used_bytes=used_bytes,
            )
            truncated = truncated or clipped
            break
        if b"\0" in raw:
            raise _BinaryFileError("File appears to be binary")

    text = "".join(retained)
    if not truncated and ended_with_lf:
        text = text[:-1]
        if text.endswith("\r"):
            text = text[:-1]
    return text, file.tell(), truncated


def _escape_read_character(character: str) -> str:
    return _READ_ESCAPE_TRANSLATION.get(ord(character), character)


def _escape_read_line(value: str) -> str:
    return value.translate(_READ_ESCAPE_TRANSLATION)


def _render_read_row(line: _ReadLine) -> str:
    return line.rendered


def _ordered_read_reasons(
    reasons: set[ReadIncompleteReason],
) -> tuple[ReadIncompleteReason, ...]:
    return tuple(reason for reason in _READ_REASON_ORDER if reason in reasons)


def _render_read_page(
    *,
    path: str,
    total_bytes: int,
    emitted: tuple[_ReadLine, ...],
    rows: list[str],
    start_line: int,
    eof: bool,
    reasons: tuple[ReadIncompleteReason, ...],
    next_cursor: str | None,
) -> str:
    end_line = emitted[-1].number if emitted else None
    lines_text = f"{start_line}-{end_line}" if end_line is not None else "-"
    reason_text = ",".join(reason.value for reason in reasons) or "-"
    header = (
        f"read {json.dumps(normalize_utf8(path), ensure_ascii=True)} "
        f"lines={lines_text} returned={len(emitted)} bytes={total_bytes} "
        f"eof={'true' if eof else 'false'} "
        f"complete={'true' if eof and not reasons else 'false'} "
        f"reasons={reason_text}"
    )
    output = [header, *rows]
    if next_cursor is not None:
        output.append(f"next {json.dumps(next_cursor)}")
    return "\n".join(output)


def _fit_read_page(
    *,
    path: str,
    total_bytes: int,
    candidates: list[_ReadLine],
    start_line: int,
    source_has_more: bool,
    page_limit_reached: bool,
    collection_output_limited: bool,
    path_fingerprint: str,
    stat_fingerprint: str,
    max_output_bytes: int,
) -> _FittedReadPage:
    if not candidates:
        rendered = _render_read_page(
            path=path,
            total_bytes=total_bytes,
            emitted=(),
            rows=[],
            start_line=start_line,
            eof=True,
            reasons=(),
            next_cursor=None,
        )
        if len(rendered.encode("utf-8")) <= max_output_bytes:
            return _FittedReadPage(rendered, (), (), (), True, None)
        reason = (ReadIncompleteReason.OUTPUT_BUDGET,)
        minimal = "read incomplete reasons=output_budget"
        rendered = minimal if len(minimal.encode("utf-8")) <= max_output_bytes else ""
        return _FittedReadPage(rendered, (), reason, (), True, None)

    for count in range(len(candidates), 0, -1):
        emitted = tuple(candidates[:count])
        pending = count < len(candidates) or source_has_more
        reasons: set[ReadIncompleteReason] = set()
        if page_limit_reached:
            reasons.add(ReadIncompleteReason.PAGE_LIMIT)
        if collection_output_limited or count < len(candidates):
            reasons.add(ReadIncompleteReason.OUTPUT_BUDGET)
        truncated_lines = tuple(line.number for line in emitted if line.truncated)
        if truncated_lines:
            reasons.add(ReadIncompleteReason.LINE_TRUNCATED)
        ordered_reasons = _ordered_read_reasons(reasons)
        next_cursor = (
            _encode_read_cursor(
                path_fingerprint,
                stat_fingerprint,
                emitted[-1].number + 1,
                emitted[-1].end_offset,
            )
            if pending
            else None
        )
        rendered = _render_read_page(
            path=path,
            total_bytes=total_bytes,
            emitted=emitted,
            rows=[_render_read_row(line) for line in emitted],
            start_line=start_line,
            eof=not pending,
            reasons=ordered_reasons,
            next_cursor=next_cursor,
        )
        if len(rendered.encode("utf-8")) <= max_output_bytes:
            return _FittedReadPage(
                rendered,
                emitted,
                ordered_reasons,
                truncated_lines,
                not pending,
                next_cursor,
            )

    first = candidates[0]
    pending = len(candidates) > 1 or source_has_more
    reasons = {ReadIncompleteReason.OUTPUT_BUDGET, ReadIncompleteReason.LINE_TRUNCATED}
    if page_limit_reached:
        reasons.add(ReadIncompleteReason.PAGE_LIMIT)
    ordered_reasons = _ordered_read_reasons(reasons)
    next_cursor = (
        _encode_read_cursor(
            path_fingerprint,
            stat_fingerprint,
            first.number + 1,
            first.end_offset,
        )
        if pending
        else None
    )
    emitted = (first,)
    fixed_row = f"{first.number}:{_READ_TRUNCATION_MARKER}"
    fixed_rendered = _render_read_page(
        path=path,
        total_bytes=total_bytes,
        emitted=emitted,
        rows=[fixed_row],
        start_line=start_line,
        eof=not pending,
        reasons=ordered_reasons,
        next_cursor=next_cursor,
    )
    if len(fixed_rendered.encode("utf-8")) <= max_output_bytes:
        remaining = max_output_bytes - len(fixed_rendered.encode("utf-8"))
        retained_units: list[str] = []
        for character in first.text:
            unit = _escape_read_character(character)
            unit_bytes = len(unit.encode("utf-8"))
            if unit_bytes > remaining:
                break
            retained_units.append(unit)
            remaining -= unit_bytes
        row = f"{first.number}:{''.join(retained_units)}{_READ_TRUNCATION_MARKER}"
        rendered = _render_read_page(
            path=path,
            total_bytes=total_bytes,
            emitted=emitted,
            rows=[row],
            start_line=start_line,
            eof=not pending,
            reasons=ordered_reasons,
            next_cursor=next_cursor,
        )
        return _FittedReadPage(
            rendered,
            emitted,
            ordered_reasons,
            (first.number,),
            not pending,
            next_cursor,
        )

    omitted_cursor = next_cursor
    omitted = (
        f"read omitted=1 line={first.number} reasons="
        f"{','.join(reason.value for reason in ordered_reasons)}"
    )
    if omitted_cursor is not None:
        omitted += f"\nnext {json.dumps(omitted_cursor)}"
    if len(omitted.encode("utf-8")) <= max_output_bytes:
        return _FittedReadPage(
            omitted,
            (),
            ordered_reasons,
            (first.number,),
            not pending,
            omitted_cursor,
        )

    minimal = "read incomplete reasons=output_budget"
    rendered = minimal if len(minimal.encode("utf-8")) <= max_output_bytes else ""
    return _FittedReadPage(
        rendered,
        (),
        ordered_reasons,
        (first.number,),
        not pending,
        None,
    )


def _read_start_position(
    file: BinaryIO,
    *,
    start_line: int,
    scan_budget: _ReadScanBudget,
) -> None:
    for _line_number in range(1, start_line):
        if (
            _read_utf8_line(
                file,
                retained_bytes=0,
                scan_budget=scan_budget,
            )
            is None
        ):
            break


def read(
    ctx: RunContext[ToolDeps],
    path: str,
    *,
    start_line: Annotated[int, Field(ge=1)] = 1,
    limit: Annotated[int | None, Field(gt=0)] = None,
    cursor: Annotated[
        str | None,
        Field(max_length=_MAX_READ_CURSOR_LENGTH),
    ] = None,
) -> ToolReturn[str]:
    """Read a UTF-8 text file as a compact, line-numbered page.

    Args:
        path: Path to the file, relative to the working directory.
        start_line: First 1-based line to return. Defaults to 1.
        limit: Maximum lines for this page, capped by the configured read limit.
        cursor: Opaque continuation cursor returned by a previous read call.

    Use either start_line for random access or cursor for efficient sequential
    paging. Binary and invalid UTF-8 files are rejected rather than returned as
    replacement-character noise. Configured line, scan-byte, and output limits
    remain hard ceilings for every call.
    """
    if start_line < 1:
        raise ValueError("start_line must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if cursor is not None and start_line != 1:
        raise ValueError("start_line cannot be combined with cursor")
    if cursor is not None and len(cursor) > _MAX_READ_CURSOR_LENGTH:
        raise ValueError("Invalid read cursor")

    effective_limit = min(
        ctx.deps.limits.max_read_lines,
        ctx.deps.limits.max_read_lines if limit is None else limit,
    )
    max_bytes = ctx.deps.limits.max_output_bytes
    with ctx.deps.workspace.open_file(path) as opened:
        display_path = normalize_utf8(
            ctx.deps.workspace.display_path(opened.path.absolute)
        )
        path_fingerprint = _read_path_fingerprint(opened.path.absolute)
        stat_fingerprint = _read_stat_fingerprint(opened.stat)
        scan_budget = _ReadScanBudget(
            ctx.deps.limits.max_read_scan_bytes,
            opened.stat.st_size,
        )
        try:
            if cursor is None:
                current_line = start_line
                _read_start_position(
                    opened.file,
                    start_line=start_line,
                    scan_budget=scan_budget,
                )
            else:
                current_line, _ = _decode_read_cursor(
                    cursor,
                    expected_path_fingerprint=path_fingerprint,
                    expected_stat_fingerprint=stat_fingerprint,
                    file=opened.file,
                    file_size=opened.stat.st_size,
                    scan_budget=scan_budget,
                )

            candidates: list[_ReadLine] = []
            collected_row_bytes = 0
            collection_output_limited = False
            for _index in range(effective_limit):
                line = _read_utf8_line(
                    opened.file,
                    retained_bytes=max_bytes + 2,
                    scan_budget=scan_budget,
                )
                if line is None:
                    break
                text, end_offset, line_truncated = line
                marker = _READ_TRUNCATION_MARKER if line_truncated else ""
                candidates.append(
                    _ReadLine(
                        number=current_line,
                        text=text,
                        rendered=(f"{current_line}:{_escape_read_line(text)}{marker}"),
                        end_offset=end_offset,
                        truncated=line_truncated,
                    )
                )
                collected_row_bytes += (
                    len(_render_read_row(candidates[-1]).encode("utf-8")) + 1
                )
                current_line += 1
                if collected_row_bytes >= max_bytes:
                    collection_output_limited = True
                    break

            after_candidates = opened.file.tell()
            source_has_more = after_candidates < opened.stat.st_size
            page_limit_reached = (
                source_has_more
                and len(candidates) >= effective_limit
                and not collection_output_limited
            )
        except _BinaryFileError as error:
            raise ValueError(f"Cannot read binary file: {display_path}") from error
        except UnicodeDecodeError as error:
            raise ValueError(f"File is not valid UTF-8: {display_path}") from error
        except _ReadScanLimitError as error:
            raise ValueError(
                f"Read scan limit exceeded for {display_path} "
                f"({ctx.deps.limits.max_read_scan_bytes} bytes)"
            ) from error

        if _read_stat_components(os.fstat(opened.file.fileno())) != (
            _read_stat_components(opened.stat)
        ):
            raise ValueError("The file changed; restart read pagination")

        page = _fit_read_page(
            path=display_path,
            total_bytes=opened.stat.st_size,
            candidates=candidates,
            start_line=current_line - len(candidates),
            source_has_more=source_has_more,
            page_limit_reached=page_limit_reached,
            collection_output_limited=collection_output_limited,
            path_fingerprint=path_fingerprint,
            stat_fingerprint=stat_fingerprint,
            max_output_bytes=max_bytes,
        )

    metadata = ReadResult(
        path=display_path,
        start_line=current_line - len(candidates),
        end_line=page.emitted[-1].number if page.emitted else None,
        lines_returned=len(page.emitted),
        total_bytes=opened.stat.st_size,
        eof=page.eof,
        complete=page.eof and not page.reasons,
        reasons=list(page.reasons),
        truncated_lines=list(page.truncated_lines),
        next_cursor=page.next_cursor,
    )
    return ToolReturn[str](return_value=page.rendered, metadata=metadata)


class FileKind(str, Enum):
    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    OTHER = "other"


class ListingIncompleteReason(str, Enum):
    SCAN_LIMIT = "scan_limit"
    ENTRY_ERROR = "entry_error"
    ENTRY_LIMIT = "entry_limit"
    OUTPUT_BUDGET = "output_budget"


class DirEntry(BaseModel):
    name: str
    type: FileKind = Field(description="'file', 'directory', 'symlink', or 'other'")
    size: int | None = Field(
        default=None,
        description="File size in bytes when requested; otherwise None",
    )


class DirectoryListing(BaseModel):
    path: str = Field(
        description=("Workspace-relative path, or absolute path for an additional root")
    )
    depth: int
    entries: list[DirEntry]
    scanned: int = Field(description="Raw directory entries examined")
    skipped: int = Field(
        default=0,
        description="Entries skipped because one row exceeded the output budget",
    )
    complete: bool
    reasons: list[ListingIncompleteReason]
    next_cursor: str | None = None

    @property
    def truncated(self) -> bool:
        """Compatibility view of the former truncated field."""

        return not self.complete


_MAX_LS_DEPTH = 3
_MAX_CURSOR_LENGTH = 4_096
_REASON_ORDER = (
    ListingIncompleteReason.SCAN_LIMIT,
    ListingIncompleteReason.ENTRY_ERROR,
    ListingIncompleteReason.ENTRY_LIMIT,
    ListingIncompleteReason.OUTPUT_BUDGET,
)
_HARD_PAGINATION_REASONS = {
    ListingIncompleteReason.SCAN_LIMIT,
    ListingIncompleteReason.ENTRY_ERROR,
}
_KIND_MARKERS = {
    FileKind.FILE: "f",
    FileKind.DIRECTORY: "d",
    FileKind.SYMLINK: "l",
    FileKind.OTHER: "o",
}


@dataclass(frozen=True, slots=True)
class _ListingCandidate:
    display_name: str
    key: bytes
    type: FileKind
    size: int | None

    def as_entry(self) -> DirEntry:
        return DirEntry(name=self.display_name, type=self.type, size=self.size)


@dataclass(slots=True)
class _ListingScan:
    depth: int
    glob: str | None
    show_hidden: bool
    include_noise: bool
    include_size: bool
    scan_limit: int
    scanned: int = 0
    candidates: list[_ListingCandidate] = field(default_factory=list)
    reasons: set[ListingIncompleteReason] = field(default_factory=set)


def _classify_entry(
    child: os.DirEntry[str],
) -> FileKind:
    if child.is_symlink():
        return FileKind.SYMLINK
    if child.is_dir(follow_symlinks=False):
        return FileKind.DIRECTORY
    if child.is_file(follow_symlinks=False):
        return FileKind.FILE
    return FileKind.OTHER


def _join_relative(prefix: str, name: str) -> str:
    return f"{prefix}/{name}" if prefix else name


def _matches_glob(relative_name: str, pattern: str | None) -> bool:
    if pattern is None:
        return True
    return glob_matches(PurePosixPath(relative_name), pattern)


def _scan_directory(
    ctx: RunContext[ToolDeps],
    *,
    directory_fd: int,
    absolute_path: Path,
    relative_prefix: str,
    level: int,
    scan: _ListingScan,
) -> None:
    batch: list[tuple[str, str, FileKind, int | None, bool]] = []
    try:
        with os.scandir(directory_fd) as children:
            for child in children:
                if scan.scanned >= scan.scan_limit:
                    scan.reasons.add(ListingIncompleteReason.SCAN_LIMIT)
                    break
                scan.scanned += 1
                if not scan.show_hidden and child.name.startswith("."):
                    continue
                try:
                    entry_type = _classify_entry(child)
                    if (
                        not scan.include_noise
                        and entry_type is FileKind.DIRECTORY
                        and child.name in NOISE_DIRECTORY_NAMES
                    ):
                        continue
                    relative_name = _join_relative(relative_prefix, child.name)
                    matches = _matches_glob(relative_name, scan.glob)
                    size = (
                        child.stat(follow_symlinks=False).st_size
                        if matches and scan.include_size and entry_type is FileKind.FILE
                        else None
                    )
                except OSError:
                    scan.reasons.add(ListingIncompleteReason.ENTRY_ERROR)
                    continue
                batch.append((child.name, relative_name, entry_type, size, matches))
    except OSError:
        scan.reasons.add(ListingIncompleteReason.ENTRY_ERROR)

    batch.sort(key=lambda item: os.fsencode(item[1]))
    for child_name, relative_name, entry_type, size, matches in batch:
        if matches:
            scan.candidates.append(
                _ListingCandidate(
                    display_name=normalize_utf8(relative_name),
                    key=os.fsencode(relative_name),
                    type=entry_type,
                    size=size,
                )
            )

        if entry_type is not FileKind.DIRECTORY or level >= scan.depth:
            continue
        if ListingIncompleteReason.SCAN_LIMIT in scan.reasons:
            # Every item in this batch was already charged to the shared scan
            # budget. Keep those rows, but do not descend any further.
            continue
        try:
            with ctx.deps.workspace.open_directory(
                absolute_path / child_name
            ) as opened_child:
                _scan_directory(
                    ctx,
                    directory_fd=opened_child.fd,
                    absolute_path=opened_child.path.absolute,
                    relative_prefix=relative_name,
                    level=level + 1,
                    scan=scan,
                )
        except OSError:
            # A child may disappear or be replaced after scandir. Reopening it
            # through Workspace preserves O_NOFOLLOW and allowed-root checks.
            scan.reasons.add(ListingIncompleteReason.ENTRY_ERROR)


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
        raise ValueError("Invalid ls cursor") from error


def _query_fingerprint(
    absolute_path: Path,
    *,
    device: int,
    inode: int,
    depth: int,
    glob: str | None,
    show_hidden: bool,
    include_noise: bool,
    include_size: bool,
) -> str:
    digest = hashlib.sha256()
    components = (
        os.fsencode(absolute_path),
        str(device).encode("ascii"),
        str(inode).encode("ascii"),
        str(depth).encode("ascii"),
        b"none" if glob is None else b"glob:" + os.fsencode(glob),
        b"hidden:1" if show_hidden else b"hidden:0",
        b"noise:1" if include_noise else b"noise:0",
        b"size:1" if include_size else b"size:0",
    )
    for component in components:
        digest.update(len(component).to_bytes(8, "big"))
        digest.update(component)
    return _urlsafe_encode(digest.digest()[:12])


def _snapshot_fingerprint(candidates: list[_ListingCandidate]) -> str:
    digest = hashlib.sha256()
    for candidate in candidates:
        digest.update(len(candidate.key).to_bytes(4, "big"))
        digest.update(candidate.key)
        digest.update(candidate.type.value.encode("ascii"))
        if candidate.size is not None:
            digest.update(str(candidate.size).encode("ascii"))
    return _urlsafe_encode(digest.digest()[:12])


def _encode_cursor(
    query_fingerprint: str,
    snapshot_fingerprint: str,
    offset: int,
) -> str:
    return f"v2.{query_fingerprint}.{snapshot_fingerprint}.{offset:x}"


def _decode_cursor(cursor: str | None, query_fingerprint: str) -> tuple[str, int]:
    if cursor is None:
        return "", 0
    if len(cursor) > _MAX_CURSOR_LENGTH:
        raise ValueError("Invalid ls cursor")
    parts = cursor.split(".")
    if len(parts) != 4 or parts[0] != "v2" or parts[1] != query_fingerprint:
        raise ValueError("Invalid or mismatched ls cursor")
    if len(_urlsafe_decode(parts[2])) != 12:
        raise ValueError("Invalid ls cursor")
    try:
        offset = int(parts[3], 16)
    except ValueError as error:
        raise ValueError("Invalid ls cursor") from error
    if offset <= 0:
        raise ValueError("Invalid ls cursor")
    return parts[2], offset


def _ordered_reasons(
    reasons: set[ListingIncompleteReason],
) -> list[ListingIncompleteReason]:
    return [reason for reason in _REASON_ORDER if reason in reasons]


def _quote(value: str) -> str:
    # ASCII JSON escapes also cover Unicode line separators and terminal
    # controls that could otherwise render as forged rows.
    return json.dumps(normalize_utf8(value), ensure_ascii=True)


def _render_entry(candidate: _ListingCandidate) -> str:
    rendered = f"{_KIND_MARKERS[candidate.type]} {_quote(candidate.display_name)}"
    if candidate.size is not None:
        rendered += f" {candidate.size}"
    return rendered


def _render_listing(
    *,
    path: str,
    depth: int,
    scanned: int,
    entries: list[_ListingCandidate],
    skipped: int,
    reasons: list[ListingIncompleteReason],
    next_cursor: str | None,
) -> str:
    reason_text = ",".join(reason.value for reason in reasons) or "-"
    skipped_text = f" skipped={skipped}" if skipped else ""
    lines = [
        f"ls {_quote(path)} depth={depth} returned={len(entries)}{skipped_text} "
        f"scanned={scanned} complete={'false' if reasons else 'true'} "
        f"reasons={reason_text}"
    ]
    lines.extend(_render_entry(entry) for entry in entries)
    if next_cursor is not None:
        lines.append(f"next {_quote(next_cursor)}")
    return "\n".join(lines)


def _fit_listing_page(
    *,
    path: str,
    depth: int,
    scanned: int,
    candidates: list[_ListingCandidate],
    limit: int,
    base_reasons: set[ListingIncompleteReason],
    query_fingerprint: str,
    snapshot_fingerprint: str,
    start_offset: int,
    max_output_bytes: int,
) -> tuple[
    str,
    list[_ListingCandidate],
    list[ListingIncompleteReason],
    str | None,
    int,
]:
    page = candidates[:limit]
    entry_limit_reached = len(candidates) > limit

    for count in range(len(page), -1, -1):
        emitted = page[:count]
        skipped = 0
        reasons = set(base_reasons)
        if entry_limit_reached:
            reasons.add(ListingIncompleteReason.ENTRY_LIMIT)
        if count < len(page):
            reasons.add(ListingIncompleteReason.OUTPUT_BUDGET)

        has_more = len(candidates) > count
        next_cursor = None
        if has_more and not reasons.intersection(_HARD_PAGINATION_REASONS):
            cursor_offset = start_offset + count
            if not emitted and page:
                # The first escaped row cannot fit this configured budget.
                # Advance past it explicitly instead of returning the same
                # non-progressing page forever.
                skipped = 1
                cursor_offset += 1
            if emitted or skipped:
                next_cursor = _encode_cursor(
                    query_fingerprint,
                    snapshot_fingerprint,
                    cursor_offset,
                )

        ordered_reasons = _ordered_reasons(reasons)
        rendered = _render_listing(
            path=path,
            depth=depth,
            scanned=scanned,
            entries=emitted,
            skipped=skipped,
            reasons=ordered_reasons,
            next_cursor=next_cursor,
        )
        if len(rendered.encode("utf-8")) <= max_output_bytes:
            return rendered, emitted, ordered_reasons, next_cursor, skipped

    reasons = set(base_reasons)
    if entry_limit_reached:
        reasons.add(ListingIncompleteReason.ENTRY_LIMIT)
    reasons.add(ListingIncompleteReason.OUTPUT_BUDGET)
    ordered_reasons = _ordered_reasons(reasons)
    reason_text = ",".join(reason.value for reason in ordered_reasons)
    minimal = f"ls incomplete reasons={reason_text}"
    rendered = minimal if len(minimal.encode("utf-8")) <= max_output_bytes else ""
    return rendered, [], ordered_reasons, None, 0


def ls(
    ctx: RunContext[ToolDeps],
    path: str = ".",
    *,
    depth: Annotated[int, Field(ge=1, le=_MAX_LS_DEPTH)] = 1,
    glob: Annotated[str | None, Field(min_length=1)] = None,
    limit: Annotated[int | None, Field(gt=0)] = None,
    cursor: str | None = None,
    show_hidden: bool = False,
    include_noise: bool = False,
    include_size: bool = False,
) -> ToolReturn[str]:
    """List a directory as compact, sorted text.

    Args:
        path: Directory to list, relative to the working directory. Defaults to '.'.
        depth: Recursive listing depth from 1 through 3. Defaults to 1.
        glob: Optional case-sensitive glob matched against relative result paths.
        limit: Maximum rows for this page, capped by the configured entry limit.
        cursor: Opaque continuation cursor returned by a previous matching call.
        show_hidden: If True, include entries starting with '.'. Defaults to False.
        include_noise: Include common dependency and cache directories.
        include_size: If True, include exact byte sizes for regular files.

    Rows use f/d/l/o markers for files, directories, symlinks, and other entries.
    The header reports completeness and any scan, entry, page, or output limit.
    """
    if depth < 1 or depth > _MAX_LS_DEPTH:
        raise ValueError(f"depth must be between 1 and {_MAX_LS_DEPTH}")
    if glob is not None and (not glob or PurePosixPath(glob).is_absolute()):
        raise ValueError("glob must be a non-empty relative pattern")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    effective_limit = min(
        ctx.deps.limits.max_entries,
        ctx.deps.limits.max_entries if limit is None else limit,
    )

    with ctx.deps.workspace.open_directory(path) as opened:
        display_path = normalize_utf8(
            ctx.deps.workspace.display_path(opened.path.absolute)
        )
        fingerprint = _query_fingerprint(
            opened.path.absolute,
            device=opened.stat.st_dev,
            inode=opened.stat.st_ino,
            depth=depth,
            glob=glob,
            show_hidden=show_hidden,
            include_noise=include_noise,
            include_size=include_size,
        )
        cursor_snapshot, start_offset = _decode_cursor(cursor, fingerprint)
        scan = _ListingScan(
            depth=depth,
            glob=glob,
            show_hidden=show_hidden,
            include_noise=include_noise,
            include_size=include_size,
            scan_limit=ctx.deps.limits.max_directory_scan_entries,
        )
        _scan_directory(
            ctx,
            directory_fd=opened.fd,
            absolute_path=opened.path.absolute,
            relative_prefix="",
            level=1,
            scan=scan,
        )

    scan.candidates.sort(key=lambda candidate: candidate.key)
    snapshot = _snapshot_fingerprint(scan.candidates)
    if cursor is not None and cursor_snapshot != snapshot:
        raise ValueError("The directory changed; restart ls pagination")
    if start_offset > len(scan.candidates):
        raise ValueError("Invalid ls cursor")
    remaining_candidates = scan.candidates[start_offset:]

    rendered, emitted, reasons, next_cursor, skipped = _fit_listing_page(
        path=display_path,
        depth=depth,
        scanned=scan.scanned,
        candidates=remaining_candidates,
        limit=effective_limit,
        base_reasons=scan.reasons,
        query_fingerprint=fingerprint,
        snapshot_fingerprint=snapshot,
        start_offset=start_offset,
        max_output_bytes=ctx.deps.limits.max_output_bytes,
    )
    metadata = DirectoryListing(
        path=display_path,
        depth=depth,
        entries=[candidate.as_entry() for candidate in emitted],
        scanned=scan.scanned,
        skipped=skipped,
        complete=not reasons,
        reasons=reasons,
        next_cursor=next_cursor,
    )
    return ToolReturn[str](return_value=rendered, metadata=metadata)
