"""Bounded, paginated directory-listing tool."""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_ai.messages import ToolReturn

from .deps import ToolDeps
from .output import normalize_utf8
from .pagination import decode_offset_cursor, encode_fingerprint, encode_offset_cursor
from .path_filters import NOISE_DIRECTORY_NAMES, glob_matches
from .workspace import OpenedWorkspaceDirectory, Workspace


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
    workspace: Workspace,
    *,
    directory: OpenedWorkspaceDirectory,
    relative_prefix: str,
    level: int,
    scan: _ListingScan,
) -> None:
    batch: list[tuple[str, str, FileKind, int | None, bool]] = []
    try:
        with workspace.scan_directory(directory) as children:
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
            with workspace.open_child_directory(directory, child_name) as opened_child:
                _scan_directory(
                    workspace,
                    directory=opened_child,
                    relative_prefix=relative_name,
                    level=level + 1,
                    scan=scan,
                )
        except OSError:
            # A child may disappear or be replaced after scandir. Reopening it
            # through Workspace preserves O_NOFOLLOW and allowed-root checks.
            scan.reasons.add(ListingIncompleteReason.ENTRY_ERROR)


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
    return encode_fingerprint(digest.digest()[:12])


def _snapshot_fingerprint(candidates: list[_ListingCandidate]) -> str:
    digest = hashlib.sha256()
    for candidate in candidates:
        digest.update(len(candidate.key).to_bytes(4, "big"))
        digest.update(candidate.key)
        digest.update(candidate.type.value.encode("ascii"))
        if candidate.size is not None:
            digest.update(str(candidate.size).encode("ascii"))
    return encode_fingerprint(digest.digest()[:12])


def _encode_cursor(
    query_fingerprint: str,
    snapshot_fingerprint: str,
    offset: int,
) -> str:
    return encode_offset_cursor(
        "v2",
        query_fingerprint,
        snapshot_fingerprint,
        offset,
    )


def _decode_cursor(cursor: str | None, query_fingerprint: str) -> tuple[str, int]:
    return decode_offset_cursor(
        cursor,
        version="v2",
        query_fingerprint=query_fingerprint,
        tool_name="ls",
    )


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
            ctx.deps.workspace,
            directory=opened,
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
