"""Exact, atomic text replacement for existing workspace files."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import threading
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.messages import ToolReturn

from .deps import ToolDeps
from .output import normalize_utf8
from .workspace import AtomicRewriteCancelled, ConcurrentFileChange

_DIFF_STAT_MAX_BYTES = 1_000_000
_DIFF_STAT_MAX_LINES = 20_000


class EditResult(BaseModel):
    path: str
    replacements: int
    bytes_before: int
    bytes_after: int
    added_lines: int | None
    removed_lines: int | None
    applied: bool
    dry_run: bool
    durable: bool | None
    warning: str | None
    summary: str


class EditCommittedDuringCancellation(RuntimeError):
    """Report an edit that committed before cancellation could stop it."""

    def __init__(self, path: str, *, warning: str | None = None) -> None:
        self.path = path
        self.warning = warning
        message = (
            "Edit committed during cancellation for path "
            f"{json.dumps(path, ensure_ascii=False)}."
        )
        if warning is not None:
            message += f" Durability warning: {warning}."
        message += " Inspect git diff and do not retry blindly."
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class _PreparedEdit:
    replacement: bytes
    replacements: int
    bytes_before: int
    bytes_after: int
    added_lines: int | None
    removed_lines: int | None


def _logical_text(value: str) -> tuple[str, str | None]:
    """Normalize a consistently CRLF file for matching read-tool output."""

    without_crlf = value.replace("\r\n", "")
    if "\r\n" in value and "\n" not in without_crlf:
        return value.replace("\r\n", "\n"), "\r\n"
    return value, None


def _restore_newlines(value: str, newline: str | None) -> str:
    if newline is None:
        return value
    return value.replace("\n", newline)


def _line_change_counts(before: str, after: str) -> tuple[int | None, int | None]:
    if len(before.encode("utf-8")) + len(after.encode("utf-8")) > (
        _DIFF_STAT_MAX_BYTES
    ):
        return None, None
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    if len(before_lines) + len(after_lines) > _DIFF_STAT_MAX_LINES:
        return None, None
    added = 0
    removed = 0
    for operation, before_start, before_end, after_start, after_end in SequenceMatcher(
        None, before_lines, after_lines
    ).get_opcodes():
        if operation in {"replace", "delete"}:
            removed += before_end - before_start
        if operation in {"replace", "insert"}:
            added += after_end - after_start
    return added, removed


def _summary(
    replacements: int,
    *,
    added_lines: int | None,
    removed_lines: int | None,
    dry_run: bool,
    warning: bool,
) -> str:
    noun = "replacement" if replacements == 1 else "replacements"
    value = f"{replacements} {noun}"
    if added_lines is not None and removed_lines is not None:
        value += f", +{added_lines} −{removed_lines}"
    if warning:
        value += ", sync warning"
    return f"dry run: {value}" if dry_run else value


async def edit(
    ctx: RunContext[ToolDeps],
    path: str,
    old_string: Annotated[str, Field(min_length=1)],
    new_string: str,
    *,
    replace_all: bool = False,
) -> ToolReturn[str]:
    """Replace exact text in an existing UTF-8 workspace file.

    Args:
        path: Path to an existing file, relative to the working directory.
        old_string: Exact text to replace. Read the file first and include enough
            unchanged surrounding text to make this unique.
        new_string: Replacement text. Use an empty string to delete the match.
        replace_all: Replace every non-overlapping match instead of requiring one
            unique match. Defaults to False.

    The tool never creates files, follows symlinks, or guesses at a partial match.
    It preserves consistent CRLF line endings and commits through an atomic
    workspace rewrite. If the file changes during the operation, re-read it and
    retry with current text.
    """
    if not old_string:
        raise ValueError("old_string must not be empty")
    if old_string == new_string:
        raise ModelRetry("old_string and new_string must be different")
    if "\0" in new_string:
        raise ModelRetry("new_string must not contain NUL")

    cancellation = threading.Event()
    worker = asyncio.create_task(
        asyncio.to_thread(
            _edit_sync,
            ctx,
            path,
            old_string,
            new_string,
            replace_all=replace_all,
            cancelled=cancellation.is_set,
        )
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancellation.set()
        completed = await _drain_cancelled_worker(worker)
        if completed is not None:
            metadata = completed.metadata
            if isinstance(metadata, EditResult) and metadata.applied:
                raise EditCommittedDuringCancellation(
                    metadata.path,
                    warning=metadata.warning,
                ) from None
        raise


async def _drain_cancelled_worker(
    worker: asyncio.Task[ToolReturn[str]],
) -> ToolReturn[str] | None:
    """Wait for the mutation worker and return any completed tool result."""

    while not worker.done():
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    if worker.cancelled():
        return None
    try:
        return worker.result()
    except asyncio.CancelledError:
        return None
    except AtomicRewriteCancelled:
        return None
    except Exception:
        # The caller's cancellation remains authoritative, but retrieving the
        # worker exception prevents an unobserved-task warning.
        return None


def _edit_sync(
    ctx: RunContext[ToolDeps],
    path: str,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool,
    cancelled: Callable[[], bool],
) -> ToolReturn[str]:
    """Run the descriptor-bound edit in a worker thread."""

    prepared: _PreparedEdit | None = None

    def transform(source: bytes) -> bytes:
        nonlocal prepared
        if b"\0" in source:
            raise ValueError(f"Cannot edit binary file: {path}")
        try:
            decoded = source.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError(f"File is not valid UTF-8: {path}") from error

        logical, newline = _logical_text(decoded)
        logical_old = old_string
        logical_new = new_string
        first_match = logical.find(logical_old)
        if newline is not None:
            if first_match >= 0 and "\r\n" not in old_string:
                # Only an exact CR+logical-newline boundary in old_string proves
                # that replacement CRLF pairs should preserve a literal CR.
                # Otherwise prefer native CRLF compatibility; callers needing
                # literal-CR semantics can include that boundary in the exact
                # surrounding context.
                logical_new = new_string.replace("\r\n", "\n")
            elif first_match < 0:
                # Prefer the read tool's logical representation so a literal CR
                # at a CRLF boundary remains targetable. Raw CRLF arguments are
                # a fallback for callers that preserve file newlines.
                folded_old = old_string.replace("\r\n", "\n")
                if folded_old != old_string:
                    logical_old = folded_old
                    logical_new = new_string.replace("\r\n", "\n")
                    first_match = logical.find(logical_old)
        if logical_old == logical_new:
            raise ModelRetry("old_string and new_string must be different")

        if first_match < 0:
            raise ModelRetry(f"old_string was not found in {path}; read the file again")
        if replace_all:
            replacements = logical.count(logical_old)
            updated = logical.replace(logical_old, logical_new)
        else:
            if logical.find(logical_old, first_match + 1) >= 0:
                raise ModelRetry(
                    f"old_string matches more than once in {path}; "
                    "include more unchanged context or set replace_all=true"
                )
            replacements = 1
            updated = (
                logical[:first_match]
                + logical_new
                + logical[first_match + len(logical_old) :]
            )

        restored = _restore_newlines(updated, newline)
        try:
            replacement = restored.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise ValueError("new_string must contain valid Unicode") from error
        added_lines, removed_lines = _line_change_counts(decoded, restored)
        prepared = _PreparedEdit(
            replacement=replacement,
            replacements=replacements,
            bytes_before=len(source),
            bytes_after=len(replacement),
            added_lines=added_lines,
            removed_lines=removed_lines,
        )
        return replacement

    try:
        rewrite = ctx.deps.workspace.atomic_rewrite_regular_file(
            path,
            transform,
            max_bytes=ctx.deps.limits.max_read_scan_bytes,
            dry_run=ctx.deps.dry_run,
            cancelled=cancelled,
        )
    except ConcurrentFileChange as error:
        raise ModelRetry(f"{path} changed; read the file again and retry") from error

    if prepared is None:
        raise RuntimeError("edit transform did not run")
    display_path = normalize_utf8(
        ctx.deps.workspace.display_path(rewrite.path.absolute)
    )
    summary = _summary(
        prepared.replacements,
        added_lines=prepared.added_lines,
        removed_lines=prepared.removed_lines,
        dry_run=ctx.deps.dry_run,
        warning=rewrite.warning is not None,
    )
    warning = normalize_utf8(rewrite.warning) if rewrite.warning is not None else None
    metadata = EditResult(
        path=display_path,
        replacements=prepared.replacements,
        bytes_before=prepared.bytes_before,
        bytes_after=prepared.bytes_after,
        added_lines=prepared.added_lines,
        removed_lines=prepared.removed_lines,
        applied=rewrite.committed,
        dry_run=ctx.deps.dry_run,
        durable=rewrite.durable,
        warning=warning,
        summary=summary,
    )
    action = "edited" if rewrite.committed else "edit dry-run"
    rendered = (
        f"{action} path={json.dumps(display_path, ensure_ascii=False)} "
        f"replacements={prepared.replacements} "
        f"bytes={prepared.bytes_before}->{prepared.bytes_after}"
    )
    if warning is not None:
        rendered += f" warning={json.dumps(warning, ensure_ascii=False)}"
    return ToolReturn[str](return_value=rendered, metadata=metadata)
