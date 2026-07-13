import asyncio
from collections.abc import Callable
import os
from pathlib import Path
import threading

from pydantic_ai import ModelRetry
import pytest

from ghostwheel.tools.edit import EditCommittedDuringCancellation, edit
from ghostwheel.tools.workspace import AtomicRewriteCancelled, AtomicRewriteResult

from .support import edit_metadata, tool_ctx


def _run_edit(
    ctx: object,
    path: str,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool = False,
):
    return asyncio.run(
        edit(
            ctx,  # type: ignore[arg-type]
            path,
            old_string,
            new_string,
            replace_all=replace_all,
        )
    )


def test_edit_replaces_one_exact_match_and_preserves_mode(tmp_path: Path) -> None:
    target = tmp_path / "module.py"
    target.write_text("before\nvalue = 1\nafter\n", encoding="utf-8")
    target.chmod(0o755)

    result = _run_edit(tool_ctx(tmp_path), "module.py", "value = 1", "value = 2")
    metadata = edit_metadata(result)

    assert target.read_text(encoding="utf-8") == "before\nvalue = 2\nafter\n"
    assert os.stat(target).st_mode & 0o777 == 0o755
    assert result.return_value == (
        'edited path="module.py" replacements=1 bytes=23->23'
    )
    assert metadata.path == "module.py"
    assert metadata.replacements == 1
    assert metadata.bytes_before == metadata.bytes_after == 23
    assert metadata.added_lines == 1
    assert metadata.removed_lines == 1
    assert metadata.applied is True
    assert metadata.dry_run is False
    assert metadata.durable is True
    assert metadata.warning is None
    assert metadata.summary == "1 replacement, +1 −1"


def test_edit_supports_multiline_unicode_deletion_and_crlf(tmp_path: Path) -> None:
    target = tmp_path / "windows.txt"
    target.write_bytes("start\r\nold 👻\r\nremove me\r\nend\r\n".encode())

    result = _run_edit(
        tool_ctx(tmp_path),
        "windows.txt",
        "old 👻\nremove me\n",
        "new ✨\n",
    )

    assert target.read_bytes() == "start\r\nnew ✨\r\nend\r\n".encode()
    assert edit_metadata(result).replacements == 1


def test_edit_targets_literal_carriage_return_before_crlf(tmp_path: Path) -> None:
    target = tmp_path / "literal-cr.txt"
    target.write_bytes(b"keep\r\r\nold\r\n")

    _run_edit(
        tool_ctx(tmp_path),
        "literal-cr.txt",
        "keep\r\nold",
        "kept\r\nnew",
    )

    assert target.read_bytes() == b"kept\r\r\nnew\r\n"


def test_edit_accepts_raw_crlf_arguments_as_a_fallback(tmp_path: Path) -> None:
    target = tmp_path / "raw-crlf.txt"
    target.write_bytes(b"old\r\nline\r\n")

    _run_edit(
        tool_ctx(tmp_path),
        "raw-crlf.txt",
        "old\r\nline",
        "new\r\nline",
    )

    assert target.read_bytes() == b"new\r\nline\r\n"


@pytest.mark.parametrize(
    ("source", "old_string", "expected"),
    [
        (
            b"old\r\nsuffix\r\n",
            "old",
            b"new\r\ninserted\r\nsuffix\r\n",
        ),
        (
            b"old\r\nline\r\nsuffix\r\n",
            "old\nline",
            b"new\r\ninserted\r\nsuffix\r\n",
        ),
    ],
)
def test_edit_normalizes_raw_crlf_replacement_without_literal_cr_evidence(
    tmp_path: Path,
    source: bytes,
    old_string: str,
    expected: bytes,
) -> None:
    target = tmp_path / "insert-crlf.txt"
    target.write_bytes(source)

    _run_edit(
        tool_ctx(tmp_path),
        "insert-crlf.txt",
        old_string,
        "new\r\ninserted",
    )

    assert target.read_bytes() == expected


def test_edit_requires_a_unique_match_by_default(tmp_path: Path) -> None:
    target = tmp_path / "repeated.txt"
    original = b"aaa"
    target.write_bytes(original)

    with pytest.raises(ModelRetry, match="matches more than once"):
        _run_edit(tool_ctx(tmp_path), "repeated.txt", "aa", "b")

    assert target.read_bytes() == original


def test_edit_missing_match_requests_a_fresh_read(tmp_path: Path) -> None:
    target = tmp_path / "value.txt"
    target.write_text("current", encoding="utf-8")

    with pytest.raises(ModelRetry, match="was not found.*read the file again"):
        _run_edit(tool_ctx(tmp_path), "value.txt", "stale", "new")

    assert target.read_text(encoding="utf-8") == "current"


def test_edit_replace_all_is_explicit_and_reports_count(tmp_path: Path) -> None:
    target = tmp_path / "values.txt"
    target.write_text("old\nkeep\nold\n", encoding="utf-8")

    result = _run_edit(
        tool_ctx(tmp_path),
        "values.txt",
        "old",
        "new",
        replace_all=True,
    )

    assert target.read_text(encoding="utf-8") == "new\nkeep\nnew\n"
    metadata = edit_metadata(result)
    assert metadata.replacements == 2
    assert metadata.summary == "2 replacements, +2 −2"


def test_edit_dry_run_validates_without_writing(tmp_path: Path) -> None:
    target = tmp_path / "value.txt"
    target.write_text("old", encoding="utf-8")

    result = _run_edit(
        tool_ctx(tmp_path, dry_run=True),
        "value.txt",
        "old",
        "new",
    )

    assert target.read_text(encoding="utf-8") == "old"
    metadata = edit_metadata(result)
    assert metadata.applied is False
    assert metadata.dry_run is True
    assert metadata.durable is None
    assert metadata.summary == "dry run: 1 replacement, +1 −1"
    assert result.return_value.startswith("edit dry-run ")


@pytest.mark.parametrize(
    ("old_string", "new_string", "error_type", "message"),
    [
        ("", "new", ValueError, "old_string must not be empty"),
        ("same", "same", ModelRetry, "must be different"),
    ],
)
def test_edit_rejects_invalid_or_noop_replacements(
    tmp_path: Path,
    old_string: str,
    new_string: str,
    error_type: type[Exception],
    message: str,
) -> None:
    (tmp_path / "value.txt").write_text("same", encoding="utf-8")

    with pytest.raises(error_type, match=message):
        _run_edit(tool_ctx(tmp_path), "value.txt", old_string, new_string)


def test_edit_rejects_noop_after_crlf_normalization(tmp_path: Path) -> None:
    target = tmp_path / "windows.txt"
    target.write_bytes(b"same\r\n")

    with pytest.raises(ModelRetry, match="must be different"):
        _run_edit(tool_ctx(tmp_path), "windows.txt", "same\r\n", "same\n")

    assert target.read_bytes() == b"same\r\n"


def test_edit_rejects_nul_in_replacement(tmp_path: Path) -> None:
    target = tmp_path / "value.txt"
    target.write_text("old", encoding="utf-8")

    with pytest.raises(ModelRetry, match="must not contain NUL"):
        _run_edit(tool_ctx(tmp_path), "value.txt", "old", "new\0value")

    assert target.read_text(encoding="utf-8") == "old"


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (b"prefix\0suffix", "binary file"),
        (b"\xff", "not valid UTF-8"),
    ],
)
def test_edit_rejects_non_text_files_without_changes(
    tmp_path: Path,
    contents: bytes,
    message: str,
) -> None:
    target = tmp_path / "value.bin"
    target.write_bytes(contents)

    with pytest.raises(ValueError, match=message):
        _run_edit(tool_ctx(tmp_path), "value.bin", "prefix", "new")

    assert target.read_bytes() == contents


def test_edit_enforces_source_and_result_size_limit(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("12345", encoding="utf-8")
    with pytest.raises(ValueError, match="size limit"):
        _run_edit(
            tool_ctx(tmp_path, max_read_scan_bytes=4),
            "source.txt",
            "1",
            "x",
        )

    result = tmp_path / "result.txt"
    result.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="size limit"):
        _run_edit(
            tool_ctx(tmp_path, max_read_scan_bytes=3),
            "result.txt",
            "x",
            "1234",
        )

    assert source.read_text(encoding="utf-8") == "12345"
    assert result.read_text(encoding="utf-8") == "x"


def test_edit_rejects_paths_outside_workspace_and_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (root / "linked.txt").symlink_to(outside)
    ctx = tool_ctx(root)

    with pytest.raises(ValueError, match="outside allowed roots"):
        _run_edit(ctx, "../outside.txt", "secret", "changed")
    with pytest.raises(OSError):
        _run_edit(ctx, "linked.txt", "secret", "changed")

    assert outside.read_text(encoding="utf-8") == "secret"


def test_edit_cancellation_waits_for_worker_and_prevents_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = tool_ctx(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    worker_finished = threading.Event()
    committed = threading.Event()
    callbacks: list[Callable[[], bool]] = []

    def blocking_rewrite(
        _workspace: object,
        _path: object,
        _transform: object,
        *,
        max_bytes: int,
        dry_run: bool,
        cancelled: Callable[[], bool] | None = None,
    ) -> object:
        assert max_bytes > 0
        assert dry_run is False
        assert cancelled is not None
        callbacks.append(cancelled)
        entered.set()
        try:
            assert release.wait(timeout=2)
            if cancelled():
                raise AtomicRewriteCancelled("cancelled before commit")
            committed.set()
            raise AssertionError("edit worker reached commit after cancellation")
        finally:
            worker_finished.set()

    monkeypatch.setattr(
        type(ctx.deps.workspace),
        "atomic_rewrite_regular_file",
        blocking_rewrite,
    )

    async def cancel_edit() -> None:
        task = asyncio.create_task(edit(ctx, "value.txt", "old", "new"))
        try:
            async with asyncio.timeout(2):
                while not entered.is_set():
                    await asyncio.sleep(0)
                task.cancel()
                while not callbacks[0]():
                    await asyncio.sleep(0)
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
        finally:
            release.set()

    asyncio.run(cancel_edit())

    assert worker_finished.is_set()
    assert not committed.is_set()


def test_edit_reports_when_commit_wins_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "value.txt"
    target.write_text("old", encoding="utf-8")
    ctx = tool_ctx(tmp_path)
    committed = threading.Event()
    release = threading.Event()
    worker_finished = threading.Event()
    callbacks: list[Callable[[], bool]] = []
    warning = "directory sync failed"

    def committed_rewrite(
        _workspace: object,
        path: object,
        transform: Callable[[bytes], bytes],
        *,
        max_bytes: int,
        dry_run: bool,
        cancelled: Callable[[], bool] | None = None,
    ) -> AtomicRewriteResult:
        assert max_bytes > 0
        assert dry_run is False
        assert cancelled is not None
        callbacks.append(cancelled)
        source = target.read_bytes()
        replacement = transform(source)
        target.write_bytes(replacement)
        committed.set()
        try:
            assert release.wait(timeout=2)
            assert cancelled()
            return AtomicRewriteResult(
                path=ctx.deps.workspace.locate(path),  # type: ignore[arg-type]
                bytes_before=len(source),
                bytes_after=len(replacement),
                changed=True,
                committed=True,
                durable=False,
                warning=warning,
            )
        finally:
            worker_finished.set()

    monkeypatch.setattr(
        type(ctx.deps.workspace),
        "atomic_rewrite_regular_file",
        committed_rewrite,
    )

    async def cancel_after_commit() -> EditCommittedDuringCancellation:
        task = asyncio.create_task(edit(ctx, "value.txt", "old", "new"))
        try:
            async with asyncio.timeout(2):
                while not committed.is_set():
                    await asyncio.sleep(0)
                task.cancel()
                while not callbacks[0]():
                    await asyncio.sleep(0)
                release.set()
                with pytest.raises(EditCommittedDuringCancellation) as raised:
                    await task
                return raised.value
        finally:
            release.set()

    error = asyncio.run(cancel_after_commit())

    assert target.read_text(encoding="utf-8") == "new"
    assert worker_finished.is_set()
    assert error.path == "value.txt"
    assert error.warning == warning
    assert "value.txt" in str(error)
    assert f"Durability warning: {warning}" in str(error)
    assert "inspect git diff" in str(error).lower()
    assert "do not retry blindly" in str(error).lower()
