import json
import os
from pathlib import Path

import pytest
import ghostwheel.tools.filesystem as filesystem_module
import ghostwheel.tools.workspace as workspace_module

from ghostwheel.tools.filesystem import FileKind, ListingIncompleteReason, ls

from .support import listing_metadata, tool_ctx


def test_ls_does_not_count_filtered_hidden_entries_toward_limit(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    for index in range(205):
        (root / f".hidden-{index:03d}").write_text("", encoding="utf-8")
    (root / "visible.txt").write_text("", encoding="utf-8")

    result = ls(tool_ctx(root), show_hidden=False)
    listing = listing_metadata(result)

    assert [entry.name for entry in listing.entries] == ["visible.txt"]
    assert listing.complete is True


def test_ls_bounds_scanning_of_filtered_hidden_entries(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f".hidden-{index}").write_text("", encoding="utf-8")

    result = ls(tool_ctx(tmp_path, max_directory_scan_entries=2))
    listing = listing_metadata(result)

    assert listing.entries == []
    assert listing.complete is False
    assert listing.reasons == [ListingIncompleteReason.SCAN_LIMIT]
    assert listing.next_cursor is None


def test_ls_uses_typed_kinds_and_configured_entry_limit(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")

    result = ls(tool_ctx(tmp_path, max_entries=1))
    listing = listing_metadata(result)

    assert len(listing.entries) == 1
    assert listing.entries[0].type is FileKind.FILE
    assert listing.complete is False
    assert listing.reasons == [ListingIncompleteReason.ENTRY_LIMIT]


def test_ls_returns_compact_sorted_and_escaped_rows(tmp_path: Path) -> None:
    names = [
        "z.txt",
        "line\nbreak.txt",
        'quote".txt',
        "slash\\.txt",
        "next\u0085line.txt",
        "separator\u2028line.txt",
        "paragraph\u2029line.txt",
        "é.txt",
    ]
    for name in names:
        (tmp_path / name).write_text("value", encoding="utf-8")

    result = ls(tool_ctx(tmp_path))
    listing = listing_metadata(result)

    assert isinstance(result.return_value, str)
    assert [entry.name for entry in listing.entries] == sorted(
        names,
        key=os.fsencode,
    )
    rows = result.return_value.splitlines()[1:]
    assert len(rows) == len(names)
    assert [json.loads(row[2:]) for row in rows] == [
        entry.name for entry in listing.entries
    ]
    assert "line\\nbreak.txt" in result.return_value
    assert listing.complete is True


def test_ls_normalizes_undecodable_names_before_json_escaping(tmp_path: Path) -> None:
    raw_name = b"bad-\xff.txt"
    try:
        descriptor = os.open(
            os.fsencode(tmp_path) + b"/" + raw_name,
            os.O_WRONLY | os.O_CREAT,
            0o600,
        )
    except OSError as error:
        pytest.skip(f"filesystem rejects undecodable names: {error}")
    os.close(descriptor)

    result = ls(tool_ctx(tmp_path))
    listing = listing_metadata(result)

    assert [entry.name for entry in listing.entries] == ["bad-\\udcff.txt"]
    assert json.loads(result.return_value.splitlines()[1][2:]) == "bad-\\udcff.txt"
    result.return_value.encode("utf-8", errors="strict")


def test_ls_paginates_deterministically_with_query_bound_cursor(
    tmp_path: Path,
) -> None:
    for name in reversed(["a", "b", "c", "d", "e"]):
        (tmp_path / name).write_text(name, encoding="utf-8")
    ctx = tool_ctx(tmp_path)

    first = ls(ctx, limit=2)
    first_listing = listing_metadata(first)
    second = ls(ctx, limit=2, cursor=first_listing.next_cursor)
    second_listing = listing_metadata(second)
    third = ls(ctx, limit=2, cursor=second_listing.next_cursor)
    third_listing = listing_metadata(third)

    assert [
        entry.name
        for listing in (first_listing, second_listing, third_listing)
        for entry in listing.entries
    ] == ["a", "b", "c", "d", "e"]
    assert first_listing.reasons == [ListingIncompleteReason.ENTRY_LIMIT]
    assert second_listing.reasons == [ListingIncompleteReason.ENTRY_LIMIT]
    assert third_listing.complete is True
    assert third_listing.next_cursor is None
    with pytest.raises(ValueError, match="mismatched ls cursor"):
        ls(ctx, limit=2, cursor=first_listing.next_cursor, show_hidden=True)


def test_ls_cursor_rejects_a_changed_directory_snapshot(tmp_path: Path) -> None:
    (tmp_path / "a").write_text("", encoding="utf-8")
    (tmp_path / "b").write_text("", encoding="utf-8")
    ctx = tool_ctx(tmp_path)
    first = listing_metadata(ls(ctx, limit=1))
    (tmp_path / "c").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="directory changed"):
        ls(ctx, limit=1, cursor=first.next_cursor)


def test_ls_glob_filters_recursive_results_without_pruning_traversal(
    tmp_path: Path,
) -> None:
    (tmp_path / "src" / "nested").mkdir(parents=True)
    (tmp_path / "src" / "main.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "nested" / "test.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "nested" / "notes.txt").write_text("", encoding="utf-8")

    shallow = listing_metadata(ls(tool_ctx(tmp_path), depth=1, glob="**/*.py"))
    recursive = listing_metadata(ls(tool_ctx(tmp_path), depth=3, glob="**/*.py"))
    simple_glob = listing_metadata(ls(tool_ctx(tmp_path), depth=3, glob="*.py"))

    assert shallow.entries == []
    assert [entry.name for entry in recursive.entries] == [
        "src/main.py",
        "src/nested/test.py",
    ]
    assert [entry.name for entry in simple_glob.entries] == [
        "src/main.py",
        "src/nested/test.py",
    ]


@pytest.mark.parametrize("glob", ["", "/"])
def test_ls_rejects_empty_or_absolute_globs(tmp_path: Path, glob: str) -> None:
    with pytest.raises(ValueError, match="non-empty relative pattern"):
        ls(tool_ctx(tmp_path), glob=glob)


def test_recursive_ls_prunes_common_noise_before_spending_scan_budget(
    tmp_path: Path,
) -> None:
    (tmp_path / "node_modules").mkdir()
    for index in range(5):
        (tmp_path / "node_modules" / f"dependency-{index}.js").write_text(
            "",
            encoding="utf-8",
        )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("", encoding="utf-8")
    ctx = tool_ctx(tmp_path, max_directory_scan_entries=4)

    pruned = listing_metadata(ls(ctx, depth=2, glob="*.py"))
    with_noise = listing_metadata(ls(ctx, depth=2, glob="*.py", include_noise=True))

    assert [entry.name for entry in pruned.entries] == ["src/app.py"]
    assert pruned.complete is True
    assert with_noise.entries == []
    assert with_noise.reasons == [ListingIncompleteReason.SCAN_LIMIT]


def test_ls_includes_regular_file_sizes_only_when_requested(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_bytes(b"12345")
    (tmp_path / "directory").mkdir()
    (tmp_path / "link").symlink_to("file.txt")

    compact_result = ls(tool_ctx(tmp_path))
    detailed_result = ls(tool_ctx(tmp_path), include_size=True)
    compact = listing_metadata(compact_result)
    detailed = listing_metadata(detailed_result)

    assert all(entry.size is None for entry in compact.entries)
    sizes = {entry.name: entry.size for entry in detailed.entries}
    assert sizes == {"directory": None, "file.txt": 5, "link": None}
    file_row = next(
        row for row in detailed_result.return_value.splitlines() if '"file.txt"' in row
    )
    assert file_row.endswith(" 5")


def test_ls_caps_the_exact_model_facing_output_without_partial_rows(
    tmp_path: Path,
) -> None:
    for index in range(20):
        (tmp_path / f"long-name-{index:02d}-{'x' * 30}.txt").write_text(
            "",
            encoding="utf-8",
        )

    result = ls(tool_ctx(tmp_path, max_output_bytes=220))
    listing = listing_metadata(result)

    assert len(result.return_value.encode("utf-8")) <= 220
    assert ListingIncompleteReason.OUTPUT_BUDGET in listing.reasons
    for row in result.return_value.splitlines()[1:]:
        if row.startswith("next "):
            json.loads(row.removeprefix("next "))
        else:
            json.loads(row[2:])
    assert listing.entries
    assert listing.next_cursor is not None


def test_ls_tiny_output_budget_never_emits_a_partial_entry(tmp_path: Path) -> None:
    (tmp_path / f"{'x' * 100}.txt").write_text("", encoding="utf-8")

    result = ls(tool_ctx(tmp_path, max_output_bytes=10))
    listing = listing_metadata(result)

    assert len(result.return_value.encode("utf-8")) <= 10
    assert "\n" not in result.return_value
    assert listing.entries == []
    assert listing.reasons == [ListingIncompleteReason.OUTPUT_BUDGET]
    assert listing.next_cursor is None


def test_ls_output_limited_page_can_advance_past_an_oversized_row(
    tmp_path: Path,
) -> None:
    (tmp_path / ("a" + "x" * 200)).write_text("", encoding="utf-8")
    (tmp_path / "b").write_text("", encoding="utf-8")
    ctx = tool_ctx(tmp_path, max_output_bytes=250)

    first = ls(ctx, limit=1)
    first_listing = listing_metadata(first)
    second = ls(ctx, limit=1, cursor=first_listing.next_cursor)
    second_listing = listing_metadata(second)

    assert first_listing.entries == []
    assert first_listing.skipped == 1
    assert first_listing.next_cursor is not None
    assert len(first.return_value.encode("utf-8")) <= 250
    assert [entry.name for entry in second_listing.entries] == ["b"]
    assert second_listing.complete is True


def test_ls_compact_payload_avoids_repeated_entry_keys(tmp_path: Path) -> None:
    for index in range(200):
        (tmp_path / f"module-{index:03d}.py").write_text("", encoding="utf-8")

    result = ls(tool_ctx(tmp_path))
    listing = listing_metadata(result)

    assert len(listing.entries) == 200
    assert listing.complete is True
    assert len(result.return_value.encode("utf-8")) < 5_000
    assert '"name":' not in result.return_value


def test_ls_classifies_special_files_as_other(tmp_path: Path) -> None:
    fifo = tmp_path / "events.fifo"
    os.mkfifo(fifo)

    listing = listing_metadata(ls(tool_ctx(tmp_path)))

    assert [(entry.name, entry.type) for entry in listing.entries] == [
        ("events.fifo", FileKind.OTHER)
    ]


def test_ls_reports_directory_iteration_errors_as_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_scandir(_fd: int):
        raise OSError("directory changed")

    monkeypatch.setattr(filesystem_module.os, "scandir", failing_scandir)

    listing = listing_metadata(ls(tool_ctx(tmp_path)))

    assert listing.entries == []
    assert listing.reasons == [ListingIncompleteReason.ENTRY_ERROR]
    assert listing.next_cursor is None


def test_recursive_ls_never_follows_a_symlinked_directory(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET", encoding="utf-8")
    (root / "linked").symlink_to(outside, target_is_directory=True)

    result = ls(tool_ctx(root), depth=3)
    listing = listing_metadata(result)

    assert [(entry.name, entry.type) for entry in listing.entries] == [
        ("linked", FileKind.SYMLINK)
    ]
    assert "secret.txt" not in result.return_value


def test_recursive_ls_is_safe_when_child_is_replaced_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    safe = root / "safe"
    outside = tmp_path / "outside"
    safe.mkdir(parents=True)
    outside.mkdir()
    (safe / "visible.txt").write_text("SAFE", encoding="utf-8")
    (outside / "secret.txt").write_text("SECRET", encoding="utf-8")
    ctx = tool_ctx(root)
    original_open = os.open
    swapped = False

    def racing_open(path: object, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if path == "safe" and not swapped:
            swapped = True
            safe.rename(root / "original-safe")
            safe.symlink_to(outside, target_is_directory=True)
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(workspace_module.os, "open", racing_open)

    result = ls(ctx, depth=2)
    listing = listing_metadata(result)

    assert "secret.txt" not in result.return_value
    assert listing.reasons == [ListingIncompleteReason.ENTRY_ERROR]
    assert listing.next_cursor is None
