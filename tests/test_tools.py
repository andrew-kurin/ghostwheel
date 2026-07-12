import asyncio
import copy
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import time
from types import SimpleNamespace

import pytest
import ghostwheel.tools.filesystem as filesystem_module
import ghostwheel.tools.search as search_module
import ghostwheel.tools.workspace as workspace_module

from ghostwheel.tools.bash import bash
from ghostwheel.tools.catalog import DEFAULT_TOOL_CATALOG, ToolProfile
from ghostwheel.tools.command import CommandResult
from ghostwheel.tools.deps import ToolDeps, ToolLimits
from ghostwheel.tools.filesystem import (
    DirectoryListing,
    FileKind,
    ListingIncompleteReason,
    ls,
    read,
)
from ghostwheel.tools.output import OutputBudget, truncate_utf8
from ghostwheel.tools.search import GrepIncompleteReason, GrepResult, grep
from ghostwheel.tools.workspace import Workspace


def tool_ctx(root: Path, **overrides: object) -> SimpleNamespace:
    deps_kwargs = {
        "cwd": root,
        "allowed_roots": [root],
        "max_output_bytes": 100_000,
        "bash_timeout_seconds": 30,
        "dry_run": False,
        **overrides,
    }
    return SimpleNamespace(deps=ToolDeps(**deps_kwargs))


def run_bash(ctx: SimpleNamespace, command: str):
    return asyncio.run(bash(ctx, command))


def listing_metadata(result: object) -> DirectoryListing:
    metadata = getattr(result, "metadata", None)
    assert isinstance(metadata, DirectoryListing)
    return metadata


def grep_metadata(result: object) -> GrepResult:
    metadata = getattr(result, "metadata", None)
    assert isinstance(metadata, GrepResult)
    return metadata


def test_read_resolves_relative_paths_against_tool_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    (root / "README.md").write_text("hello\nworld\n", encoding="utf-8")

    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    result = read(tool_ctx(root), "README.md")

    assert result.path == "README.md"
    assert result.content == "   1 | hello\n   2 | world"
    assert result.line_count == 2
    assert result.truncated is False


def test_relative_filesystem_roots_are_resolved_from_tool_cwd(tmp_path: Path) -> None:
    cwd = (tmp_path / "repo").resolve()
    shared = cwd / "shared"
    shared.mkdir(parents=True)
    (shared / "value.txt").write_text("value", encoding="utf-8")
    ctx = SimpleNamespace(deps=ToolDeps(cwd=cwd, filesystem_roots=["shared"]))

    result = read(ctx, str(shared / "value.txt"))

    assert result.path == "shared/value.txt"


def test_filesystem_tools_reject_paths_outside_allowed_roots(tmp_path: Path) -> None:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    ctx = tool_ctx(root)

    with pytest.raises(ValueError, match="outside allowed roots"):
        read(ctx, "../outside.txt")

    with pytest.raises(ValueError, match="outside allowed roots"):
        ls(ctx, "..")


def test_filesystem_tools_never_traverse_symlinked_parent_directories(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "repo").resolve()
    outside = (tmp_path / "outside").resolve()
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET", encoding="utf-8")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    ctx = tool_ctx(root)

    with pytest.raises(OSError):
        read(ctx, "linked/secret.txt")
    with pytest.raises(OSError):
        ls(ctx, "linked")

    assert grep_metadata(grep(ctx, "SECRET")).matches == []


def test_read_is_safe_when_parent_is_swapped_after_descriptor_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "repo").resolve()
    safe = root / "safe"
    outside = (tmp_path / "outside").resolve()
    safe.mkdir(parents=True)
    outside.mkdir()
    (safe / "value.txt").write_text("SAFE", encoding="utf-8")
    (outside / "value.txt").write_text("SECRET", encoding="utf-8")
    original_open = os.open
    swapped = False

    def racing_open(path: object, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if path == "value.txt" and not swapped:
            swapped = True
            safe.rename(root / "original-safe")
            safe.symlink_to(outside, target_is_directory=True)
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(workspace_module.os, "open", racing_open)

    result = read(tool_ctx(root), "safe/value.txt")

    assert result.content.endswith("SAFE")
    assert "SECRET" not in result.content


def test_workspace_pins_allowed_root_across_ancestor_swap(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    root = parent / "repo"
    outside_parent = tmp_path / "outside-parent"
    outside_root = outside_parent / "repo"
    root.mkdir(parents=True)
    outside_root.mkdir(parents=True)
    (root / "value.txt").write_text("SAFE", encoding="utf-8")
    (outside_root / "value.txt").write_text("SECRET", encoding="utf-8")
    ctx = tool_ctx(root)
    parent.rename(tmp_path / "original-parent")
    parent.symlink_to(outside_parent, target_is_directory=True)

    result = read(ctx, "value.txt")

    assert result.content.endswith("SAFE")
    assert "SECRET" not in result.content


def test_workspace_initialization_rejects_ancestor_symlink_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "allowed-parent-race"
    root = parent / "repo"
    outside_parent = tmp_path / "outside-parent-race"
    (outside_parent / "repo").mkdir(parents=True)
    root.mkdir(parents=True)
    original_open = os.open
    swapped = False

    def racing_open(path: object, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if not swapped and path == parent.name:
            swapped = True
            parent.rename(tmp_path / "original-allowed-parent")
            parent.symlink_to(outside_parent, target_is_directory=True)
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(workspace_module.os, "open", racing_open)

    with pytest.raises(OSError):
        tool_ctx(root)


def test_closed_workspace_never_reuses_or_recloses_descriptor_numbers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "value.txt").write_text("SAFE", encoding="utf-8")
    workspace = Workspace(root)
    workspace.close()
    outside_fd = os.open(outside, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(RuntimeError, match="Workspace is closed"):
            with workspace.open_file("value.txt"):
                pass
        workspace.close()
        os.fstat(outside_fd)
    finally:
        os.close(outside_fd)


def test_workspace_copies_share_one_descriptor_owner(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    workspace = Workspace(root)

    assert copy.copy(workspace) is workspace
    assert copy.deepcopy(workspace) is workspace


def test_read_rejects_non_positive_max_output_bytes(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    (root / "README.md").write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="max_output_bytes must be positive"):
        read(tool_ctx(root, max_output_bytes=0), "README.md")


@pytest.mark.parametrize("field", ["regex_timeout_seconds", "bash_timeout_seconds"])
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_runtime_timeouts_must_be_finite(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        ToolLimits(**{field: value})


def test_output_budget_normalizes_surrogateescaped_names() -> None:
    value = "bad\udcff.py"
    normalized, changed = truncate_utf8(value, 100)
    budget = OutputBudget(100)

    assert normalized == "bad\\udcff.py"
    assert changed is True
    assert budget.consume(value) is True


def test_grep_finds_matches_and_skips_noise_directories(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("needle\n", encoding="utf-8")
    (root / ".venv").mkdir()
    (root / ".venv" / "ignored.py").write_text("needle\n", encoding="utf-8")

    search = grep_metadata(grep(tool_ctx(root), "needle", file_glob="*.py"))

    assert search.files_searched == 1
    assert search.complete is True
    assert [
        (match.file, match.line, match.column, match.text) for match in search.matches
    ] == [("src/app.py", 1, 1, "needle")]


def test_grep_returns_paths_relative_to_tool_cwd_for_scoped_search(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("needle\n", encoding="utf-8")

    search = grep_metadata(grep(tool_ctx(root), "needle", path="src", file_glob="*.py"))

    assert [(match.file, match.line, match.text) for match in search.matches] == [
        ("src/app.py", 1, "needle")
    ]


def test_grep_returns_readable_absolute_paths_for_allowed_roots_outside_cwd(
    tmp_path: Path,
) -> None:
    cwd = (tmp_path / "repo").resolve()
    outside_root = (tmp_path / "shared").resolve()
    cwd.mkdir()
    outside_root.mkdir()
    (outside_root / "shared.py").write_text("needle\n", encoding="utf-8")
    ctx = tool_ctx(cwd, allowed_roots=[cwd, outside_root])

    search = grep_metadata(
        grep(ctx, "needle", path=str(outside_root), file_glob="*.py")
    )

    assert [(match.file, match.line, match.text) for match in search.matches] == [
        (str(outside_root / "shared.py"), 1, "needle")
    ]
    assert read(ctx, search.matches[0].file).path == str(outside_root / "shared.py")


def test_grep_skips_symlinked_files_outside_allowed_roots(tmp_path: Path) -> None:
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    outside = (tmp_path / "secret.txt").resolve()
    outside.write_text("needle\n", encoding="utf-8")
    (root / "link.txt").symlink_to(outside)

    search = grep_metadata(grep(tool_ctx(root), "needle", file_glob="*.txt"))

    assert search.matches == []
    assert search.files_searched == 0
    assert search.complete is True


def test_grep_is_safe_when_a_file_is_replaced_by_a_symlink_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside.txt"
    root.mkdir()
    (root / "safe.txt").write_text("SAFE", encoding="utf-8")
    outside.write_text("SECRET", encoding="utf-8")
    ctx = tool_ctx(root)
    original_open = search_module.os.open
    swapped = False

    def racing_open(path: object, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if path == "safe.txt" and not swapped:
            swapped = True
            (root / "safe.txt").rename(root / "original-safe.txt")
            (root / "safe.txt").symlink_to(outside)
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(search_module.os, "open", racing_open)

    search = grep_metadata(grep(ctx, "SECRET", file_glob="*.txt"))

    assert search.matches == []
    assert search.files_skipped == 1
    assert search.reasons == [GrepIncompleteReason.FILE_ERROR]


def test_grep_is_safe_when_a_directory_is_replaced_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    safe = root / "safe"
    outside = tmp_path / "outside"
    safe.mkdir(parents=True)
    outside.mkdir()
    (safe / "value.txt").write_text("SAFE", encoding="utf-8")
    (outside / "secret.txt").write_text("SECRET", encoding="utf-8")
    ctx = tool_ctx(root)
    original_open = search_module.os.open
    swapped = False

    def racing_open(path: object, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if path == "safe" and not swapped:
            swapped = True
            safe.rename(root / "original-safe")
            safe.symlink_to(outside, target_is_directory=True)
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(search_module.os, "open", racing_open)

    search = grep_metadata(grep(ctx, "SECRET"))

    assert search.matches == []
    assert search.reasons == [GrepIncompleteReason.ENTRY_ERROR]


def test_grep_continues_after_an_unreadable_subtree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "a-blocked").mkdir()
    (tmp_path / "z-visible").mkdir()
    (tmp_path / "z-visible" / "value.txt").write_text(
        "needle",
        encoding="utf-8",
    )
    ctx = tool_ctx(tmp_path)
    original_open = search_module.os.open

    def denying_open(path: object, *args: object, **kwargs: object) -> int:
        if path == "a-blocked":
            raise PermissionError("blocked by test")
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(search_module.os, "open", denying_open)

    search = grep_metadata(grep(ctx, "needle"))

    assert [match.file for match in search.matches] == ["z-visible/value.txt"]
    assert search.reasons == [GrepIncompleteReason.ENTRY_ERROR]


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


def test_read_caps_numbered_content_to_output_budget(tmp_path: Path) -> None:
    (tmp_path / "many-lines.txt").write_text("a\nb\nc\n", encoding="utf-8")

    result = read(tool_ctx(tmp_path, max_output_bytes=10), "many-lines.txt")

    assert len(result.content.encode("utf-8")) <= 10
    assert result.truncated is True


def test_grep_caps_exact_escaped_model_output_without_partial_rows(
    tmp_path: Path,
) -> None:
    content = 'prefix needle "quote" \\ slash \u2028 suffix'
    for index in range(12):
        name = f"{index:02d}-{'line\nbreak' if index == 0 else 'quoted"name'}.txt"
        (tmp_path / name).write_text(content, encoding="utf-8")

    result = grep(
        tool_ctx(tmp_path, max_output_bytes=420),
        "needle",
        file_glob="*.txt",
    )
    search = grep_metadata(result)

    assert len(result.return_value.encode("utf-8")) <= 420
    assert GrepIncompleteReason.OUTPUT_BUDGET in search.reasons
    assert search.matches
    assert search.next_cursor is not None
    assert "\u2028" not in result.return_value
    assert "\\u2028" in result.return_value
    for row in result.return_value.splitlines()[1:]:
        if row.startswith(("f ", "next ")):
            json.loads(row.split(" ", 1)[1])
            continue
        location, encoded_text = row.split(" ", 1)
        line, column = location.rstrip("~").split(":", 1)
        assert line.isdigit()
        assert column.isdigit()
        json.loads(encoded_text)


def test_grep_tiny_output_budget_reports_an_omitted_row(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text(
        "needle " + "x" * 1_000,
        encoding="utf-8",
    )

    result = grep(tool_ctx(tmp_path, max_output_bytes=40), "needle")
    search = grep_metadata(result)

    assert len(result.return_value.encode("utf-8")) <= 40
    assert result.return_value == "grep omitted=1 output_budget"
    assert search.matches == []
    assert search.output_skipped == 1
    assert search.next_cursor is None
    assert search.reasons == [GrepIncompleteReason.OUTPUT_BUDGET]


def test_grep_centers_long_line_snippets_and_preserves_original_column(
    tmp_path: Path,
) -> None:
    line = "L" * 500 + "NEEDLE" + "R" * 500
    (tmp_path / "long.txt").write_text(line, encoding="utf-8")

    result = grep(tool_ctx(tmp_path), "NEEDLE", path="long.txt")
    search = grep_metadata(result)

    assert len(search.matches) == 1
    match = search.matches[0]
    assert (match.line, match.column) == (1, 501)
    assert match.text_truncated is True
    assert len(match.text) == 600
    assert match.text.startswith("…") and match.text.endswith("…")
    assert "NEEDLE" in match.text
    assert "\n1:501~ " in result.return_value


def test_grep_uses_only_lf_and_crlf_as_line_boundaries(tmp_path: Path) -> None:
    first = "first\u0085middle\u2028needle\u2029tail"
    (tmp_path / "separators.txt").write_text(
        f"{first}\nnext needle\r\n",
        encoding="utf-8",
    )

    search = grep_metadata(grep(tool_ctx(tmp_path), "needle"))

    assert [(match.line, match.column, match.text) for match in search.matches] == [
        (1, 14, first),
        (2, 6, "next needle"),
    ]


@pytest.mark.parametrize(
    ("name", "contents", "reason"),
    [
        ("invalid.txt", b"needle\xff\n", GrepIncompleteReason.ENCODING_ERROR),
        ("binary.txt", b"needle\0tail\n", GrepIncompleteReason.BINARY_FILE),
    ],
)
def test_grep_skips_invalid_utf8_and_binary_files_explicitly(
    tmp_path: Path,
    name: str,
    contents: bytes,
    reason: GrepIncompleteReason,
) -> None:
    (tmp_path / name).write_bytes(contents)

    search = grep_metadata(grep(tool_ctx(tmp_path), "needle", path=name))

    assert search.matches == []
    assert search.files_searched == 0
    assert search.files_skipped == 1
    assert search.reasons == [reason]


def test_grep_literal_mode_does_not_interpret_regex_syntax(tmp_path: Path) -> None:
    (tmp_path / "syntax.txt").write_text("a.b\naxb\n[x]\n", encoding="utf-8")
    ctx = tool_ctx(tmp_path)

    regex_search = grep_metadata(grep(ctx, "a.b"))
    literal_search = grep_metadata(grep(ctx, "a.b", literal=True))
    bracket_search = grep_metadata(grep(ctx, "[", literal=True))

    assert [(match.line, match.column) for match in regex_search.matches] == [
        (1, 1),
        (2, 1),
    ]
    assert [(match.line, match.column) for match in literal_search.matches] == [(1, 1)]
    assert [(match.line, match.column) for match in bracket_search.matches] == [(3, 1)]
    with pytest.raises(ValueError, match="Invalid regex"):
        grep(ctx, "[")


def test_grep_handles_lone_surrogates_in_user_patterns_and_globs(
    tmp_path: Path,
) -> None:
    (tmp_path / "value.txt").write_text("value", encoding="utf-8")
    ctx = tool_ctx(tmp_path)

    pattern = grep_metadata(grep(ctx, "\ud800"))
    file_glob = grep_metadata(grep(ctx, "value", file_glob="\ud800"))

    assert pattern.matches == []
    assert pattern.pattern == "\\ud800"
    assert file_glob.matches == []


def test_grep_paginates_in_sorted_order_with_query_bound_cursor(
    tmp_path: Path,
) -> None:
    for name in reversed(["a.txt", "b.txt", "c.txt", "d.txt", "e.txt"]):
        (tmp_path / name).write_text("needle", encoding="utf-8")
    ctx = tool_ctx(tmp_path)

    first = grep_metadata(grep(ctx, "needle", limit=2))
    second = grep_metadata(grep(ctx, "needle", limit=2, cursor=first.next_cursor))
    third = grep_metadata(grep(ctx, "needle", limit=2, cursor=second.next_cursor))

    assert [
        match.file for page in (first, second, third) for match in page.matches
    ] == ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt"]
    assert first.reasons == [GrepIncompleteReason.PAGE_LIMIT]
    assert second.reasons == [GrepIncompleteReason.PAGE_LIMIT]
    assert third.complete is True
    assert third.next_cursor is None
    with pytest.raises(ValueError, match="mismatched grep cursor"):
        grep(ctx, "different", limit=2, cursor=first.next_cursor)


def test_grep_cursor_rejects_a_changed_result_snapshot(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("needle", encoding="utf-8")
    (tmp_path / "b.txt").write_text("needle", encoding="utf-8")
    ctx = tool_ctx(tmp_path)
    first = grep_metadata(grep(ctx, "needle", limit=1))
    (tmp_path / "c.txt").write_text("needle", encoding="utf-8")

    with pytest.raises(ValueError, match="search results changed"):
        grep(ctx, "needle", limit=1, cursor=first.next_cursor)


def test_grep_cursor_is_bound_to_the_display_path_namespace(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.txt").write_text("needle", encoding="utf-8")
    (repo / "b.txt").write_text("needle", encoding="utf-8")
    first_deps = ToolDeps(cwd=repo)
    second_deps = ToolDeps(cwd=tmp_path, filesystem_roots=[repo])
    try:
        first = grep_metadata(grep(SimpleNamespace(deps=first_deps), "needle", limit=1))

        with pytest.raises(ValueError, match="mismatched grep cursor"):
            grep(
                SimpleNamespace(deps=second_deps),
                "needle",
                path=str(repo),
                limit=1,
                cursor=first.next_cursor,
            )
    finally:
        first_deps.close()
        second_deps.close()


def test_grep_preserves_recursive_glob_semantics(tmp_path: Path) -> None:
    (tmp_path / "src" / "nested").mkdir(parents=True)
    (tmp_path / "a.py").write_text("needle", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("needle", encoding="utf-8")
    (tmp_path / "src" / "nested" / "c.py").write_text("needle", encoding="utf-8")
    ctx = tool_ctx(tmp_path)

    simple = grep_metadata(grep(ctx, "needle", file_glob="*.py"))
    dotted = grep_metadata(grep(ctx, "needle", file_glob="./*.py"))
    recursive = grep_metadata(grep(ctx, "needle", file_glob="**/*.py"))
    src_python = grep_metadata(grep(ctx, "needle", file_glob="src/**/*.py"))
    redundant_separator = grep_metadata(grep(ctx, "needle", file_glob="src//*.py"))

    assert [match.file for match in simple.matches] == [
        "a.py",
        "src/b.py",
        "src/nested/c.py",
    ]
    assert [match.file for match in recursive.matches] == [
        "a.py",
        "src/b.py",
        "src/nested/c.py",
    ]
    assert [match.file for match in dotted.matches] == [
        "a.py",
        "src/b.py",
        "src/nested/c.py",
    ]
    assert [match.file for match in src_python.matches] == [
        "src/b.py",
        "src/nested/c.py",
    ]
    assert [match.file for match in redundant_separator.matches] == ["src/b.py"]
    with pytest.raises(ValueError, match="relative pattern"):
        grep(ctx, "needle", file_glob="/absolute/*.py")


def test_grep_reports_the_recursive_depth_limit(tmp_path: Path) -> None:
    directory = tmp_path
    for _index in range(65):
        directory /= "d"
        directory.mkdir()
    (directory / "deep.py").write_text("needle", encoding="utf-8")

    search = grep_metadata(grep(tool_ctx(tmp_path), "needle", file_glob="*.py"))

    assert search.matches == []
    assert GrepIncompleteReason.DEPTH_LIMIT in search.reasons


def test_grep_hidden_and_noise_controls_are_independent(tmp_path: Path) -> None:
    (tmp_path / ".hidden-dir").mkdir()
    (tmp_path / ".hidden-dir" / "nested.py").write_text("needle", encoding="utf-8")
    (tmp_path / ".hidden.py").write_text("needle", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "vendor.py").write_text("needle", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "package.py").write_text("needle", encoding="utf-8")
    (tmp_path / "visible.py").write_text("needle", encoding="utf-8")
    ctx = tool_ctx(tmp_path)

    default = grep_metadata(grep(ctx, "needle", file_glob="*.py"))
    hidden = grep_metadata(grep(ctx, "needle", file_glob="*.py", show_hidden=True))
    noise = grep_metadata(grep(ctx, "needle", file_glob="*.py", include_noise=True))
    all_entries = grep_metadata(
        grep(
            ctx,
            "needle",
            file_glob="*.py",
            show_hidden=True,
            include_noise=True,
        )
    )

    assert [match.file for match in default.matches] == ["visible.py"]
    assert [match.file for match in hidden.matches] == [
        ".hidden-dir/nested.py",
        ".hidden.py",
        "visible.py",
    ]
    assert [match.file for match in noise.matches] == [
        "node_modules/package.py",
        "visible.py",
    ]
    assert [match.file for match in all_entries.matches] == [
        ".hidden-dir/nested.py",
        ".hidden.py",
        ".venv/vendor.py",
        "node_modules/package.py",
        "visible.py",
    ]


def test_grep_file_limit_counts_files_not_scanned_directories(tmp_path: Path) -> None:
    for index in range(5):
        (tmp_path / f"directory-{index}").mkdir()
    for name in ["a.txt", "b.txt", "c.txt"]:
        (tmp_path / name).write_text("needle", encoding="utf-8")

    search = grep_metadata(
        grep(tool_ctx(tmp_path, max_search_files=2), "needle", file_glob="*.txt")
    )

    assert [match.file for match in search.matches] == ["a.txt", "b.txt"]
    assert search.files_searched == 2
    assert search.reasons == [GrepIncompleteReason.FILE_LIMIT]


def test_grep_scan_limit_counts_entries_independently_of_file_limit(
    tmp_path: Path,
) -> None:
    for index in range(4):
        (tmp_path / f"directory-{index}").mkdir()

    search = grep_metadata(
        grep(
            tool_ctx(
                tmp_path,
                max_directory_scan_entries=2,
                max_search_files=100,
            ),
            "needle",
            file_glob="*.nomatch",
        )
    )

    assert search.entries_scanned == 2
    assert search.files_searched == 0
    assert search.reasons == [GrepIncompleteReason.SCAN_LIMIT]


def test_grep_enforces_total_bytes_across_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("needle", encoding="utf-8")
    (tmp_path / "b.txt").write_text("needle", encoding="utf-8")

    exact = grep_metadata(grep(tool_ctx(tmp_path, max_search_total_bytes=12), "needle"))
    limited = grep_metadata(
        grep(tool_ctx(tmp_path, max_search_total_bytes=11), "needle")
    )

    assert [match.file for match in exact.matches] == ["a.txt", "b.txt"]
    assert exact.bytes_inspected == 12
    assert exact.complete is True
    assert [match.file for match in limited.matches] == ["a.txt"]
    assert limited.bytes_inspected == 6
    assert limited.files_skipped == 1
    assert limited.reasons == [GrepIncompleteReason.TOTAL_BYTES]


def test_grep_bounds_a_file_that_grows_past_the_total_byte_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "growing.txt").write_text("needle", encoding="utf-8")
    ctx = tool_ctx(
        tmp_path,
        max_search_file_bytes=10,
        max_search_total_bytes=3,
    )
    real_fstat = search_module.os.fstat

    def reporting_stale_size(descriptor: int) -> os.stat_result:
        result = real_fstat(descriptor)
        if not stat.S_ISREG(result.st_mode):
            return result
        values = list(result)
        values[stat.ST_SIZE] = 1
        return os.stat_result(values)

    monkeypatch.setattr(search_module.os, "fstat", reporting_stale_size)

    search = grep_metadata(grep(ctx, "needle", path="growing.txt"))

    assert search.matches == []
    assert search.bytes_inspected == 3
    assert search.files_skipped == 1
    assert search.reasons == [GrepIncompleteReason.TOTAL_BYTES]


def test_grep_operation_deadline_returns_partial_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "value.txt").write_text("needle", encoding="utf-8")
    ticks = iter((0.0, 2.0))
    monkeypatch.setattr(
        search_module,
        "time",
        SimpleNamespace(monotonic=lambda: next(ticks, 2.0)),
    )

    search = grep_metadata(
        grep(tool_ctx(tmp_path, search_timeout_seconds=1.0), "needle")
    )

    assert search.matches == []
    assert search.reasons == [GrepIncompleteReason.TIMEOUT]


def test_grep_checks_the_deadline_after_a_literal_line_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "value.txt").write_text("needle", encoding="utf-8")
    ticks = iter((0.0, 0.0, 0.0, 2.0))
    monkeypatch.setattr(
        search_module,
        "time",
        SimpleNamespace(monotonic=lambda: next(ticks, 2.0)),
    )

    search = grep_metadata(
        grep(
            tool_ctx(tmp_path, search_timeout_seconds=1.0),
            "needle",
            path="value.txt",
        )
    )

    assert search.matches == []
    assert search.files_searched == 1
    assert search.reasons == [GrepIncompleteReason.TIMEOUT]


def test_grep_stops_at_the_exact_match_cap(tmp_path: Path) -> None:
    for name in ["a.txt", "b.txt", "c.txt"]:
        (tmp_path / name).write_text("needle", encoding="utf-8")

    search = grep_metadata(grep(tool_ctx(tmp_path, max_matches=2), "needle"))

    assert [match.file for match in search.matches] == ["a.txt", "b.txt"]
    assert search.files_searched == 2
    assert search.reasons == [GrepIncompleteReason.MATCH_LIMIT]
    assert search.next_cursor is None


def test_grep_skips_files_over_the_input_limit(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text("needle", encoding="utf-8")

    search = grep_metadata(
        grep(
            tool_ctx(tmp_path, max_search_file_bytes=5),
            "needle",
            file_glob="*.txt",
        )
    )

    assert search.matches == []
    assert search.files_skipped == 1
    assert search.reasons == [GrepIncompleteReason.FILE_TOO_LARGE]


def test_grep_reports_a_direct_unreadable_file_as_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "blocked.txt").write_text("needle", encoding="utf-8")
    ctx = tool_ctx(tmp_path)
    original_open = workspace_module.os.open

    def denying_open(path: object, *args: object, **kwargs: object) -> int:
        if path == "blocked.txt":
            raise PermissionError("blocked by test")
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(workspace_module.os, "open", denying_open)

    search = grep_metadata(grep(ctx, "needle", path="blocked.txt"))

    assert search.matches == []
    assert search.files_skipped == 1
    assert search.reasons == [GrepIncompleteReason.FILE_ERROR]


def test_grep_does_not_treat_workspace_ancestors_as_noise(tmp_path: Path) -> None:
    root = tmp_path / ".venv" / "project"
    root.mkdir(parents=True)
    (root / "app.py").write_text("needle", encoding="utf-8")

    search = grep_metadata(grep(tool_ctx(root), "needle", file_glob="*.py"))

    assert [match.file for match in search.matches] == ["app.py"]


def test_grep_times_out_pathological_regular_expressions(tmp_path: Path) -> None:
    (tmp_path / "pathological.txt").write_text("a" * 50_000 + "!", encoding="utf-8")

    search = grep_metadata(
        grep(
            tool_ctx(
                tmp_path,
                regex_timeout_seconds=0.000001,
                search_timeout_seconds=1.0,
            ),
            "(a+)+$",
            file_glob="*.txt",
        )
    )

    assert search.matches == []
    assert search.reasons == [GrepIncompleteReason.TIMEOUT]


def test_bash_dry_run_does_not_execute_command(tmp_path: Path) -> None:
    target = tmp_path / "created.txt"

    result = run_bash(tool_ctx(tmp_path, dry_run=True), f"touch {target}")

    assert result.exit_code is None
    assert result.stderr == "Dry run: command was not executed."
    assert not target.exists()


def test_bash_truncates_combined_output_to_max_bytes(tmp_path: Path) -> None:
    result = run_bash(
        tool_ctx(tmp_path, max_output_bytes=12),
        "printf 'abcdefghij'; printf 'klmnopqrst' >&2",
    )

    assert result.truncated is True
    assert len((result.stdout + result.stderr).encode()) <= 12
    assert result.stdout == "abcdefghij"
    assert result.stderr == "kl"


def test_bash_truncation_preserves_stderr_when_stdout_exhausts_budget(
    tmp_path: Path,
) -> None:
    result = run_bash(
        tool_ctx(tmp_path, max_output_bytes=10),
        "printf 'abcdefghijklmnopqrst'; printf 'uvwxyz' >&2",
    )

    assert result.truncated is True
    assert len((result.stdout + result.stderr).encode()) <= 10
    assert result.stdout == "abcde"
    assert result.stderr == "uvwxy"


def test_bash_decodes_invalid_utf8_without_failing(tmp_path: Path) -> None:
    result = run_bash(tool_ctx(tmp_path), r"printf '\377'")

    assert result.exit_code == 0
    assert result.stdout == "�"


def test_bash_timeout_terminates_the_process_group(tmp_path: Path) -> None:
    result = run_bash(
        tool_ctx(tmp_path, bash_timeout_seconds=0.1),
        "sleep 10 & child=$!; echo $child; wait $child",
    )

    assert result.timed_out is True
    child_pid = int(result.stdout.strip())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(child_pid)],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not state or state.startswith("Z"):
            break
        time.sleep(0.02)
    else:
        os.kill(child_pid, 9)
        pytest.fail(f"child process {child_pid} survived command timeout")


def test_detached_child_cannot_stall_timeout_cleanup(tmp_path: Path) -> None:
    started = time.monotonic()
    result = run_bash(
        tool_ctx(tmp_path, bash_timeout_seconds=0.1),
        (
            "python -c 'import os,time; print(os.getpid(), flush=True); "
            "os.setsid(); time.sleep(2)' & wait"
        ),
    )
    elapsed = time.monotonic() - started

    assert result.timed_out is True
    assert elapsed < 1.0
    # A process that deliberately leaves the owned process group is outside the
    # default runner's guarantee; clean up the adversarial fixture explicitly.
    if result.stdout.strip():
        child_pid = int(result.stdout.strip())
        try:
            os.kill(child_pid, 9)
        except ProcessLookupError:
            pass


def test_bash_cancellation_terminates_owned_process_group(tmp_path: Path) -> None:
    pid_file = tmp_path / "shell.pid"

    async def cancel_running_command() -> None:
        task = asyncio.create_task(
            bash(
                tool_ctx(tmp_path),
                f"echo $$ > {shlex.quote(str(pid_file))}; sleep 10",
            )
        )
        deadline = asyncio.get_running_loop().time() + 2
        while not pid_file.exists():
            if asyncio.get_running_loop().time() >= deadline:
                task.cancel()
                raise AssertionError("command did not start")
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_running_command())

    shell_pid = int(pid_file.read_text(encoding="utf-8"))
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(shell_pid)],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert not state or state.startswith("Z")


def test_repeated_cancellation_cannot_interrupt_process_cleanup(tmp_path: Path) -> None:
    shell_file = tmp_path / "shell.pid"
    child_file = tmp_path / "child.pid"

    async def cancel_repeatedly() -> None:
        command = (
            f"echo $$ > {shlex.quote(str(shell_file))}; trap '' TERM; "
            f"sleep 10 & echo $! > {shlex.quote(str(child_file))}; wait"
        )
        task = asyncio.create_task(bash(tool_ctx(tmp_path), command))
        deadline = asyncio.get_running_loop().time() + 2
        while not shell_file.exists() or not child_file.exists():
            if asyncio.get_running_loop().time() >= deadline:
                task.cancel()
                raise AssertionError("command did not start")
            await asyncio.sleep(0.01)
        loop = asyncio.get_running_loop()
        task.cancel()
        loop.call_later(0.01, task.cancel)
        loop.call_later(0.02, task.cancel)
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_repeatedly())

    for pid_file in (shell_file, child_file):
        pid = int(pid_file.read_text(encoding="utf-8"))
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert not state or state.startswith("Z")


def test_bash_uses_injected_command_runner(tmp_path: Path) -> None:
    class FakeCommandRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Path, int, float]] = []

        async def run(
            self,
            command: str,
            *,
            cwd: Path,
            max_output_bytes: int,
            timeout_seconds: float,
        ) -> CommandResult:
            self.calls.append((command, cwd, max_output_bytes, timeout_seconds))
            return CommandResult(0, "injected", "", False, False)

    runner = FakeCommandRunner()
    ctx = SimpleNamespace(
        deps=ToolDeps(cwd=tmp_path, command_runner=runner, max_output_bytes=123)
    )

    result = run_bash(ctx, "status")

    assert result.stdout == "injected"
    assert runner.calls == [("status", tmp_path.resolve(), 123, 30)]


def test_falsey_command_runner_is_not_replaced(tmp_path: Path) -> None:
    class FalseyRunner:
        def __bool__(self) -> bool:
            return False

        async def run(self, *args: object, **kwargs: object) -> CommandResult:
            return CommandResult(0, "injected", "", False, False)

    runner = FalseyRunner()

    deps = ToolDeps(cwd=tmp_path, command_runner=runner)

    assert deps.command_runner is runner


def test_tool_catalog_exposes_immutable_capability_profiles() -> None:
    assert DEFAULT_TOOL_CATALOG.for_profile(ToolProfile.READ_ONLY) == (
        read,
        ls,
        grep,
    )
    assert DEFAULT_TOOL_CATALOG.for_profile(ToolProfile.FULL)[-1] is bash
