import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest
import ghostwheel.tools.search as search_module
import ghostwheel.tools.workspace as workspace_module

from ghostwheel.tools.deps import ToolDeps
from ghostwheel.tools.filesystem import read
from ghostwheel.tools.search import GrepIncompleteReason, grep

from .support import grep_metadata, read_metadata, tool_ctx


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
    assert read_metadata(read(ctx, search.matches[0].file)).path == str(
        outside_root / "shared.py"
    )


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
