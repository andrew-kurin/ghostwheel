from pathlib import Path

import pytest

from ghostwheel.tools.filesystem import ReadIncompleteReason, read

from .support import read_metadata, read_rows, tool_ctx


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
    metadata = read_metadata(result)

    assert read_rows(result) == [(1, "hello"), (2, "world")]
    assert metadata.path == "README.md"
    assert metadata.start_line == 1
    assert metadata.end_line == 2
    assert metadata.lines_returned == 2
    assert metadata.total_bytes == 12
    assert metadata.eof is True
    assert metadata.complete is True
    assert metadata.reasons == []
    assert metadata.truncated_lines == []
    assert metadata.next_cursor is None


def test_read_pages_with_a_cursor_and_honors_the_configured_line_cap(
    tmp_path: Path,
) -> None:
    (tmp_path / "values.txt").write_text(
        "one\ntwo\nthree\nfour\nfive\n",
        encoding="utf-8",
    )
    ctx = tool_ctx(tmp_path, max_read_lines=2)

    first = read(ctx, "values.txt")
    first_metadata = read_metadata(first)
    second = read(ctx, "values.txt", cursor=first_metadata.next_cursor)
    second_metadata = read_metadata(second)
    third = read(ctx, "values.txt", cursor=second_metadata.next_cursor)
    third_metadata = read_metadata(third)

    assert read_rows(first) == [(1, "one"), (2, "two")]
    assert first_metadata.start_line == 1
    assert first_metadata.end_line == 2
    assert first_metadata.lines_returned == 2
    assert first_metadata.eof is False
    assert first_metadata.complete is False
    assert first_metadata.reasons == [ReadIncompleteReason.PAGE_LIMIT]
    assert first_metadata.next_cursor is not None

    assert read_rows(second) == [(3, "three"), (4, "four")]
    assert second_metadata.start_line == 3
    assert second_metadata.end_line == 4
    assert second_metadata.lines_returned == 2
    assert second_metadata.eof is False
    assert second_metadata.reasons == [ReadIncompleteReason.PAGE_LIMIT]
    assert second_metadata.next_cursor is not None

    assert read_rows(third) == [(5, "five")]
    assert third_metadata.start_line == 5
    assert third_metadata.end_line == 5
    assert third_metadata.lines_returned == 1
    assert third_metadata.eof is True
    assert third_metadata.complete is True
    assert third_metadata.reasons == []
    assert third_metadata.next_cursor is None


def test_read_defaults_to_a_two_hundred_line_page(tmp_path: Path) -> None:
    (tmp_path / "many.txt").write_text(
        "".join(f"{line}\n" for line in range(1, 202)),
        encoding="utf-8",
    )

    result = read(tool_ctx(tmp_path), "many.txt")
    metadata = read_metadata(result)

    assert metadata.lines_returned == 200
    assert metadata.end_line == 200
    assert metadata.eof is False
    assert metadata.reasons == [ReadIncompleteReason.PAGE_LIMIT]
    assert metadata.next_cursor is not None


def test_read_start_line_and_limit_return_the_requested_window(
    tmp_path: Path,
) -> None:
    (tmp_path / "values.txt").write_text(
        "one\ntwo\nthree\nfour\nfive\n",
        encoding="utf-8",
    )

    result = read(tool_ctx(tmp_path), "values.txt", start_line=3, limit=2)
    metadata = read_metadata(result)

    assert read_rows(result) == [(3, "three"), (4, "four")]
    assert metadata.start_line == 3
    assert metadata.end_line == 4
    assert metadata.lines_returned == 2
    assert metadata.eof is False
    assert metadata.complete is False
    assert metadata.reasons == [ReadIncompleteReason.PAGE_LIMIT]
    assert metadata.next_cursor is not None


def test_read_cursor_can_resize_pages_but_cannot_combine_with_start_line(
    tmp_path: Path,
) -> None:
    (tmp_path / "values.txt").write_text(
        "one\ntwo\nthree\nfour\n",
        encoding="utf-8",
    )
    ctx = tool_ctx(tmp_path, max_read_lines=3)
    first_metadata = read_metadata(read(ctx, "values.txt", limit=1))

    second = read(ctx, "values.txt", limit=2, cursor=first_metadata.next_cursor)

    assert read_rows(second) == [(2, "two"), (3, "three")]
    with pytest.raises(ValueError, match="start_line.*cursor"):
        read(
            ctx,
            "values.txt",
            start_line=2,
            cursor=first_metadata.next_cursor,
        )


def test_read_start_beyond_eof_returns_an_empty_complete_page(tmp_path: Path) -> None:
    (tmp_path / "value.txt").write_text("value\n", encoding="utf-8")

    result = read(tool_ctx(tmp_path), "value.txt", start_line=10)
    metadata = read_metadata(result)

    assert read_rows(result) == []
    assert metadata.start_line == 10
    assert metadata.end_line is None
    assert metadata.lines_returned == 0
    assert metadata.eof is True
    assert metadata.complete is True
    assert metadata.reasons == []
    assert metadata.next_cursor is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start_line": 0}, "start_line must be positive"),
        ({"start_line": -1}, "start_line must be positive"),
        ({"limit": 0}, "limit must be positive"),
        ({"limit": -1}, "limit must be positive"),
        ({"cursor": "not-a-cursor"}, "Invalid read cursor"),
    ],
)
def test_read_rejects_invalid_page_arguments(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    (tmp_path / "value.txt").write_text("value\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read(tool_ctx(tmp_path), "value.txt", **kwargs)  # type: ignore[arg-type]


def test_read_preserves_utf8_text(tmp_path: Path) -> None:
    (tmp_path / "unicode.txt").write_bytes("café\n雪\n".encode())

    result = read(tool_ctx(tmp_path), "unicode.txt")

    assert read_rows(result) == [
        (1, "café"),
        (2, "雪"),
    ]
    result.return_value.encode("utf-8", errors="strict")


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b"", []),
        (b"value", [(1, "value")]),
        (b"value\n", [(1, "value")]),
        (b"value\n\n", [(1, "value"), (2, "")]),
    ],
)
def test_read_lf_boundaries(
    tmp_path: Path,
    content: bytes,
    expected: list[tuple[int, str]],
) -> None:
    (tmp_path / "boundaries.txt").write_bytes(content)

    result = read(tool_ctx(tmp_path), "boundaries.txt")

    assert read_rows(result) == expected
    assert read_metadata(result).eof is True


def test_read_escapes_display_breaking_controls(tmp_path: Path) -> None:
    (tmp_path / "controls.txt").write_text(
        "literal\\r before\x1b[31m\rafter\u2028separator\n",
        encoding="utf-8",
    )

    result = read(tool_ctx(tmp_path), "controls.txt")

    assert read_rows(result) == [
        (1, r"literal\\r before\x1b[31m\rafter\u2028separator"),
    ]
    assert "\x1b" not in result.return_value
    assert "\u2028" not in result.return_value


def test_read_rejects_nul_bytes(tmp_path: Path) -> None:
    (tmp_path / "binary.txt").write_bytes(b"before\x00after\n")

    with pytest.raises(ValueError, match="NUL|binary"):
        read(tool_ctx(tmp_path), "binary.txt")


def test_read_rejects_invalid_utf8_and_nul_in_scanned_skipped_lines(
    tmp_path: Path,
) -> None:
    (tmp_path / "invalid.txt").write_bytes(b"\xff\nsafe\n")
    (tmp_path / "nul.txt").write_bytes(b"bad\x00line\nsafe\n")

    with pytest.raises(ValueError, match="UTF-8|encoding"):
        read(tool_ctx(tmp_path), "invalid.txt", start_line=2)
    with pytest.raises(ValueError, match="NUL|binary"):
        read(tool_ctx(tmp_path), "nul.txt", start_line=2)


def test_read_defers_validation_of_unseen_pages(tmp_path: Path) -> None:
    (tmp_path / "later-invalid.txt").write_bytes(b"safe\n\xff\n")
    ctx = tool_ctx(tmp_path, max_read_lines=1)

    first = read(ctx, "later-invalid.txt")

    assert read_rows(first) == [(1, "safe")]
    with pytest.raises(ValueError, match="UTF-8"):
        read(ctx, "later-invalid.txt", cursor=read_metadata(first).next_cursor)


def test_read_treats_lf_and_crlf_as_lines_but_preserves_a_lone_cr(
    tmp_path: Path,
) -> None:
    (tmp_path / "newlines.txt").write_bytes(b"alpha\r\nbeta\ncharlie\rdelta\r\n\n")

    result = read(tool_ctx(tmp_path), "newlines.txt")
    metadata = read_metadata(result)

    assert read_rows(result) == [
        (1, "alpha"),
        (2, "beta"),
        (3, r"charlie\rdelta"),
        (4, ""),
    ]
    assert metadata.end_line == 4
    assert metadata.eof is True


def test_read_cursor_rejects_a_changed_file(tmp_path: Path) -> None:
    path = tmp_path / "values.txt"
    path.write_text("one\ntwo\n", encoding="utf-8")
    ctx = tool_ctx(tmp_path, max_read_lines=1)
    first_metadata = read_metadata(read(ctx, "values.txt"))
    path.write_text("changed-and-longer\ntwo\n", encoding="utf-8")

    with pytest.raises(ValueError, match="file changed"):
        read(ctx, "values.txt", cursor=first_metadata.next_cursor)


def test_read_cursor_is_bound_to_its_file(tmp_path: Path) -> None:
    (tmp_path / "first.txt").write_text("one\ntwo\n", encoding="utf-8")
    (tmp_path / "second.txt").write_text("one\ntwo\n", encoding="utf-8")
    ctx = tool_ctx(tmp_path, max_read_lines=1)
    first_metadata = read_metadata(read(ctx, "first.txt"))

    with pytest.raises(ValueError, match="Mismatched read cursor.*path"):
        read(ctx, "second.txt", cursor=first_metadata.next_cursor)


def test_read_rejects_a_tampered_cursor(tmp_path: Path) -> None:
    (tmp_path / "values.txt").write_text("one\ntwo\n", encoding="utf-8")
    ctx = tool_ctx(tmp_path, max_read_lines=1)
    cursor = read_metadata(read(ctx, "values.txt")).next_cursor
    assert cursor is not None
    parts = cursor.split(".")
    parts[-2] = "1"

    with pytest.raises(ValueError, match="Invalid read cursor"):
        read(ctx, "values.txt", cursor=".".join(parts))


def test_read_scan_limit_bounds_pathological_lines_and_random_jumps(
    tmp_path: Path,
) -> None:
    (tmp_path / "giant.txt").write_text("x" * 256, encoding="utf-8")
    (tmp_path / "many.txt").write_text(
        f"{'a' * 39}\n{'b' * 39}\n{'c' * 39}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="scan limit"):
        read(
            tool_ctx(tmp_path, max_read_scan_bytes=64),
            "giant.txt",
        )
    with pytest.raises(ValueError, match="scan limit"):
        read(
            tool_ctx(tmp_path, max_read_scan_bytes=64),
            "many.txt",
            start_line=3,
        )


def test_read_scan_limit_is_an_exact_byte_ceiling(tmp_path: Path) -> None:
    (tmp_path / "exact.txt").write_bytes(b"abcd")
    (tmp_path / "over.txt").write_bytes(b"abcde")

    exact = read(
        tool_ctx(tmp_path, max_read_scan_bytes=4),
        "exact.txt",
    )

    assert read_rows(exact) == [(1, "abcd")]
    with pytest.raises(ValueError, match="scan limit"):
        read(
            tool_ctx(tmp_path, max_read_scan_bytes=4),
            "over.txt",
        )


def test_read_truncates_a_giant_line_and_cursor_still_makes_progress(
    tmp_path: Path,
) -> None:
    (tmp_path / "giant.txt").write_text(
        f"{'x' * 20_000}\nsecond\n",
        encoding="utf-8",
    )
    ctx = tool_ctx(tmp_path, max_output_bytes=512, max_read_lines=1)

    first = read(ctx, "giant.txt")
    first_metadata = read_metadata(first)
    second = read(ctx, "giant.txt", cursor=first_metadata.next_cursor)
    second_metadata = read_metadata(second)

    first_rows = read_rows(first)
    assert len(first.return_value.encode("utf-8")) <= 512
    assert len(first_rows) == 1
    assert first_rows[0][0] == 1
    assert first_rows[0][1].startswith("x")
    assert first_rows[0][1].endswith("… [line truncated]")
    assert len(first_rows[0][1]) < 20_000
    assert first_metadata.lines_returned == 1
    assert first_metadata.truncated_lines == [1]
    assert ReadIncompleteReason.LINE_TRUNCATED in first_metadata.reasons
    assert first_metadata.next_cursor is not None

    assert read_rows(second) == [(2, "second")]
    assert second_metadata.start_line == 2
    assert second_metadata.end_line == 2
    assert second_metadata.eof is True
    assert second_metadata.next_cursor is None


def test_read_caps_exact_model_output_without_partial_rows(tmp_path: Path) -> None:
    (tmp_path / "many-lines.txt").write_text(
        "".join(f"line-{index:02d}-{'x' * 40}\n" for index in range(20)),
        encoding="utf-8",
    )

    result = read(
        tool_ctx(tmp_path, max_output_bytes=300),
        "many-lines.txt",
    )
    metadata = read_metadata(result)
    rows = read_rows(result)

    assert len(result.return_value.encode("utf-8")) <= 300
    assert rows
    assert metadata.lines_returned == len(rows)
    assert metadata.end_line == rows[-1][0]
    assert ReadIncompleteReason.OUTPUT_BUDGET in metadata.reasons
    assert metadata.complete is False
    assert metadata.next_cursor is not None


def test_read_tiny_output_budget_never_emits_a_partial_row(
    tmp_path: Path,
) -> None:
    (tmp_path / "value.txt").write_text("value\n", encoding="utf-8")

    result = read(tool_ctx(tmp_path, max_output_bytes=10), "value.txt")
    metadata = read_metadata(result)

    assert len(result.return_value.encode("utf-8")) <= 10
    assert "\n" not in result.return_value
    assert read_rows(result) == []
    assert metadata.lines_returned == 0
    assert metadata.end_line is None
    assert metadata.complete is False
    assert ReadIncompleteReason.OUTPUT_BUDGET in metadata.reasons
