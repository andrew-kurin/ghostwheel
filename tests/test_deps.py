import math

import pytest

from ghostwheel.tools.deps import ToolDeps, ToolLimits


def test_tool_deps_exposes_search_aggregate_defaults(tmp_path) -> None:
    deps = ToolDeps(cwd=tmp_path)
    try:
        assert deps.max_read_lines == 200
        assert deps.max_read_scan_bytes == 5_000_000
        assert deps.max_search_total_bytes == 50_000_000
        assert deps.search_timeout_seconds == 5.0
    finally:
        deps.close()


def test_tool_deps_accepts_search_aggregate_scalar_overrides(tmp_path) -> None:
    deps = ToolDeps(
        cwd=tmp_path,
        max_read_lines=321,
        max_read_scan_bytes=654_321,
        max_search_total_bytes=123_456,
        search_timeout_seconds=0.75,
    )
    try:
        assert deps.limits.max_read_lines == 321
        assert deps.limits.max_read_scan_bytes == 654_321
        assert deps.limits.max_search_total_bytes == 123_456
        assert deps.limits.search_timeout_seconds == 0.75
        assert deps.limits.max_output_bytes == ToolLimits().max_output_bytes
        assert deps.limits.max_entries == ToolLimits().max_entries
    finally:
        deps.close()


def test_tool_deps_rejects_mixing_canonical_limits_with_legacy_scalars(
    tmp_path,
) -> None:
    with pytest.raises(TypeError, match="limits or scalar"):
        ToolDeps(cwd=tmp_path, limits=ToolLimits(), max_entries=10)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_read_lines", 0, "positive integer"),
        ("max_read_scan_bytes", 0, "positive integer"),
        ("max_search_total_bytes", 0, "positive integer"),
        ("search_timeout_seconds", 0, "positive and finite"),
        ("search_timeout_seconds", math.inf, "positive and finite"),
        ("search_timeout_seconds", math.nan, "positive and finite"),
    ],
)
def test_search_aggregate_tool_limits_must_be_positive_and_finite(
    field: str,
    value: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ToolLimits(**{field: value})
