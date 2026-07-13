from pathlib import Path

from pydantic_ai import Tool
import pytest

from ghostwheel.tools.bash import bash
from ghostwheel.tools.catalog import DEFAULT_TOOL_CATALOG, ToolProfile
from ghostwheel.tools.deps import ToolLimits
from ghostwheel.tools.edit import edit
from ghostwheel.tools.filesystem import ls, read
from ghostwheel.tools.output import OutputBudget, truncate_utf8
from ghostwheel.tools.search import grep

from .support import tool_ctx


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


def test_tool_catalog_exposes_immutable_capability_profiles() -> None:
    edit_tool = DEFAULT_TOOL_CATALOG.write[0]

    assert DEFAULT_TOOL_CATALOG.for_profile(ToolProfile.READ_ONLY) == (
        read,
        ls,
        grep,
    )
    assert DEFAULT_TOOL_CATALOG.for_profile(ToolProfile.SHELL_ONLY) == (bash,)
    assert isinstance(edit_tool, Tool)
    assert edit_tool.function is edit
    assert edit_tool.sequential is True
    assert DEFAULT_TOOL_CATALOG.for_profile(ToolProfile.FULL) == (
        read,
        ls,
        grep,
        edit_tool,
        bash,
    )
    assert DEFAULT_TOOL_CATALOG.for_profile(ToolProfile.FULL)[-1] is bash
