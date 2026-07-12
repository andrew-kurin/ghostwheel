import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from ghostwheel.tools.bash import bash
from ghostwheel.tools.deps import ToolDeps
from ghostwheel.tools.filesystem import DirectoryListing, ReadResult
from ghostwheel.tools.search import GrepResult


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


def read_metadata(result: object) -> ReadResult:
    metadata = getattr(result, "metadata", None)
    assert isinstance(metadata, ReadResult)
    return metadata


def read_rows(result: object) -> list[tuple[int, str]]:
    """Parse complete numbered rows from read's compact model payload."""

    return_value = getattr(result, "return_value", None)
    assert isinstance(return_value, str)
    rows: list[tuple[int, str]] = []
    for row in return_value.splitlines()[1:]:
        if row.startswith("next "):
            json.loads(row.removeprefix("next "))
            continue
        line, text = row.split(":", 1)
        rows.append((int(line), text))
    return rows
