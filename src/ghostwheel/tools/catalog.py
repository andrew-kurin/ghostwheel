from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

from pydantic_ai import Tool

from ghostwheel.tool_config import ToolProfile
from ghostwheel.tools.bash import bash
from ghostwheel.tools.edit import edit
from ghostwheel.tools.listing import ls
from ghostwheel.tools.read import read
from ghostwheel.tools.search import grep

ToolCallable: TypeAlias = Callable[..., object]
CatalogTool: TypeAlias = Tool[Any] | ToolCallable


@dataclass(frozen=True, slots=True)
class ToolCatalog:
    """Immutable catalog with explicit capability profiles."""

    read_only: tuple[CatalogTool, ...]
    shell: tuple[CatalogTool, ...]
    write: tuple[CatalogTool, ...] = ()

    def for_profile(
        self,
        profile: ToolProfile | str,
    ) -> tuple[CatalogTool, ...]:
        profile = ToolProfile(profile)
        if profile is ToolProfile.READ_ONLY:
            return self.read_only
        if profile is ToolProfile.SHELL_ONLY:
            return self.shell
        if profile is ToolProfile.FULL:
            return self.read_only + self.write + self.shell
        raise ValueError(f"Unknown tool profile: {profile}")


DEFAULT_TOOL_CATALOG = ToolCatalog(
    read_only=(read, ls, grep),
    shell=(bash,),
    write=(Tool(edit, sequential=True),),
)
