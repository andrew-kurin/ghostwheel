from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias

from ghostwheel.tools.bash import bash
from ghostwheel.tools.filesystem import ls, read
from ghostwheel.tools.search import grep

ToolCallable: TypeAlias = Callable[..., object]


class ToolProfile(str, Enum):
    READ_ONLY = "read-only"
    SHELL_ONLY = "shell-only"
    FULL = "full"


class ToolRegistrar(Protocol):
    def tool(self, tool: ToolCallable, /) -> object: ...


@dataclass(frozen=True, slots=True)
class ToolCatalog:
    """Immutable catalog with explicit capability profiles."""

    read_only: tuple[ToolCallable, ...]
    shell: tuple[ToolCallable, ...]

    def for_profile(self, profile: ToolProfile) -> tuple[ToolCallable, ...]:
        if profile is ToolProfile.READ_ONLY:
            return self.read_only
        if profile is ToolProfile.SHELL_ONLY:
            return self.shell
        if profile is ToolProfile.FULL:
            return self.read_only + self.shell
        raise ValueError(f"Unknown tool profile: {profile}")


DEFAULT_TOOL_CATALOG = ToolCatalog(
    read_only=(read, ls, grep),
    shell=(bash,),
)


def register_tools(
    agent: ToolRegistrar,
    tools: Iterable[ToolCallable] | None = None,
    *,
    catalog: ToolCatalog = DEFAULT_TOOL_CATALOG,
    profile: ToolProfile = ToolProfile.FULL,
) -> None:
    selected_tools = catalog.for_profile(profile) if tools is None else tuple(tools)
    for tool in selected_tools:
        agent.tool(tool)
