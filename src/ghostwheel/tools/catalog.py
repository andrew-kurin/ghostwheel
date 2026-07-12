from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from ghostwheel.tool_config import ToolProfile
from ghostwheel.tools.bash import bash
from ghostwheel.tools.listing import ls
from ghostwheel.tools.read import read
from ghostwheel.tools.search import grep

ToolCallable: TypeAlias = Callable[..., object]


class ToolRegistrar(Protocol):
    def tool(self, tool: ToolCallable, /) -> object: ...


@dataclass(frozen=True, slots=True)
class ToolCatalog:
    """Immutable catalog with explicit capability profiles."""

    read_only: tuple[ToolCallable, ...]
    shell: tuple[ToolCallable, ...]

    def for_profile(
        self,
        profile: ToolProfile | str,
    ) -> tuple[ToolCallable, ...]:
        profile = ToolProfile(profile)
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
    profile: ToolProfile | str = ToolProfile.FULL,
) -> None:
    selected_tools = catalog.for_profile(profile) if tools is None else tuple(tools)
    for tool in selected_tools:
        agent.tool(tool)
