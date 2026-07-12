"""UI-neutral application metadata."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolInfo:
    """Display metadata for one tool available to the chat agent."""

    name: str
    description: str


@dataclass(frozen=True, slots=True)
class AppInfo:
    """Resolved runtime details displayed by terminal front ends."""

    workspace: str
    provider: str
    model: str
    tool_profile: str
    tools: tuple[ToolInfo, ...] = ()
