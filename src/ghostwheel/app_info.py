"""UI-neutral application metadata."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolInfo:
    """Display metadata for one resolved agent tool."""

    name: str
    description: str


@dataclass(frozen=True, slots=True)
class ToolSetInfo:
    """Resolved tool capabilities for one independently configured agent."""

    profile: str
    tools: tuple[ToolInfo, ...] = ()
    has_shell_access: bool = False


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Resolved provider and model for one independently configured agent."""

    provider: str
    model: str


@dataclass(frozen=True, slots=True)
class AppInfo:
    """Resolved runtime details displayed by terminal front ends."""

    workspace: str
    chat_model: ModelInfo
    review_model: ModelInfo
    chat_tools: ToolSetInfo
    review_tools: ToolSetInfo
