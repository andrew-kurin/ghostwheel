"""UI-neutral application metadata."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppInfo:
    """Resolved runtime details displayed by terminal front ends."""

    workspace: str
    provider: str
    model: str
    tool_profile: str
