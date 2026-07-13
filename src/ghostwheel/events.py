from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class TextOutput:
    content: str
    starts_part: bool = False


@dataclass(frozen=True, slots=True)
class ThinkingOutput:
    content: str
    starts_part: bool = False


@dataclass(frozen=True, slots=True)
class ToolStarted:
    name: str
    arguments: str
    call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolFinished:
    name: str
    result: str
    call_id: str | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ToolFailed:
    name: str
    error: str
    call_id: str | None = None


AgentEvent: TypeAlias = (
    TextOutput | ThinkingOutput | ToolStarted | ToolFinished | ToolFailed
)
