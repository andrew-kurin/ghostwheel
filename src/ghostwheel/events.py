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


@dataclass(frozen=True, slots=True)
class ToolFinished:
    name: str
    result: str


@dataclass(frozen=True, slots=True)
class ToolFailed:
    name: str
    error: str


AgentEvent: TypeAlias = (
    TextOutput | ThinkingOutput | ToolStarted | ToolFinished | ToolFailed
)
