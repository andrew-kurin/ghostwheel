"""UI-neutral state and formatting shared by terminal presenters."""

from __future__ import annotations

import ast
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from ghostwheel.events import (
    AgentEvent,
    TextOutput,
    ThinkingOutput,
    ToolFailed,
    ToolFinished,
    ToolStarted,
)
from ghostwheel.runtime_contracts import FailureKind

ToolStatus: TypeAlias = Literal["running", "succeeded", "failed"]


@dataclass(slots=True)
class ToolActivity:
    """One correlated tool call as understood by a presentation layer."""

    name: str
    arguments: str
    call_id: str | None
    started_at: float
    status: ToolStatus = "running"
    detail: str = ""
    metadata: dict[str, object] | None = None
    finished_at: float | None = None


@dataclass(slots=True)
class TurnState:
    """Reduce streamed agent events into renderer-independent turn state."""

    status: str = "Thinking…"
    answer: str = ""
    thinking: str = ""
    tools: list[ToolActivity] = field(default_factory=list)
    clock: Callable[[], float] = field(
        default=time.monotonic,
        repr=False,
        compare=False,
    )

    def reset(self, label: str = "Thinking…") -> None:
        self.status = label
        self.answer = ""
        self.thinking = ""
        self.tools.clear()

    def apply(self, event: AgentEvent) -> ToolActivity | None:
        """Apply an event and return its affected tool, when applicable."""

        if isinstance(event, ThinkingOutput):
            self.thinking += event.content
            self.status = "Thinking…"
            return None
        if isinstance(event, TextOutput):
            self.answer += event.content
            self.status = "Responding…"
            return None
        if isinstance(event, ToolStarted):
            activity = ToolActivity(
                name=event.name,
                arguments=event.arguments,
                call_id=event.call_id,
                started_at=self.clock(),
            )
            self.tools.append(activity)
            self.status = f"Running {event.name}…"
            return activity
        if isinstance(event, ToolFinished):
            activity = self._finish_tool(
                event.name,
                event.call_id,
                "succeeded",
                event.result,
                event.metadata,
            )
            self.status = "Thinking…"
            return activity

        if isinstance(event, ToolFailed):
            activity = self._finish_tool(
                event.name,
                event.call_id,
                "failed",
                event.error,
            )
            self.status = f"{event.name} failed"
            return activity
        return None

    def _finish_tool(
        self,
        name: str,
        call_id: str | None,
        status: ToolStatus,
        detail: str,
        metadata: dict[str, object] | None = None,
    ) -> ToolActivity:
        activity = next(
            (
                item
                for item in reversed(self.tools)
                if item.status == "running"
                and (
                    (call_id is not None and item.call_id == call_id)
                    or (
                        call_id is not None
                        and item.call_id is None
                        and item.name == name
                    )
                    or (call_id is None and item.name == name)
                )
            ),
            None,
        )
        if activity is None:
            activity = ToolActivity(name, "", call_id, self.clock())
            self.tools.append(activity)
        activity.status = status
        activity.detail = detail
        activity.metadata = metadata
        activity.finished_at = self.clock()
        return activity


@dataclass(frozen=True, slots=True)
class FailurePresentation:
    title: str
    hint: str


_FAILURE_PRESENTATIONS = {
    FailureKind.PROVIDER: FailurePresentation(
        "Provider Error",
        "Check that the configured model server is running; use /model to inspect it.",
    ),
    FailureKind.CONFIGURATION: FailurePresentation(
        "Configuration Error",
        "Check the model, context-window, and compaction settings.",
    ),
    FailureKind.TOOL: FailurePresentation(
        "Tool Error",
        "Verify the workspace state and tool output, then adjust the request.",
    ),
    FailureKind.MODEL_OUTPUT: FailurePresentation(
        "Model Output Error",
        "The model returned an unsupported result; use /retry or change models.",
    ),
    FailureKind.UNKNOWN: FailurePresentation(
        "Agent Failed",
        "Use /retry to try the turn again.",
    ),
}


def failure_presentation(kind: FailureKind) -> FailurePresentation:
    return _FAILURE_PRESENTATIONS[kind]


def format_token_count(value: int) -> str:
    if value < 1_000:
        return str(value)
    thousands = value / 1_000
    if thousands < 10 and not thousands.is_integer():
        compact = f"{thousands:.1f}".rstrip("0").rstrip(".")
        return f"{compact}k"
    return f"{thousands:.0f}k"


def primary_argument(arguments: str) -> str:
    if not arguments:
        return ""
    try:
        parsed = ast.literal_eval(arguments)
    except SyntaxError, ValueError:
        return preview(" ".join(arguments.split()), 72)
    if not isinstance(parsed, dict):
        return preview(" ".join(str(parsed).split()), 72)
    for key in ("path", "command", "pattern", "query", "paths"):
        value = parsed.get(key)
        if value not in (None, ""):
            return " ".join(str(value).split())
    return ""


def duration(seconds: float) -> str:
    if seconds < 0.001:
        return "<1 ms"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.1f} s"


def preview(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit] + "…"
