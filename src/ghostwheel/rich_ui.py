from __future__ import annotations

import ast
import time
from dataclasses import dataclass

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from ghostwheel.events import (
    AgentEvent,
    TextOutput,
    ThinkingOutput,
    ToolFailed,
    ToolFinished,
    ToolStarted,
)
from ghostwheel.rendering import render_review
from ghostwheel.review import RawReview, ReviewFailed, ReviewOutcome, StructuredReview
from ghostwheel.session import (
    FailureKind,
    TurnFailed,
    TurnNoResult,
    TurnOutcome,
    TurnSucceeded,
)


@dataclass(frozen=True, slots=True)
class AppInfo:
    workspace: str
    provider: str
    model: str
    tool_profile: str


@dataclass(slots=True)
class _ToolActivity:
    name: str
    arguments: str
    call_id: str | None
    started_at: float
    status: str = "running"
    detail: str = ""
    finished_at: float | None = None


class RichPresenter:
    """Render application events without interpreting dynamic values as markup."""

    def __init__(
        self,
        console: Console,
        *,
        app_info: AppInfo | None = None,
        live: bool = False,
    ) -> None:
        self.console = console
        self.app_info = app_info
        self.live_enabled = live and console.is_terminal
        self.verbose_tools = False
        self.show_thinking = False
        self._live: Live | None = None
        self._active = False
        self._status = "Thinking…"
        self._answer: list[str] = []
        self._thinking: list[str] = []
        self._tools: list[_ToolActivity] = []
        self._last_live_update = 0.0

    async def handle_event(self, event: AgentEvent) -> None:
        # Keep the direct-event behavior useful for callers that do not opt in to
        # managed turns, including the compatibility tests and embedders.
        if not self._active:
            self._print_legacy_event(event)
            return

        if isinstance(event, ThinkingOutput):
            self._thinking.append(event.content)
            self._status = "Thinking…"
            if not self.live_enabled and self.show_thinking:
                if event.starts_part:
                    self.console.print(Text("\nThinking  ", style="dim"), end="")
                self.console.print(Text(event.content, style="dim"), end="")
        elif isinstance(event, TextOutput):
            self._answer.append(event.content)
            self._status = "Responding…"
            if not self.live_enabled:
                if event.starts_part:
                    self.console.print(Text("\nGhostwheel\n", style="bold magenta"))
                self.console.print(Text(event.content), end="")
        elif isinstance(event, ToolStarted):
            self._tools.append(
                _ToolActivity(
                    name=event.name,
                    arguments=event.arguments,
                    call_id=event.call_id,
                    started_at=time.monotonic(),
                )
            )
            self._status = f"Running {event.name}…"
            if not self.live_enabled:
                self.console.print(self._tool_line(self._tools[-1]))
        elif isinstance(event, ToolFinished):
            activity = self._finish_tool(
                event.name,
                event.call_id,
                "succeeded",
                event.result,
            )
            self._status = "Thinking…"
            if not self.live_enabled:
                self.console.print(self._tool_line(activity))
                self._print_verbose_tool_detail(activity)
        elif isinstance(event, ToolFailed):
            activity = self._finish_tool(
                event.name,
                event.call_id,
                "failed",
                event.error,
            )
            self._status = f"{event.name} failed"
            if not self.live_enabled:
                self.console.print(self._tool_line(activity))
                self._print_verbose_tool_detail(activity)

        self._refresh_live()

    def turn_started(self, label: str = "Thinking…") -> None:
        self._reset_turn()
        self._active = True
        self._status = label
        if self.live_enabled:
            self._live = Live(
                self._active_renderable(),
                console=self.console,
                refresh_per_second=12,
                transient=True,
            )
            self._live.start(refresh=True)
        else:
            self.console.print(Text(f"\n{label}", style="dim"))

    def turn_cancelled(self) -> None:
        self._finish_activity()
        self.console.print(Text("Turn cancelled.", style="yellow"))
        self._reset_turn()

    def welcome(self) -> None:
        if self.app_info is None:
            self.console.print(
                Text(
                    "Ghostwheel chat. Type '/help' for commands or '/quit' to exit.",
                    style="dim",
                )
            )
            return

        heading = Text("Ghostwheel", style="bold magenta")
        details = Text()
        details.append(f"{self.app_info.provider}/{self.app_info.model}", style="cyan")
        details.append("  ·  ")
        details.append(self.app_info.workspace)
        details.append("  ·  tools: ")
        tool_style = "bold yellow" if self.app_info.tool_profile == "full" else "green"
        details.append(self.app_info.tool_profile.upper(), style=tool_style)
        if self.app_info.tool_profile == "full":
            details.append(" (unrestricted shell)", style="yellow")
        self.console.print(heading)
        self.console.print(details)
        self.console.print(Text("Type /help for commands.", style="dim"))

    def goodbye(self) -> None:
        self._stop_live()
        self.console.print(Text("\nGoodbye!", style="dim"))

    def help(self) -> None:
        lines = (
            ("/help", "show commands and keyboard shortcuts"),
            ("/review [path]", "review code; defaults to the workspace"),
            ("/retry", "repeat the previous chat or review"),
            ("/clear", "clear model conversation history"),
            ("/model", "show the active provider and model"),
            ("/tools", "show the active tool profile"),
            ("/quit", "exit Ghostwheel"),
        )
        body = Text()
        for index, (command, description) in enumerate(lines):
            if index:
                body.append("\n")
            body.append(f"{command:<22}", style="bold cyan")
            body.append(description)
        body.append("\n\nShortcuts\n", style="bold")
        body.append("Shift+Enter         insert a newline\n", style="dim")
        body.append("Ctrl+C              cancel a turn; quit while idle\n", style="dim")
        body.append(
            "Ctrl+O              toggle thinking and tool details\n", style="dim"
        )
        body.append("Tab                 complete commands and paths", style="dim")
        self.console.print(Panel(body, title="Commands", border_style="cyan"))

    def model_info(self) -> None:
        if self.app_info is None:
            self.console.print(
                Text("Model information is unavailable.", style="yellow")
            )
            return
        self.console.print(
            Text.assemble(
                Text("Model  ", style="bold"),
                Text(f"{self.app_info.provider}/{self.app_info.model}", style="cyan"),
            )
        )

    def tools_info(self) -> None:
        if self.app_info is None:
            self.console.print(Text("Tool information is unavailable.", style="yellow"))
            return
        body = Text.assemble(
            Text("Tool profile  ", style="bold"),
            Text(self.app_info.tool_profile, style="yellow"),
        )
        if self.app_info.tool_profile == "full":
            body.append("\nShell commands run with unrestricted environment access.")
        self.console.print(Panel(body, title="Tools", border_style="yellow"))

    def unknown_command(self, command: str, suggestion: str | None = None) -> None:
        message = Text.assemble(
            Text("Unknown command: ", style="yellow"),
            Text(command),
        )
        if suggestion:
            message.append(f"\nDid you mean {suggestion}?", style="dim")
        message.append("\nType /help to list commands.", style="dim")
        self.console.print(message)

    def retry_unavailable(self) -> None:
        self.console.print(Text("Nothing to retry yet.", style="yellow"))

    def history_cleared(self) -> None:
        self.console.print(Text("Conversation history cleared.", style="dim"))

    def history_compacted(self, dropped_turns: int) -> None:
        noun = "turn" if dropped_turns == 1 else "turns"
        self.console.print(
            Text(
                f"Context compacted: dropped {dropped_turns} {noun} to fit the budget.",
                style="dim",
            )
        )

    def turn_outcome(self, outcome: TurnOutcome) -> None:
        managed_live = self._active and self.live_enabled
        self._finish_activity()
        if isinstance(outcome, TurnSucceeded):
            if managed_live:
                self.console.print(Text("\nGhostwheel", style="bold magenta"))
                self.console.print(Markdown(outcome.output))
            elif not self._answer:
                self.console.print(Text("\nGhostwheel\n", style="bold magenta"))
                self.console.print(Text(outcome.output))
        elif isinstance(outcome, TurnNoResult):
            self.console.print(Text(outcome.message, style="yellow"))
        elif isinstance(outcome, TurnFailed):
            self._render_turn_failure(outcome)
        self._reset_turn()

    def review_outcome(self, outcome: ReviewOutcome) -> None:
        self._finish_activity()
        if isinstance(outcome, StructuredReview):
            self.console.print("")
            if outcome.used_fallback:
                self.console.print(
                    Text(
                        "Structured-output fallback was used for this review.",
                        style="dim",
                    )
                )
            render_review(outcome.review, self.console)
        elif isinstance(outcome, RawReview):
            body = Text()
            body.append(
                "Couldn't produce a structured review.\n",
                style="yellow",
            )
            body.append("Reason: ", style="dim")
            body.append(outcome.structured_failure, style="dim")
            body.append("\n\nShowing the raw review instead:\n\n", style="bold")
            body.append(outcome.prose)
            self.console.print(
                Panel(body, title="Structured Review Failed", border_style="yellow")
            )
        elif isinstance(outcome, ReviewFailed):
            body = Text(outcome.message)
            body.append(
                "\n\nCheck the review model configuration, then use /retry.",
                style="dim",
            )
            self.console.print(Panel(body, title="Review Failed", border_style="red"))
        self._reset_turn()

    def _render_turn_failure(self, outcome: TurnFailed) -> None:
        title, hint = {
            FailureKind.PROVIDER: (
                "Provider Error",
                "Check that the configured model server is running; use /model to inspect it.",
            ),
            FailureKind.CONFIGURATION: (
                "Configuration Error",
                "Check the GHOSTWHEEL_MODEL_* settings and provider name.",
            ),
            FailureKind.TOOL: (
                "Tool Error",
                "Inspect the tool output above, adjust the request, or use /retry.",
            ),
            FailureKind.MODEL_OUTPUT: (
                "Model Output Error",
                "The model returned an unsupported result; use /retry or change models.",
            ),
            FailureKind.UNKNOWN: (
                "Agent Failed",
                "Use /retry to try the turn again.",
            ),
        }[outcome.kind]
        body = Text(outcome.message)
        body.append(f"\n\n{hint}", style="dim")
        self.console.print(Panel(body, title=title, border_style="red"))

    def _active_renderable(self) -> Group:
        renderables: list[object] = [
            Spinner("dots", Text(self._status, style="bold cyan"))
        ]
        for activity in self._tools[-5:]:
            renderables.append(self._tool_line(activity))
            if self.verbose_tools and activity.detail:
                renderables.append(self._tool_detail_panel(activity))
        if self.show_thinking and self._thinking:
            renderables.append(
                Panel(
                    Text(_preview("".join(self._thinking), 800), style="dim"),
                    title="Thinking",
                    border_style="dim",
                )
            )
        if self._answer:
            renderables.extend(
                (
                    Text("Ghostwheel", style="bold magenta"),
                    Markdown("".join(self._answer)),
                )
            )
        return Group(*renderables)

    def _refresh_live(self, *, force: bool = False) -> None:
        if self._live is None:
            return
        now = time.monotonic()
        if not force and now - self._last_live_update < 0.05:
            return
        self._last_live_update = now
        self._live.update(self._active_renderable(), refresh=force)

    def _finish_activity(self) -> None:
        if not self._active:
            return
        self._refresh_live(force=True)
        self._stop_live()
        if self.live_enabled:
            for activity in self._tools:
                self.console.print(self._tool_line(activity))
                self._print_verbose_tool_detail(activity)
            if self.show_thinking and self._thinking:
                self.console.print(
                    Panel(
                        Text("".join(self._thinking), style="dim"),
                        title="Thinking",
                        border_style="dim",
                    )
                )

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _finish_tool(
        self,
        name: str,
        call_id: str | None,
        status: str,
        detail: str,
    ) -> _ToolActivity:
        activity = next(
            (
                item
                for item in reversed(self._tools)
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
            activity = _ToolActivity(name, "", call_id, time.monotonic())
            self._tools.append(activity)
        activity.status = status
        activity.detail = _preview(detail, 800)
        activity.finished_at = time.monotonic()
        return activity

    def _tool_line(self, activity: _ToolActivity) -> Text:
        icon, style = {
            "running": ("▸", "yellow"),
            "succeeded": ("✓", "green"),
            "failed": ("✗", "red"),
        }[activity.status]
        line = Text(f"  {icon} ", style=style)
        line.append(activity.name, style=f"bold {style}")
        argument = _primary_argument(activity.arguments)
        if argument:
            line.append("  ")
            line.append(_preview(argument, 72))
        if activity.finished_at is not None:
            line.append("  ·  ", style="dim")
            line.append(
                _duration(activity.finished_at - activity.started_at),
                style="dim",
            )
        if activity.status == "failed" and activity.detail:
            line.append("  ·  ", style="red")
            line.append(_preview(" ".join(activity.detail.split()), 100), style="red")
        return line

    def _print_verbose_tool_detail(self, activity: _ToolActivity) -> None:
        if not self.verbose_tools:
            return
        self.console.print(self._tool_detail_panel(activity))

    def _tool_detail_panel(self, activity: _ToolActivity) -> Panel:
        body = Text()
        body.append("Arguments\n", style="bold")
        body.append(activity.arguments or "(none)")
        if activity.detail:
            body.append("\n\nResult\n", style="bold")
            body.append(activity.detail)
        return Panel(
            body,
            title=Text(f"{activity.name} details", style="dim"),
            border_style="dim",
            padding=(0, 1),
        )

    def _print_legacy_event(self, event: AgentEvent) -> None:
        if isinstance(event, ThinkingOutput):
            if not self.show_thinking:
                return
            if event.starts_part:
                self.console.print(Text("\n💭 ", style="dim"), end="")
            self.console.print(Text(event.content, style="dim"), end="")
        elif isinstance(event, TextOutput):
            if event.starts_part:
                self.console.print(Text("\n💬 "), end="")
            self.console.print(Text(event.content), end="")
        elif isinstance(event, ToolStarted):
            arguments = _preview(event.arguments, 80)
            line = Text("\n🔧 ", style="yellow")
            line.append(event.name, style="yellow")
            line.append(f"({arguments})", style="yellow")
            self.console.print(line)
        elif isinstance(event, ToolFinished):
            result = _preview(event.result, 120)
            line = Text("← ", style="green")
            line.append(event.name, style="green")
            line.append(": ", style="green")
            line.append(result, style="green")
            self.console.print(line)
        elif isinstance(event, ToolFailed):
            error = _preview(event.error, 120)
            line = Text("← ", style="red")
            line.append(event.name, style="red")
            line.append(" failed: ", style="red")
            line.append(error, style="red")
            self.console.print(line)

    def _reset_turn(self) -> None:
        self._stop_live()
        self._active = False
        self._status = "Thinking…"
        self._answer = []
        self._thinking = []
        self._tools = []
        self._last_live_update = 0.0


def _primary_argument(arguments: str) -> str:
    if not arguments:
        return ""
    try:
        parsed = ast.literal_eval(arguments)
    except SyntaxError, ValueError:
        return _preview(" ".join(arguments.split()), 72)
    if not isinstance(parsed, dict):
        return _preview(" ".join(str(parsed).split()), 72)
    for key in ("path", "command", "pattern", "query", "paths"):
        value = parsed.get(key)
        if value not in (None, ""):
            return " ".join(str(value).split())
    return ""


def _duration(seconds: float) -> str:
    if seconds < 0.001:
        return "<1 ms"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.1f} s"


def _preview(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit] + "…"
