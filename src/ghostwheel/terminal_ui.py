"""Inline terminal UI with Rich output and prompt_toolkit input.

Unlike a full-screen TUI, this adapter never enters the terminal's alternate
screen. Completed turns remain in native scrollback, while prompt_toolkit owns
only the active composer at the bottom of the terminal.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app_or_none
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.output import Output
from prompt_toolkit.utils import get_cwidth
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.text import Text

from ghostwheel.app_info import AppInfo, ModelInfo, ToolSetInfo
from ghostwheel.cancellation import TurnCancellation
from ghostwheel.controller import (
    CancellationPort,
    ReviewPort,
    TerminalSessionPort,
    run_command_loop,
)
from ghostwheel.events import (
    AgentEvent,
    TextOutput,
    ToolFailed,
    ToolFinished,
    ToolStarted,
)
from ghostwheel.presentation import (
    ToolActivity,
    TurnState,
    duration,
    failure_presentation,
    format_token_count,
    preview,
    primary_argument,
)
from ghostwheel.rendering import (
    render_review,
    sanitize_terminal_line,
    sanitize_terminal_text,
)
from ghostwheel.review import RawReview, ReviewFailed, ReviewOutcome, StructuredReview
from ghostwheel.runtime_contracts import (
    TurnFailed,
    TurnNoResult,
    TurnOutcome,
    TurnSucceeded,
)
from ghostwheel.terminal_composer import (
    ComposerWarning,
    PrivateHistory,
    TerminalComposer,
    default_history_path,
)
from ghostwheel.terminal_io import (
    ActiveTurnInputMonitor,
    RawTerminalGuard,
    RedirectedLineReader,
    supports_prompt_toolkit,
)

__all__ = ["TerminalUI", "default_history_path"]

_LIVE_TOOL_LIMIT = 2
_LIVE_ANSWER_MAX_LINES = 1
_LIVE_ANSWER_MAX_CHARACTERS = 400
# prompt_toolkit's default VT-prefix timeout, used only when no composer
# application exists to provide its configured ``ttimeoutlen``.
_DEFAULT_TTIMEOUT_SECONDS = 0.5


class _TerminalCancellation:
    """Add terminal-key cancellation around the UI-neutral controller."""

    def __init__(self, ui: TerminalUI, delegate: CancellationPort) -> None:
        self._ui = ui
        self._delegate = delegate

    def cancel(self) -> bool:
        return self._delegate.cancel()

    async def run(self, awaitable: Awaitable[object]) -> object:
        with self._ui._capture_turn_input(self._delegate):
            return await self._delegate.run(awaitable)


class TerminalUI:
    """Combined controller presenter and async inline input reader.

    ``TerminalUI`` is both a ``PresenterPort`` and an ``InputPort``. Bind an
    event dispatcher to :meth:`handle_event`, then either call :meth:`run` or
    pass this object as both ports to ``run_command_loop``.
    """

    def __init__(
        self,
        console: Console,
        *,
        session: TerminalSessionPort,
        app_info: AppInfo,
        history_path: Path | None = None,
        vim_mode: bool = True,
        interactive: bool | None = None,
        input_stream: TextIO | None = None,
        prompt_input: Input | None = None,
        prompt_output: Output | None = None,
        live: bool = True,
    ) -> None:
        self.console = console
        self.session = session
        self.app_info = app_info
        self.vim_mode = vim_mode
        self.input_stream = sys.stdin if input_stream is None else input_stream
        terminal_ready = (
            _isatty(self.input_stream)
            and console.is_terminal
            and not console.is_dumb_terminal
            and supports_prompt_toolkit()
        )
        self.interactive = (
            prompt_input is not None or terminal_ready
            if interactive is None
            else interactive
        )
        self.live_enabled = (
            live
            and self.interactive
            and console.is_terminal
            and not console.is_dumb_terminal
        )

        self._live: Live | None = None
        self._active = False
        self._turn_uses_live = False
        self._turn = TurnState()
        self._last_live_update = 0.0
        self._stream_at_line_start = True

        self._prompt_active = False

        self._composer = TerminalComposer(
            workspace=Path(app_info.workspace).expanduser(),
            history_path=history_path,
            vim_mode=vim_mode,
            prompt_input=prompt_input,
            prompt_output=prompt_output,
            bottom_toolbar=self._bottom_toolbar,
            rprompt=self._rprompt,
        )
        self._terminal_guard = RawTerminalGuard(
            self.input_stream,
            externally_managed_input=prompt_input is not None,
            is_active=lambda: self._active or self._prompt_active,
        )
        self._turn_input_monitor = ActiveTurnInputMonitor(
            get_input=lambda: self._composer.input_for_turn(self.input_stream),
            get_timeout=self._turn_input_timeout,
            terminal_guard=self._terminal_guard,
        )
        self._redirected_reader = RedirectedLineReader(self.input_stream)

    async def handle_event(self, event: AgentEvent) -> None:
        """Reduce one streamed event into the active turn presentation."""

        if not self._active:
            raise RuntimeError("received an agent event outside an active turn")

        activity = self._turn.apply(event)
        if not self._turn_uses_live:
            if isinstance(event, TextOutput):
                if event.starts_part:
                    self.console.print(
                        Text("\nGhostwheel\n", style="bold magenta"),
                        end="",
                    )
                    self._stream_at_line_start = True
                safe_content = sanitize_terminal_text(event.content)
                self.console.print(Text(safe_content), end="", soft_wrap=True)
                if safe_content:
                    self._stream_at_line_start = safe_content.endswith("\n")
            elif isinstance(event, ToolStarted | ToolFinished | ToolFailed):
                assert activity is not None
                self._ensure_stream_line_start()
                self.console.print(self._tool_line(activity))
                self._stream_at_line_start = True

        self._refresh_live()

    def welcome(self) -> None:
        heading = Text("Ghostwheel", style="bold magenta")
        details = Text()
        details.append("chat ", style="dim")
        self._append_model(details, self.app_info.chat_model)
        details.append("  ·  review ", style="dim")
        self._append_model(details, self.app_info.review_model)
        details.append("  ·  ")
        details.append(sanitize_terminal_line(self.app_info.workspace))
        for label, tool_set in (
            ("chat", self.app_info.chat_tools),
            ("review", self.app_info.review_tools),
        ):
            details.append(f"  ·  {label} ")
            tool_style = "bold yellow" if tool_set.has_shell_access else "green"
            details.append(
                sanitize_terminal_line(tool_set.profile.upper()),
                style=tool_style,
            )
            if tool_set.has_shell_access:
                details.append(" (unrestricted shell)", style="yellow")
        self.console.print(heading)
        self.console.print(details)
        self.console.print(Text("Type /help for commands.", style="dim"))

    def goodbye(self) -> None:
        self._stop_live()
        self.console.print(Text("\nGoodbye!", style="dim"))

    async def run(
        self,
        reviews: ReviewPort,
        cancellation: CancellationPort | None = None,
    ) -> None:
        """Run the neutral command loop with this object as both UI ports."""

        cancellation = _TerminalCancellation(
            self,
            cancellation or TurnCancellation(),
        )
        try:
            await run_command_loop(
                self.session,
                reviews,
                presenter=self,
                input_reader=self,
                cancellation=cancellation,
            )
        finally:
            self.close()

    async def read(self) -> str:
        """Read one submitted prompt, falling back cleanly for redirected stdin."""

        if self._active:
            raise RuntimeError("cannot open the prompt while a turn is active")
        if self._turn_input_monitor.quit_requested:
            raise EOFError
        self._stop_live()
        self._terminal_guard.restore()

        if not self._use_prompt_toolkit():
            if not self._fallback_prompt_is_visible():
                return await self._redirected_reader.read()
            self._render_fallback_prompt()
            return await self._redirected_reader.read(
                on_terminal_line_cleared=self._render_fallback_prompt,
            )

        prompt_session = self._composer.get_session(self.input_stream)
        self._render_composer_warning()
        if self.vim_mode:
            prompt_session.app.vi_state.input_mode = InputMode.INSERT
        self._prompt_active = True
        self._terminal_guard.guard_prompt()
        try:
            value = await prompt_session.prompt_async()
        finally:
            self._prompt_active = False
            self._terminal_guard.restore()
            self._render_composer_warning()
        return value

    def turn_started(self, label: str = "Thinking…") -> None:
        """Start a bounded Live region only after the inline prompt has closed."""

        if self._prompt_active:
            raise RuntimeError("cannot start Rich Live while the prompt is active")
        self._reset_turn(label)
        self._active = True
        try:
            self._terminal_guard.silence()
            self._turn_uses_live = self.live_enabled
            if self._turn_uses_live:
                self._live = Live(
                    self._active_renderable(),
                    console=self.console,
                    refresh_per_second=12,
                    screen=False,
                    transient=True,
                    vertical_overflow="crop",
                )
                self._live.start(refresh=True)
            else:
                self.console.print(
                    Text(f"\n{sanitize_terminal_line(label)}", style="dim")
                )
        except BaseException:
            self._reset_turn()
            raise

    def help(self) -> None:
        lines = (
            ("/help", "show commands and keyboard shortcuts"),
            ("/review [path]", "review code; defaults to the workspace"),
            ("/retry", "repeat the previous chat or review"),
            ("/clear", "clear model conversation history"),
            ("/model", "show the active chat and review models"),
            ("/tools", "list available tools and active profiles"),
            ("/quit", "exit Ghostwheel"),
        )
        body = Text()
        for index, (command, description) in enumerate(lines):
            if index:
                body.append("\n")
            body.append(f"{command:<22}", style="bold cyan")
            body.append(description)
        if self._use_prompt_toolkit():
            body.append("\n\nComposer shortcuts\n", style="bold")
            body.append("Enter               submit\n", style="dim")
            body.append("Shift+Enter         insert a newline\n", style="dim")
            body.append("Ctrl+C              clear the current prompt\n", style="dim")
            body.append("Ctrl+D              quit\n", style="dim")
            body.append("Esc                 cancel the active turn\n", style="dim")
            body.append("Tab                 complete commands and paths", style="dim")
        else:
            body.append("\n\nLine-oriented input\n", style="bold")
            body.append("One prompt per line; end of input exits.", style="dim")
            if self._fallback_prompt_is_visible():
                body.append("\nEnter               submit", style="dim")
                body.append("\nCtrl+C              clear the current line", style="dim")
                body.append("\nCtrl+D              quit", style="dim")
                body.append("\nEsc                 cancel the active turn", style="dim")
        self.console.print(Panel(body, title="Commands", border_style="cyan"))

    def model_info(self) -> None:
        body = Text("Chat model    ", style="bold")
        self._append_model(body, self.app_info.chat_model)
        body.append("\nReview model  ", style="bold")
        self._append_model(body, self.app_info.review_model)
        self.console.print(Panel(body, title="Models", border_style="cyan"))

    @staticmethod
    def _append_model(body: Text, model: ModelInfo) -> None:
        provider = sanitize_terminal_line(model.provider)
        model_name = sanitize_terminal_line(model.model)
        body.append(f"{provider}/{model_name}", style="cyan")

    def tools_info(self) -> None:
        body = Text()
        self._append_tool_set(body, "Chat", self.app_info.chat_tools)
        body.append("\n\n")
        self._append_tool_set(body, "Review", self.app_info.review_tools)
        self.console.print(Panel(body, title="Tools", border_style="yellow"))

    @staticmethod
    def _append_tool_set(body: Text, title: str, tool_set: ToolSetInfo) -> None:
        body.append(title, style="bold")
        body.append("\nTool profile  ", style="bold")
        body.append(sanitize_terminal_line(tool_set.profile), style="yellow")
        body.append(f"\nAvailable tools ({len(tool_set.tools)})", style="bold")
        if tool_set.tools:
            safe_tools = tuple(
                (
                    sanitize_terminal_line(tool.name),
                    sanitize_terminal_line(tool.description),
                )
                for tool in tool_set.tools
            )
            name_width = max(len(name) for name, _description in safe_tools)
            for name, description in safe_tools:
                body.append("\n")
                body.append(f"{name:<{name_width}}", style="bold cyan")
                body.append("  ")
                body.append(description)
        else:
            body.append("\nNone for this profile.", style="dim")
        if tool_set.has_shell_access:
            body.append(
                "\nShell commands run with unrestricted environment access.",
                style="yellow",
            )

    def unknown_command(self, command: str, suggestion: str | None = None) -> None:
        message = Text.assemble(
            Text("Unknown command: ", style="yellow"),
            Text(sanitize_terminal_line(command)),
        )
        if suggestion:
            message.append(
                f"\nDid you mean {sanitize_terminal_line(suggestion)}?",
                style="dim",
            )
        message.append("\nType /help to list commands.", style="dim")
        self.console.print(message)

    def retry_unavailable(self) -> None:
        self.console.print(Text("Nothing to retry yet.", style="yellow"))

    def history_cleared(self) -> None:
        self.console.print(Text("Conversation history cleared.", style="dim"))

    def history_compacted(self, before_tokens: int, after_tokens: int) -> None:
        self.console.print(
            Text(
                "Context compacted: "
                f"{format_token_count(before_tokens)} → "
                f"~{format_token_count(after_tokens)}.",
                style="dim",
            )
        )

    def turn_cancelled(self) -> None:
        try:
            self._finish_activity()
            if not self._turn_input_monitor.quit_requested:
                self.console.print(Text("Turn cancelled.", style="yellow"))
        finally:
            self._reset_turn()

    def turn_outcome(self, outcome: TurnOutcome) -> None:
        managed_live = self._active and self._turn_uses_live
        try:
            self._finish_activity()
            if isinstance(outcome, TurnSucceeded):
                if managed_live:
                    self.console.print(Text("\nGhostwheel", style="bold magenta"))
                    self.console.print(Markdown(sanitize_terminal_text(outcome.output)))
                elif not self._turn.answer:
                    self.console.print(
                        Text("\nGhostwheel\n", style="bold magenta"),
                        end="",
                    )
                    self.console.print(
                        Text(sanitize_terminal_text(outcome.output)),
                        soft_wrap=True,
                    )
            elif isinstance(outcome, TurnNoResult):
                self.console.print(
                    Text(sanitize_terminal_text(outcome.message), style="yellow")
                )
            elif isinstance(outcome, TurnFailed):
                self._render_turn_failure(outcome)
        finally:
            self._reset_turn()

    def review_outcome(self, outcome: ReviewOutcome) -> None:
        try:
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
                body.append("Couldn't produce a structured review.\n", style="yellow")
                body.append("Reason: ", style="dim")
                body.append(
                    sanitize_terminal_text(outcome.structured_failure),
                    style="dim",
                )
                body.append("\n\nShowing the raw review instead:\n\n", style="bold")
                body.append(sanitize_terminal_text(outcome.prose))
                self.console.print(
                    Panel(body, title="Structured Review Failed", border_style="yellow")
                )
            elif isinstance(outcome, ReviewFailed):
                body = Text(sanitize_terminal_text(outcome.message))
                body.append(
                    "\n\nCheck the review model configuration, then use /retry.",
                    style="dim",
                )
                self.console.print(
                    Panel(body, title="Review Failed", border_style="red")
                )
        finally:
            self._reset_turn()

    @property
    def context_status(self) -> str:
        """Current model-context status used by the right-side prompt."""

        estimated_tokens = self.session.estimated_context_tokens
        context_window = self.session.context_window_tokens
        if not context_window:
            return ""
        estimate_marker = "~" if self.session.context_tokens_estimated else ""
        compaction_marker = "" if self.session.compaction_enabled else " · off"
        return (
            f"{estimate_marker}{format_token_count(estimated_tokens)}/"
            f"{format_token_count(context_window)}{compaction_marker}"
        )

    @property
    def _quit_requested(self) -> bool:
        """Expose monitor state as a compatibility seam for focused tests."""

        return self._turn_input_monitor.quit_requested

    def close(self) -> None:
        """Release UI-owned terminal input state and stop any active Live region."""

        try:
            self._stop_live()
        finally:
            try:
                self._terminal_guard.restore()
            finally:
                self._composer.close()

    def _render_composer_warning(self) -> None:
        warning = self._composer.take_warning()
        if warning is None:
            return
        self.console.print(
            Panel(
                self._composer_warning_text(warning),
                title="History Warning",
                border_style="yellow",
            )
        )

    @staticmethod
    def _composer_warning_text(warning: ComposerWarning) -> Text:
        body = Text(sanitize_terminal_text(warning.message), style="yellow")
        if warning.path is not None:
            body.append(
                f"\nPath: {sanitize_terminal_line(str(warning.path))}",
                style="dim",
            )
        if warning.detail:
            body.append(f"\n{sanitize_terminal_text(warning.detail)}", style="dim")
        body.append(
            "\nChoose a writable --history-file path or use --no-history.",
            style="dim",
        )
        return body

    def _render_turn_failure(self, outcome: TurnFailed) -> None:
        presentation = failure_presentation(outcome.kind)
        body = Text(sanitize_terminal_text(outcome.message))
        body.append(
            f"\n\n{sanitize_terminal_text(presentation.hint)}",
            style="dim",
        )
        self.console.print(
            Panel(
                body,
                title=sanitize_terminal_line(presentation.title),
                border_style="red",
            )
        )

    def _active_renderable(self) -> Group:
        renderables: list[object] = [
            Spinner(
                "dots",
                Text(
                    sanitize_terminal_line(self._turn.status),
                    style="bold cyan",
                    no_wrap=True,
                    overflow="ellipsis",
                ),
            )
        ]
        for activity in self._turn.tools[-_LIVE_TOOL_LIMIT:]:
            renderables.append(self._tool_line(activity))
        if self._turn.answer:
            renderables.extend(
                (
                    Text("Ghostwheel", style="bold magenta"),
                    Text(
                        _live_answer_tail(sanitize_terminal_text(self._turn.answer)),
                        no_wrap=True,
                        overflow="ellipsis",
                    ),
                )
            )
        renderables.append(
            Rule(
                Text(self._status_text(), style="dim"),
                characters="─",
                style="dim",
                align="right",
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
        finished = False
        try:
            try:
                self._refresh_live(force=True)
            finally:
                self._stop_live()
            if self._turn_uses_live:
                for activity in self._turn.tools:
                    self.console.print(self._tool_line(activity))
            else:
                self._ensure_stream_line_start()
            finished = True
        finally:
            # A rendering failure must never leave active-turn raw mode in
            # place. On success, _reset_turn() restores it after the complete
            # outcome has rendered, preserving active-turn typeahead handling.
            if not finished:
                self._terminal_guard.restore()

    def _ensure_stream_line_start(self) -> None:
        """Finish a partial streamed line before rendering another UI element."""

        if not self._stream_at_line_start:
            self.console.print("")
            self._stream_at_line_start = True

    def _stop_live(self) -> None:
        live = self._live
        self._live = None
        if live is not None:
            live.stop()

    def _tool_line(self, activity: ToolActivity) -> Text:
        icon, style = {
            "running": ("▸", "yellow"),
            "succeeded": ("✓", "green"),
            "failed": ("✗", "red"),
        }[activity.status]
        line = Text(
            f"  {icon} ",
            style=style,
            no_wrap=True,
            overflow="ellipsis",
        )
        line.append(sanitize_terminal_line(activity.name), style=f"bold {style}")
        argument = sanitize_terminal_line(primary_argument(activity.arguments))
        if argument:
            line.append("  ")
            line.append(preview(argument, 72))
        if activity.finished_at is not None:
            line.append("  ·  ", style="dim")
            line.append(
                duration(activity.finished_at - activity.started_at),
                style="dim",
            )
        if activity.status == "failed" and activity.detail:
            line.append("  ·  ", style="red")
            detail = sanitize_terminal_text(activity.detail)
            line.append(preview(" ".join(detail.split()), 100), style="red")
        return line

    def _reset_turn(self, label: str = "Thinking…") -> None:
        try:
            self._stop_live()
        finally:
            try:
                self._terminal_guard.restore()
            finally:
                self._active = False
                self._turn_uses_live = False
                self._turn.reset(label)
                self._last_live_update = 0.0
                self._stream_at_line_start = True

    @contextmanager
    def _capture_turn_input(
        self,
        cancellation: CancellationPort,
    ) -> Iterator[None]:
        """Delegate active-turn key handling to the terminal input monitor."""

        with self._turn_input_monitor.capture(cancellation):
            yield

    def _use_prompt_toolkit(self) -> bool:
        if not self.interactive:
            return False
        if self._composer.prompt_input is not None:
            return True
        return _isatty(self.input_stream) and (
            self.console.is_terminal or self._composer.prompt_output is not None
        )

    def _fallback_prompt_is_visible(self) -> bool:
        """Return whether cooked input and Rich output share terminal UX."""

        return _isatty(self.input_stream) and _isatty(self.console.file)

    def _render_fallback_prompt(self) -> None:
        """Print a native-scrollback prompt for the cooked TTY reader."""

        self.console.print(Text("\n> ", style="bold cyan"), end="")

    def _turn_input_timeout(self, prompt_input: Input) -> float:
        """Use the composer's VT-prefix timeout when it owns this input."""

        if (
            self._composer.session is not None
            and self._composer.session.app.input is prompt_input
        ):
            return self._composer.session.app.ttimeoutlen
        return _DEFAULT_TTIMEOUT_SECONDS

    def _get_prompt_history(self) -> PrivateHistory:
        """Return composer history (kept as a test seam)."""

        return self._composer.get_history()

    def _get_prompt_session(self) -> PromptSession[str]:
        """Return the prompt session (kept as a test seam)."""

        return self._composer.get_session(self.input_stream)

    def _rprompt(self) -> FormattedText:
        app = get_app_or_none()
        if app is not None and app.renderer.height_is_known:
            return FormattedText()
        status = self._status_text()
        return (
            FormattedText([("class:rprompt", f" {status} ")])
            if status
            else FormattedText()
        )

    def _bottom_toolbar(self) -> FormattedText:
        status = self._status_text()
        value = f" {status} " if status else ""
        app = get_app_or_none()
        width = self.console.width
        if app is not None:
            width = app.output.get_size().columns
        rule = "─" * max(1, width - get_cwidth(value))
        return FormattedText(
            [
                ("class:status.rule", rule),
                ("class:status.value", value),
            ]
        )

    def _status_text(self) -> str:
        parts = [self.context_status] if self.context_status else []
        if self.vim_mode:
            parts.append(self._mode_label())
        return " · ".join(parts)

    def _mode_label(self) -> str:
        app = get_app_or_none()
        if app is None and self._prompt_active and self._composer.session is not None:
            app = self._composer.session.app
        input_mode = (
            app.vi_state.input_mode
            if self._prompt_active and app is not None
            else InputMode.INSERT
        )
        return {
            InputMode.INSERT: "I",
            InputMode.INSERT_MULTIPLE: "I",
            InputMode.NAVIGATION: "N",
            InputMode.REPLACE: "R",
            InputMode.REPLACE_SINGLE: "R",
        }[input_mode]


def _live_answer_tail(answer: str) -> str:
    """Return a small raw-text tail suitable for a transient Live region."""

    truncated = len(answer) > _LIVE_ANSWER_MAX_CHARACTERS
    tail = answer[-_LIVE_ANSWER_MAX_CHARACTERS:]
    lines = tail.splitlines()
    if len(lines) > _LIVE_ANSWER_MAX_LINES:
        lines = lines[-_LIVE_ANSWER_MAX_LINES:]
        truncated = True
    tail = "\n".join(lines)
    if truncated:
        return "… " + tail.lstrip()
    return tail


def _isatty(stream: object) -> bool:
    try:
        return bool(getattr(stream, "isatty")())
    except AttributeError, OSError, TypeError:
        return False
