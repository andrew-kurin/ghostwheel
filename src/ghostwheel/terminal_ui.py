"""Inline terminal UI with Rich output and prompt_toolkit input.

Unlike a full-screen TUI, this adapter never enters the terminal's alternate
screen. Completed turns remain in native scrollback, while prompt_toolkit owns
only the active composer at the bottom of the terminal.
"""

from __future__ import annotations

import asyncio
import os
import signal
import stat
import sys
import termios
import time
from collections.abc import Awaitable, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import FrameType
from typing import TextIO

from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app_or_none
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.input import Input
from prompt_toolkit.input.typeahead import get_typeahead
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import Output
from prompt_toolkit.utils import get_cwidth
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.text import Text

from ghostwheel.app_info import AppInfo, ToolSetInfo
from ghostwheel.cancellation import TurnCancellation
from ghostwheel.controller import (
    CancellationPort,
    ReviewPort,
    SessionPort,
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
from ghostwheel.rendering import render_review
from ghostwheel.review import RawReview, ReviewFailed, ReviewOutcome, StructuredReview
from ghostwheel.runtime_contracts import (
    TurnFailed,
    TurnNoResult,
    TurnOutcome,
    TurnSucceeded,
)
from ghostwheel.terminal_composer import (
    PrivateHistory,
    TerminalComposer,
    default_history_path,
)

__all__ = ["TerminalUI", "default_history_path"]

_LIVE_TOOL_LIMIT = 2
_LIVE_ANSWER_MAX_LINES = 1
_LIVE_ANSWER_MAX_CHARACTERS = 400
# prompt_toolkit's default VT-prefix timeout, used only when no composer
# application exists to provide its configured ``ttimeoutlen``.
_DEFAULT_TTIMEOUT_SECONDS = 0.5
_TERMIOS_LOCAL_FLAGS = 3
_TERMINAL_GUARDED_SIGNALS = (
    signal.SIGHUP,
    signal.SIGQUIT,
    signal.SIGTERM,
    signal.SIGTSTP,
)


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
        session: SessionPort,
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
        terminal_ready = _isatty(self.input_stream) and console.is_terminal
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

        self._prompt_active = False
        self._quit_requested = False
        self._line_buffer = bytearray()
        self._line_eof = False
        self._terminal_input_descriptor: int | None = None
        self._terminal_input_attributes: list[object] | None = None
        self._terminal_signal_handlers: dict[int, object] = {}

        self._composer = TerminalComposer(
            workspace=Path(app_info.workspace).expanduser(),
            history_path=history_path,
            vim_mode=vim_mode,
            prompt_input=prompt_input,
            prompt_output=prompt_output,
            bottom_toolbar=self._bottom_toolbar,
            rprompt=self._rprompt,
        )

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
                self.console.print(Text(event.content), end="", soft_wrap=True)
            elif isinstance(event, ToolStarted | ToolFinished | ToolFailed):
                assert activity is not None
                self.console.print(self._tool_line(activity))

        self._refresh_live()

    def welcome(self) -> None:
        heading = Text("Ghostwheel", style="bold magenta")
        details = Text()
        details.append(f"{self.app_info.provider}/{self.app_info.model}", style="cyan")
        details.append("  ·  ")
        details.append(self.app_info.workspace)
        for label, tool_set in (
            ("chat", self.app_info.chat_tools),
            ("review", self.app_info.review_tools),
        ):
            details.append(f"  ·  {label} ")
            tool_style = "bold yellow" if tool_set.has_shell_access else "green"
            details.append(tool_set.profile.upper(), style=tool_style)
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
        if self._quit_requested:
            raise EOFError
        self._stop_live()
        self._restore_terminal_input()

        if not self._use_prompt_toolkit():
            return await self._read_stream_line()

        prompt_session = self._composer.get_session(self.input_stream)
        if self.vim_mode:
            prompt_session.app.vi_state.input_mode = InputMode.INSERT
        self._prompt_active = True
        self._guard_prompt_terminal()
        try:
            return await prompt_session.prompt_async()
        finally:
            self._prompt_active = False
            self._restore_terminal_input()

    def turn_started(self, label: str = "Thinking…") -> None:
        """Start a bounded Live region only after the inline prompt has closed."""

        if self._prompt_active:
            raise RuntimeError("cannot start Rich Live while the prompt is active")
        self._reset_turn(label)
        self._active = True
        self._silence_turn_input()
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
            self.console.print(Text(f"\n{label}", style="dim"))

    def help(self) -> None:
        lines = (
            ("/help", "show commands and keyboard shortcuts"),
            ("/review [path]", "review code; defaults to the workspace"),
            ("/retry", "repeat the previous chat or review"),
            ("/clear", "clear model conversation history"),
            ("/model", "show the active provider and model"),
            ("/tools", "list available tools and active profiles"),
            ("/quit", "exit Ghostwheel"),
        )
        body = Text()
        for index, (command, description) in enumerate(lines):
            if index:
                body.append("\n")
            body.append(f"{command:<22}", style="bold cyan")
            body.append(description)
        body.append("\n\nShortcuts\n", style="bold")
        body.append("Enter               submit\n", style="dim")
        body.append("Shift+Enter         insert a newline\n", style="dim")
        body.append("Ctrl+C              clear the current prompt\n", style="dim")
        body.append("Ctrl+D              quit\n", style="dim")
        body.append("Esc                 cancel the active turn\n", style="dim")
        body.append("Tab                 complete commands and paths", style="dim")
        self.console.print(Panel(body, title="Commands", border_style="cyan"))

    def model_info(self) -> None:
        self.console.print(
            Text.assemble(
                Text("Model  ", style="bold"),
                Text(f"{self.app_info.provider}/{self.app_info.model}", style="cyan"),
            )
        )

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
        body.append(tool_set.profile, style="yellow")
        body.append(f"\nAvailable tools ({len(tool_set.tools)})", style="bold")
        if tool_set.tools:
            name_width = max(len(tool.name) for tool in tool_set.tools)
            for tool in tool_set.tools:
                body.append("\n")
                body.append(f"{tool.name:<{name_width}}", style="bold cyan")
                body.append("  ")
                body.append(tool.description)
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
        self._finish_activity()
        if not self._quit_requested:
            self.console.print(Text("Turn cancelled.", style="yellow"))
        self._reset_turn()

    def turn_outcome(self, outcome: TurnOutcome) -> None:
        managed_live = self._active and self._turn_uses_live
        self._finish_activity()
        if isinstance(outcome, TurnSucceeded):
            if managed_live:
                self.console.print(Text("\nGhostwheel", style="bold magenta"))
                self.console.print(Markdown(outcome.output))
            elif not self._turn.answer:
                self.console.print(
                    Text("\nGhostwheel\n", style="bold magenta"),
                    end="",
                )
                self.console.print(Text(outcome.output), soft_wrap=True)
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
            body.append("Couldn't produce a structured review.\n", style="yellow")
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

    @property
    def context_status(self) -> str:
        """Current model-context status used by the right-side prompt."""

        estimated_tokens = getattr(self.session, "estimated_context_tokens", 0)
        context_window = getattr(self.session, "context_window_tokens", 0)
        if not context_window:
            return ""
        estimate_marker = (
            "~" if getattr(self.session, "context_tokens_estimated", True) else ""
        )
        compaction_marker = (
            "" if getattr(self.session, "compaction_enabled", True) else " · off"
        )
        return (
            f"{estimate_marker}{format_token_count(estimated_tokens)}/"
            f"{format_token_count(context_window)}{compaction_marker}"
        )

    def close(self) -> None:
        """Release UI-owned terminal input state and stop any active Live region."""

        self._stop_live()
        self._restore_terminal_input()
        self._composer.close()

    def _render_turn_failure(self, outcome: TurnFailed) -> None:
        presentation = failure_presentation(outcome.kind)
        body = Text(outcome.message)
        body.append(f"\n\n{presentation.hint}", style="dim")
        self.console.print(Panel(body, title=presentation.title, border_style="red"))

    def _active_renderable(self) -> Group:
        renderables: list[object] = [
            Spinner(
                "dots",
                Text(
                    self._turn.status,
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
                        _live_answer_tail(self._turn.answer),
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
        self._refresh_live(force=True)
        self._stop_live()
        if self._turn_uses_live:
            for activity in self._turn.tools:
                self.console.print(self._tool_line(activity))

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

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
        line.append(activity.name, style=f"bold {style}")
        argument = primary_argument(activity.arguments)
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
            line.append(preview(" ".join(activity.detail.split()), 100), style="red")
        return line

    def _reset_turn(self, label: str = "Thinking…") -> None:
        self._stop_live()
        self._restore_terminal_input()
        self._active = False
        self._turn_uses_live = False
        self._turn.reset(label)
        self._last_live_update = 0.0

    def _silence_turn_input(self) -> None:
        """Prevent type-ahead from echoing into or leaking past an active turn."""

        if (
            self._composer.prompt_input is not None
            or self._terminal_input_attributes is not None
        ):
            return
        try:
            descriptor = self.input_stream.fileno()
            if not os.isatty(descriptor):
                return
            attributes = termios.tcgetattr(descriptor)
            quiet_attributes = attributes.copy()
            quiet_attributes[_TERMIOS_LOCAL_FLAGS] &= ~(
                termios.ECHO | termios.ECHONL | termios.ICANON | termios.ISIG
            )
        except AttributeError, OSError, termios.error:
            return

        if not self._install_terminal_signal_handlers():
            return
        self._terminal_input_descriptor = descriptor
        self._terminal_input_attributes = attributes
        try:
            termios.tcsetattr(descriptor, termios.TCSANOW, quiet_attributes)
        except OSError, termios.error:
            self._restore_terminal_input()

    @contextmanager
    def _capture_turn_input(
        self,
        cancellation: CancellationPort,
    ) -> Iterator[None]:
        """Drain active-turn keys, using Escape to cancel and Ctrl+D to quit."""

        prompt_input = self._turn_input()
        if prompt_input is None:
            # ``turn_started`` may already have disabled canonical input and
            # ISIG. Restore a normal, interruptible tty when monitoring cannot
            # be attached instead of leaving the process unresponsive.
            self._restore_terminal_input()
            yield
            return

        loop = asyncio.get_running_loop()
        flush_handle: asyncio.TimerHandle | None = None
        monitor_active = True
        cancellation_requested = False

        def cancel_if_active() -> None:
            nonlocal cancellation_requested
            if monitor_active and not cancellation.cancel():
                cancellation_requested = False

        def request_cancellation() -> None:
            nonlocal cancellation_requested
            if cancellation_requested:
                return
            cancellation_requested = True
            # Controller presenters start the activity immediately before
            # CancellationPort.run owns its task. Deferring avoids losing a key
            # that was typed in the same input chunk as prompt submission.
            loop.call_soon(cancel_if_active)

        def handle_keys(keys: Iterable[KeyPress]) -> None:
            key_presses = tuple(keys)
            for key_press in key_presses:
                if key_press.key == Keys.ControlD:
                    self._quit_requested = True
                    request_cancellation()
                    return
            # VT input represents Alt/Meta combinations as Escape followed by
            # the modified key in one decoded batch. Only a trailing Escape is
            # standalone; an Escape with a following key is a prefix.
            if key_presses and key_presses[-1].key == Keys.Escape:
                request_cancellation()

        def flush_pending_escape() -> None:
            nonlocal flush_handle
            flush_handle = None
            handle_keys(prompt_input.flush_keys())

        def input_ready() -> None:
            nonlocal flush_handle
            if flush_handle is not None:
                flush_handle.cancel()
                flush_handle = None
            keys = prompt_input.read_keys()
            handle_keys(keys)
            if prompt_input.closed:
                self._quit_requested = True
                request_cancellation()
            elif monitor_active:
                flush_handle = loop.call_later(
                    self._turn_input_timeout(prompt_input),
                    flush_pending_escape,
                )

        stack = ExitStack()
        try:
            stack.enter_context(prompt_input.raw_mode())
            stack.enter_context(prompt_input.attach(input_ready))
        except EOFError, NotImplementedError, OSError, RuntimeError:
            stack.close()
            self._restore_terminal_input()
            yield
            return

        try:
            # prompt_toolkit can retain decoded keys read in the same chunk as
            # Enter. A lone Escape may instead remain pending in its VT parser;
            # defer that parser flush so a split arrow/Meta sequence can finish.
            handle_keys(get_typeahead(prompt_input))
            flush_handle = loop.call_later(
                self._turn_input_timeout(prompt_input),
                flush_pending_escape,
            )
            yield
        finally:
            monitor_active = False
            if flush_handle is not None:
                flush_handle.cancel()
            stack.close()
            prompt_input.flush_keys()
            get_typeahead(prompt_input)

    def _guard_prompt_terminal(self) -> None:
        """Preserve cooked tty state across termination while the prompt is raw."""

        if (
            self._composer.prompt_input is not None
            or self._terminal_input_attributes is not None
        ):
            return
        try:
            descriptor = self.input_stream.fileno()
            if not os.isatty(descriptor):
                return
            attributes = termios.tcgetattr(descriptor)
        except AttributeError, OSError, termios.error:
            return
        if not self._install_terminal_signal_handlers():
            return
        self._terminal_input_descriptor = descriptor
        self._terminal_input_attributes = attributes

    def _restore_terminal_input(self) -> None:
        descriptor = self._terminal_input_descriptor
        attributes = self._terminal_input_attributes
        self._terminal_input_descriptor = None
        self._terminal_input_attributes = None
        if descriptor is not None and attributes is not None:
            try:
                termios.tcflush(descriptor, termios.TCIFLUSH)
            except OSError, termios.error:
                pass
            try:
                termios.tcsetattr(descriptor, termios.TCSANOW, attributes)
            except OSError, termios.error:
                pass
        self._restore_terminal_signal_handlers()

    def _install_terminal_signal_handlers(self) -> bool:
        if self._terminal_signal_handlers:
            return True

        installed: dict[int, object] = {}
        try:
            for signum in _TERMINAL_GUARDED_SIGNALS:
                installed[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle_terminal_signal)
        except OSError, RuntimeError, ValueError:
            for signum, previous in installed.items():
                try:
                    signal.signal(signum, previous)
                except OSError, RuntimeError, ValueError:
                    pass
            return False
        self._terminal_signal_handlers = installed
        return True

    def _restore_terminal_signal_handlers(self) -> None:
        handlers = self._terminal_signal_handlers
        self._terminal_signal_handlers = {}
        for signum, previous in handlers.items():
            try:
                signal.signal(signum, previous)
            except OSError, RuntimeError, ValueError:
                pass

    def _handle_terminal_signal(
        self,
        signum: int,
        frame: FrameType | None,
    ) -> None:
        """Restore the tty before honoring termination or job-control signals."""

        previous = self._terminal_signal_handlers.get(signum, signal.SIG_DFL)
        descriptor = self._terminal_input_descriptor
        original_attributes = self._terminal_input_attributes
        active_attributes: list[object] | None = None
        if descriptor is not None:
            try:
                active_attributes = termios.tcgetattr(descriptor)
            except OSError, termios.error:
                pass
        self._restore_terminal_input()

        if previous == signal.SIG_IGN:
            self._resume_guarded_terminal(
                descriptor,
                original_attributes,
                active_attributes,
            )
            return
        if previous == signal.SIG_DFL or previous is None:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            # SIGTSTP returns here after SIGCONT; termination signals do not.
            self._resume_guarded_terminal(
                descriptor,
                original_attributes,
                active_attributes,
            )
            return
        if callable(previous):
            previous(signum, frame)
            self._resume_guarded_terminal(
                descriptor,
                original_attributes,
                active_attributes,
            )

    def _resume_guarded_terminal(
        self,
        descriptor: int | None,
        original_attributes: list[object] | None,
        active_attributes: list[object] | None,
    ) -> None:
        if (
            not (self._active or self._prompt_active)
            or descriptor is None
            or original_attributes is None
            or active_attributes is None
            or not self._install_terminal_signal_handlers()
        ):
            return
        self._terminal_input_descriptor = descriptor
        self._terminal_input_attributes = original_attributes
        try:
            termios.tcsetattr(descriptor, termios.TCSANOW, active_attributes)
        except OSError, termios.error:
            self._restore_terminal_input()

    def _use_prompt_toolkit(self) -> bool:
        if not self.interactive:
            return False
        if self._composer.prompt_input is not None:
            return True
        return _isatty(self.input_stream) and (
            self.console.is_terminal or self._composer.prompt_output is not None
        )

    def _turn_input(self) -> Input | None:
        return self._composer.input_for_turn(self.input_stream)

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

    async def _read_stream_line(self) -> str:
        """Read a redirected line without leaving an uncancellable worker thread."""

        try:
            descriptor = self.input_stream.fileno()
            descriptor_mode = os.fstat(descriptor).st_mode
        except AttributeError, OSError, TypeError, ValueError:
            return self._read_stream_line_synchronously()
        if stat.S_ISREG(descriptor_mode):
            return self._read_stream_line_synchronously()

        while b"\n" not in self._line_buffer and not self._line_eof:
            chunk = await self._read_ready_chunk(descriptor)
            if chunk:
                self._line_buffer.extend(chunk)
            else:
                self._line_eof = True

        newline = self._line_buffer.find(b"\n")
        if newline >= 0:
            raw_value = bytes(self._line_buffer[: newline + 1])
            del self._line_buffer[: newline + 1]
        elif self._line_buffer:
            raw_value = bytes(self._line_buffer)
            self._line_buffer.clear()
        else:
            raise EOFError

        encoding = getattr(self.input_stream, "encoding", None) or "utf-8"
        errors = getattr(self.input_stream, "errors", None) or "strict"
        return raw_value.decode(encoding, errors).rstrip("\r\n")

    async def _read_ready_chunk(self, descriptor: int) -> bytes:
        loop = asyncio.get_running_loop()
        readable: asyncio.Future[bytes] = loop.create_future()

        def read_ready() -> None:
            if readable.done():
                return
            try:
                chunk = os.read(descriptor, 65_536)
            except BlockingIOError:
                return
            except OSError as error:
                readable.set_exception(error)
            else:
                readable.set_result(chunk)

        try:
            loop.add_reader(descriptor, read_ready)
        except (NotImplementedError, OSError) as error:
            raise RuntimeError(
                "redirected input requires a pollable POSIX file descriptor"
            ) from error
        try:
            return await readable
        finally:
            loop.remove_reader(descriptor)

    def _read_stream_line_synchronously(self) -> str:
        value = self.input_stream.readline()
        if value == "":
            raise EOFError
        return value.rstrip("\r\n")


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
