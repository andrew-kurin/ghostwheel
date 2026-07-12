"""Inline terminal UI with Rich output and prompt_toolkit input.

Unlike a full-screen TUI, this adapter never enters the terminal's alternate
screen. Completed turns remain in native scrollback, while prompt_toolkit owns
only the active composer at the bottom of the terminal.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import signal
import stat
import sys
import termios
import time
from collections.abc import Iterable
from pathlib import Path
from types import FrameType
from typing import TextIO

from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app_or_none
from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
    PathCompleter,
)
from prompt_toolkit.document import Document
from prompt_toolkit.enums import DEFAULT_BUFFER, EditingMode
from prompt_toolkit.filters import (
    emacs_insert_mode,
    has_focus,
    vi_insert_mode,
    vi_insert_multiple_mode,
    vi_navigation_mode,
)
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import History
from prompt_toolkit.input import Input
from prompt_toolkit.input.defaults import create_input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.text import Text

from ghostwheel.app_info import AppInfo
from ghostwheel.cancellation import TurnCancellation
from ghostwheel.controller import (
    COMMANDS,
    CancellationPort,
    ReviewPort,
    SessionPort,
    run_command_loop,
)
from ghostwheel.events import (
    AgentEvent,
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

__all__ = ["TerminalUI", "default_history_path"]

_LIVE_TOOL_LIMIT = 2
_LIVE_ANSWER_MAX_LINES = 1
_LIVE_ANSWER_MAX_CHARACTERS = 400
_TERMIOS_LOCAL_FLAGS = 3
_TERMINAL_GUARDED_SIGNALS = (
    signal.SIGHUP,
    signal.SIGQUIT,
    signal.SIGTERM,
    signal.SIGTSTP,
)


def default_history_path() -> Path:
    """Return the private per-user prompt history location."""

    state_home = os.environ.get("XDG_STATE_HOME")
    base = Path(state_home).expanduser() if state_home else Path.home() / ".local/state"
    return base / "ghostwheel" / "input-history"


class _InputHistory:
    """Private prompt history compatible with prompt_toolkit's file format."""

    def __init__(self, path: Path | None) -> None:
        self.path = path.expanduser() if path is not None else None
        self.entries = self._load()
        if self.path is not None:
            self._ensure_file()

    def append(self, value: str) -> None:
        if not value.strip():
            return
        self.entries.append(value)
        if self.path is None:
            return
        self._ensure_file()
        with self.path.open("a", encoding="utf-8") as history_file:
            history_file.write(f"\n# {dt.datetime.now().isoformat()}\n")
            for line in value.split("\n"):
                history_file.write(f"+{line}\n")

    def _load(self) -> list[str]:
        if self.path is None or not self.path.exists():
            return []
        entries: list[str] = []
        lines: list[str] = []
        for line in self.path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            if line.startswith("+"):
                lines.append(line[1:])
            elif lines:
                entries.append("\n".join(lines))
                lines = []
        if lines:
            entries.append("\n".join(lines))
        return entries

    def _ensure_file(self) -> None:
        assert self.path is not None
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(
            self.path,
            os.O_CREAT | os.O_APPEND | os.O_WRONLY,
            0o600,
        )
        os.close(descriptor)
        self.path.chmod(0o600)


class _PrivateHistory(History):
    """Expose the private history through prompt_toolkit's history protocol."""

    def __init__(self, history: _InputHistory) -> None:
        super().__init__()
        self._history = history

    def load_history_strings(self) -> Iterable[str]:
        # prompt_toolkit consumes newest-first; the private store is oldest-first.
        return reversed(tuple(self._history.entries))

    def store_string(self, string: str) -> None:
        self._history.append(string)

    def append_string(self, string: str) -> None:
        # Match the on-disk store's whitespace filtering in the in-memory view.
        if string.strip():
            super().append_string(string)


class _GhostwheelCompleter(Completer):
    """Complete slash commands and paths following ``/review``."""

    def __init__(self, workspace: Path) -> None:
        self._paths = PathCompleter(
            get_paths=lambda: [str(workspace)],
            expanduser=True,
        )

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if document.text_after_cursor or "\n" in text:
            return

        review_prefix = "/review "
        if text.lower().startswith(review_prefix):
            path_text = text[len(review_prefix) :]
            path_document = Document(path_text, cursor_position=len(path_text))
            for completion in self._paths.get_completions(
                path_document,
                complete_event,
            ):
                completion_text = completion.text
                if completion.display_text.endswith(
                    os.sep
                ) and not completion_text.endswith(os.sep):
                    completion_text += os.sep
                yield Completion(
                    completion_text,
                    start_position=completion.start_position,
                    display=completion.display,
                    display_meta=completion.display_meta,
                )
            return

        if not text.startswith("/") or any(character.isspace() for character in text):
            return
        normalized = text.lower()
        for command in COMMANDS:
            if command.startswith(normalized):
                yield Completion(
                    command,
                    start_position=-len(text),
                    display=command,
                )


def _key_bindings() -> KeyBindings:
    bindings = KeyBindings()
    composer_focused = has_focus(DEFAULT_BUFFER)
    editing_focused = composer_focused & (
        emacs_insert_mode | vi_insert_mode | vi_insert_multiple_mode
    )
    history_navigation_focused = composer_focused & (emacs_insert_mode | vi_insert_mode)

    @bindings.add(Keys.ControlM, filter=composer_focused, eager=True)
    def _submit(event: KeyPressEvent) -> None:
        event.current_buffer.validate_and_handle()

    @bindings.add(Keys.ControlJ, filter=editing_focused, eager=True)
    def _newline(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add(Keys.ControlI, filter=editing_focused, eager=True)
    def _complete(event: KeyPressEvent) -> None:
        buffer = event.current_buffer
        if buffer.complete_state is None:
            buffer.start_completion(select_first=True)
        else:
            buffer.complete_next()

    @bindings.add(Keys.Up, filter=history_navigation_focused, eager=True)
    def _history_previous_or_cursor_up(event: KeyPressEvent) -> None:
        event.current_buffer.auto_up(count=event.arg)

    @bindings.add(Keys.Down, filter=history_navigation_focused, eager=True)
    def _history_next_or_cursor_down(event: KeyPressEvent) -> None:
        buffer = event.current_buffer
        working_index = buffer.working_index
        buffer.auto_down(count=event.arg)
        if buffer.working_index != working_index:
            buffer.cursor_position = len(buffer.text)

    @bindings.add(
        Keys.Up,
        filter=composer_focused & vi_navigation_mode,
        eager=True,
    )
    def _cursor_up_in_vi_navigation(event: KeyPressEvent) -> None:
        event.current_buffer.cursor_up(count=event.arg)

    @bindings.add(
        Keys.Down,
        filter=composer_focused & vi_navigation_mode,
        eager=True,
    )
    def _cursor_down_in_vi_navigation(event: KeyPressEvent) -> None:
        event.current_buffer.cursor_down(count=event.arg)

    @bindings.add(
        Keys.ControlJ,
        filter=composer_focused & vi_navigation_mode,
        eager=True,
    )
    @bindings.add(
        Keys.ControlI,
        filter=composer_focused & vi_navigation_mode,
        eager=True,
    )
    def _ignore_insert_shortcuts_in_vi_navigation(_event: KeyPressEvent) -> None:
        pass

    @bindings.add(Keys.ControlQ, eager=True, is_global=True)
    def _quit(event: KeyPressEvent) -> None:
        event.app.exit(exception=EOFError())

    return bindings


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

        self._history_store = _InputHistory(history_path)
        self._history = _PrivateHistory(self._history_store)
        self._prompt_input = prompt_input
        self._prompt_output = prompt_output
        self._owned_prompt_input: Input | None = None
        self._prompt_session: PromptSession[str] | None = None
        self._prompt_active = False
        self._line_buffer = bytearray()
        self._line_eof = False
        self._terminal_input_descriptor: int | None = None
        self._terminal_input_attributes: list[object] | None = None
        self._terminal_signal_handlers: dict[int, object] = {}

        workspace = Path(app_info.workspace).expanduser()
        self._completer = _GhostwheelCompleter(workspace)

    async def handle_event(self, event: AgentEvent) -> None:
        """Reduce one streamed event into the active turn presentation."""

        if not self._active:
            raise RuntimeError("received an agent event outside an active turn")

        activity = self._turn.apply(event)
        if not self._turn_uses_live:
            if isinstance(event, ToolStarted | ToolFinished | ToolFailed):
                assert activity is not None
                self.console.print(self._tool_line(activity))

        self._refresh_live()

    def welcome(self) -> None:
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

    async def run(
        self,
        reviews: ReviewPort,
        cancellation: CancellationPort | None = None,
    ) -> None:
        """Run the neutral command loop with this object as both UI ports."""

        cancellation = cancellation or TurnCancellation(handle_sigint=True)
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
        self._stop_live()
        self._restore_terminal_input()

        if not self._use_prompt_toolkit():
            return await self._read_stream_line()

        prompt_session = self._get_prompt_session()
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
            ("/tools", "list available tools and the active profile"),
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
        body.append("Ctrl+J              insert a newline\n", style="dim")
        body.append(
            "Ctrl+Q              quit from the interactive prompt\n", style="dim"
        )
        body.append("Ctrl+C              cancel a turn; quit while idle\n", style="dim")
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
        body = Text.assemble(
            Text("Tool profile  ", style="bold"),
            Text(self.app_info.tool_profile, style="yellow"),
        )
        body.append(f"\n\nAvailable tools ({len(self.app_info.tools)})", style="bold")
        if self.app_info.tools:
            name_width = max(len(tool.name) for tool in self.app_info.tools)
            for tool in self.app_info.tools:
                body.append("\n")
                body.append(f"{tool.name:<{name_width}}", style="bold cyan")
                body.append("  ")
                body.append(tool.description)
        else:
            body.append("\nNone for this profile.", style="dim")
        if self.app_info.tool_profile in {"full", "shell-only"}:
            body.append(
                "\n\nShell commands run with unrestricted environment access.",
                style="yellow",
            )
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
        self.console.print(Text("Turn cancelled.", style="yellow"))
        self._reset_turn()

    def turn_outcome(self, outcome: TurnOutcome) -> None:
        self._finish_activity()
        if isinstance(outcome, TurnSucceeded):
            self.console.print(Text("\nGhostwheel", style="bold magenta"))
            self.console.print(Markdown(outcome.output))
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
        self._prompt_session = None
        if self._owned_prompt_input is not None:
            self._owned_prompt_input.close()
            self._owned_prompt_input = None

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
            self._prompt_input is not None
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
                termios.ECHO | termios.ECHONL | termios.ICANON
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

    def _guard_prompt_terminal(self) -> None:
        """Preserve cooked tty state across termination while the prompt is raw."""

        if (
            self._prompt_input is not None
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
        if self._prompt_input is not None:
            return True
        return _isatty(self.input_stream) and (
            self.console.is_terminal or self._prompt_output is not None
        )

    def _get_prompt_session(self) -> PromptSession[str]:
        if self._prompt_session is None:
            prompt_input = self._prompt_input
            if prompt_input is None and self.input_stream is not sys.stdin:
                assert self.input_stream is not None
                prompt_input = create_input(stdin=self.input_stream)
                self._owned_prompt_input = prompt_input

            self._prompt_session = PromptSession(
                message=FormattedText([("class:prompt", "\n> ")]),
                multiline=True,
                wrap_lines=True,
                editing_mode=(EditingMode.VI if self.vim_mode else EditingMode.EMACS),
                history=self._history,
                enable_history_search=False,
                completer=self._completer,
                complete_while_typing=False,
                enable_suspend=True,
                key_bindings=_key_bindings(),
                bottom_toolbar=self._bottom_toolbar,
                rprompt=self._rprompt,
                mouse_support=False,
                erase_when_done=False,
                reserve_space_for_menu=4,
                style=Style.from_dict(
                    {
                        "bottom-toolbar": "noreverse",
                        "prompt": "bold ansicyan",
                        "rprompt": "ansibrightblack",
                        "status.rule": "ansibrightblack",
                        "status.value": "ansibrightblack",
                    }
                ),
                input=prompt_input,
                output=self._prompt_output,
            )
        return self._prompt_session

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
        if app is None and self._prompt_active and self._prompt_session is not None:
            app = self._prompt_session.app
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
