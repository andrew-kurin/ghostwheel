"""Prompt composition support for the inline terminal UI.

This module owns prompt_toolkit-specific concerns: persistent history, slash
and path completion, editor key bindings, and construction of the inline
``PromptSession``. Terminal presentation remains in
:mod:`ghostwheel.terminal_ui`; active-turn input handling lives in
:mod:`ghostwheel.terminal_io`.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from prompt_toolkit import PromptSession
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

from ghostwheel.controller import COMMANDS

_XTERM_SHIFT_ENTER = "\x1b[27;2;13~"


def default_history_path() -> Path | None:
    """Return a safe absolute prompt-history path, if one is available."""

    state_home = os.environ.get("XDG_STATE_HOME")
    configured_base = Path(state_home) if state_home else None
    if configured_base is not None and configured_base.is_absolute():
        return configured_base / "ghostwheel" / "input-history"

    try:
        base = Path.home() / ".local/state"
    except RuntimeError:
        return None
    if not base.is_absolute():
        return None
    return base / "ghostwheel" / "input-history"


@dataclass(frozen=True, slots=True)
class ComposerWarning:
    """A recoverable composer problem suitable for rendering by the UI."""

    code: str
    message: str
    path: Path | None = None
    detail: str | None = None


class InputHistory:
    """Private prompt history compatible with prompt_toolkit's file format."""

    def __init__(
        self,
        path: Path | None,
        *,
        on_error: Callable[[Path, OSError], None] | None = None,
    ) -> None:
        self.path = path.expanduser() if path is not None else None
        self._on_error = on_error
        self.entries: list[str] = []
        try:
            self.entries = self._load()
            if self.path is not None:
                self._ensure_file()
        except OSError as error:
            self._disable_persistence(error)

    def append(self, value: str) -> None:
        if not value.strip():
            return
        self.entries.append(value)
        if self.path is None:
            return
        try:
            self._ensure_file()
            with self.path.open("a", encoding="utf-8") as history_file:
                history_file.write(f"\n# {dt.datetime.now().isoformat()}\n")
                for line in value.split("\n"):
                    history_file.write(f"+{line}\n")
        except OSError as error:
            self._disable_persistence(error)

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

    def _disable_persistence(self, error: OSError) -> None:
        path = self.path
        if path is None:
            return
        self.path = None
        if self._on_error is not None:
            self._on_error(path, error)


class PrivateHistory(History):
    """Expose the private history through prompt_toolkit's history protocol."""

    def __init__(self, history: InputHistory) -> None:
        super().__init__()
        self.store = history

    def load_history_strings(self) -> Iterable[str]:
        # prompt_toolkit consumes newest-first; the private store is oldest-first.
        return reversed(tuple(self.store.entries))

    def store_string(self, string: str) -> None:
        self.store.append(string)

    def append_string(self, string: str) -> None:
        # Match the on-disk store's whitespace filtering in the in-memory view.
        if string.strip():
            super().append_string(string)


class GhostwheelCompleter(Completer):
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
    def _submit_or_modified_shift_enter(event: KeyPressEvent) -> None:
        data = event.key_sequence[-1].data
        if data == _XTERM_SHIFT_ENTER:
            event.current_buffer.insert_text("\n")
        else:
            event.current_buffer.validate_and_handle()

    @bindings.add(Keys.ControlJ, filter=composer_focused, eager=True)
    def _ignore_removed_newline_shortcut(_event: KeyPressEvent) -> None:
        pass

    @bindings.add(Keys.ControlC, filter=composer_focused, eager=True)
    def _clear_prompt(event: KeyPressEvent) -> None:
        event.current_buffer.reset()
        if event.app.editing_mode is EditingMode.VI:
            event.app.vi_state.input_mode = InputMode.INSERT

    @bindings.add(Keys.ControlD, eager=True, is_global=True)
    def _quit(event: KeyPressEvent) -> None:
        event.app.exit(exception=EOFError())

    @bindings.add(Keys.ControlQ, eager=True, is_global=True)
    def _ignore_removed_quit_shortcut(_event: KeyPressEvent) -> None:
        pass

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
        Keys.ControlI,
        filter=composer_focused & vi_navigation_mode,
        eager=True,
    )
    def _ignore_insert_shortcuts_in_vi_navigation(_event: KeyPressEvent) -> None:
        pass

    return bindings


class TerminalComposer:
    """Own the lifecycle and configuration of the inline prompt composer."""

    def __init__(
        self,
        *,
        workspace: Path,
        history_path: Path | None,
        vim_mode: bool,
        prompt_input: Input | None,
        prompt_output: Output | None,
        bottom_toolbar: Callable[[], FormattedText],
        rprompt: Callable[[], FormattedText],
    ) -> None:
        self.history_path = history_path
        self.vim_mode = vim_mode
        self.prompt_input = prompt_input
        self.prompt_output = prompt_output
        self.bottom_toolbar = bottom_toolbar
        self.rprompt = rprompt
        self.history_store: InputHistory | None = None
        self.history: PrivateHistory | None = None
        self.completer = GhostwheelCompleter(workspace)
        self.owned_input: Input | None = None
        self.session: PromptSession[str] | None = None
        self._warning: ComposerWarning | None = None

    def get_history(self) -> PrivateHistory:
        """Initialize persistent history only when the composer needs it."""

        if self.history is None:
            self.history_store = InputHistory(
                self.history_path,
                on_error=self._record_history_error,
            )
            self.history = PrivateHistory(self.history_store)
        return self.history

    def take_warning(self) -> ComposerWarning | None:
        """Return and consume the pending recoverable warning, if any.

        History failures disable persistence for the rest of this composer, so
        each failure is surfaced at most once. Callers can invoke this after
        session creation and after prompt completion to render late write
        failures without coupling presentation to the history implementation.
        """

        warning = self._warning
        self._warning = None
        return warning

    def _record_history_error(self, path: Path, error: OSError) -> None:
        if self._warning is not None:
            return
        self._warning = ComposerWarning(
            code="history_unavailable",
            message="Prompt history is unavailable; using in-memory history.",
            path=path,
            detail=str(error),
        )

    def get_session(self, input_stream: TextIO) -> PromptSession[str]:
        if self.session is None:
            prompt_input = self.prompt_input or self.owned_input
            if prompt_input is None and input_stream is not sys.stdin:
                prompt_input = create_input(stdin=input_stream)
                self.owned_input = prompt_input

            self.session = PromptSession(
                message=FormattedText([("class:prompt", "\n> ")]),
                multiline=True,
                wrap_lines=True,
                editing_mode=(EditingMode.VI if self.vim_mode else EditingMode.EMACS),
                history=self.get_history(),
                enable_history_search=False,
                completer=self.completer,
                complete_while_typing=False,
                enable_suspend=True,
                key_bindings=_key_bindings(),
                bottom_toolbar=self.bottom_toolbar,
                rprompt=self.rprompt,
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
                output=self.prompt_output,
            )
        return self.session

    def input_for_turn(self, input_stream: TextIO) -> Input | None:
        """Return VT input for active-turn shortcuts, even without a composer."""

        if self.session is not None:
            return self.session.app.input
        if self.prompt_input is not None:
            return self.prompt_input
        if not _isatty(input_stream):
            return None
        if self.owned_input is None:
            try:
                self.owned_input = create_input(stdin=input_stream)
            except EOFError, OSError, RuntimeError:
                return None
        return self.owned_input

    def close(self) -> None:
        self.session = None
        if self.owned_input is not None:
            self.owned_input.close()
            self.owned_input = None


def _isatty(stream: object) -> bool:
    try:
        return bool(getattr(stream, "isatty")())
    except AttributeError, OSError, TypeError:
        return False
