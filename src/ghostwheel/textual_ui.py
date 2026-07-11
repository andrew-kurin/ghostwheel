"""Persistent full-screen terminal UI for interactive Ghostwheel sessions."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.drivers.linux_driver import LinuxDriver
from textual.geometry import Size
from textual.message import Message
from textual.widgets import Static, TextArea

from ghostwheel.cancellation import TurnCancellation
from ghostwheel.events import (
    AgentEvent,
    TextOutput,
    ThinkingOutput,
    ToolFailed,
    ToolFinished,
    ToolStarted,
)
from ghostwheel.input_ui import COMMANDS, InputHistory
from ghostwheel.keyboard import macos_shift_pressed
from ghostwheel.rendering import review_renderables
from ghostwheel.review import RawReview, ReviewFailed, ReviewOutcome, StructuredReview
from ghostwheel.rich_ui import AppInfo, _duration, _primary_argument, _preview
from ghostwheel.session import (
    FailureKind,
    TurnFailed,
    TurnNoResult,
    TurnOutcome,
    TurnSucceeded,
)

if TYPE_CHECKING:
    from ghostwheel.review import ReviewService
    from ghostwheel.session import ChatSession


MODIFY_OTHER_KEYS_ENABLE = "\x1b[>4;2m"
MODIFY_OTHER_KEYS_RESET = "\x1b[>4;m"
COMPOSER_MIN_HEIGHT = 3
COMPOSER_BORDER_HEIGHT = 1
TRANSCRIPT_MIN_HEIGHT = 4


class GhostwheelTerminalDriver(LinuxDriver):
    """Add xterm extended-key fallback to Textual's Kitty negotiation."""

    def start_application_mode(self) -> None:
        super().start_application_mode()
        if self._writer_thread is not None:
            self.write(MODIFY_OTHER_KEYS_ENABLE)

    def stop_application_mode(self) -> None:
        if self._writer_thread is not None:
            self.write(MODIFY_OTHER_KEYS_RESET)
        super().stop_application_mode()


class QueueInputReader:
    """Feed submitted composer text into the existing asynchronous CLI loop."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    async def read(self) -> str:
        return await self._queue.get()

    def submit(self, value: str) -> None:
        self._queue.put_nowait(value)


class VimMode(str, Enum):
    """Editing states for the optional Vim-style composer."""

    INSERT = "insert"
    NORMAL = "normal"


@dataclass(slots=True)
class _VimRegister:
    text: str = ""
    linewise: bool = False


@dataclass(frozen=True, slots=True)
class _VimSnapshot:
    text: str
    cursor: tuple[int, int]


class Composer(TextArea):
    """Multiline input where Enter submits and modified Enter adds a line."""

    BINDINGS = [
        Binding("enter", "submit", show=False, priority=True),
        Binding("shift+enter", "insert_newline", show=False, priority=True),
        # Ghostty keybinds often map Shift+Enter to a literal LF. Textual
        # correctly names that byte Ctrl+J, so retain it as an unadvertised
        # compatibility encoding for the user-facing Shift+Enter shortcut.
        Binding("ctrl+j", "insert_newline", show=False, priority=True),
        Binding("tab", "complete", show=False, priority=True),
        Binding("up", "history_previous", show=False, priority=True),
        Binding("down", "history_next", show=False, priority=True),
    ]

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    class ModeChanged(Message):
        def __init__(self, mode: VimMode) -> None:
            super().__init__()
            self.mode = mode

    class VisualHeightChanged(Message):
        pass

    def __init__(
        self,
        history: InputHistory | None = None,
        *,
        vim_enabled: bool = True,
    ) -> None:
        super().__init__(
            soft_wrap=True,
            show_line_numbers=False,
            highlight_cursor_line=False,
            placeholder="Message Ghostwheel",
            id="composer",
        )
        self.prompt_history = history or InputHistory(None)
        self._history_index = len(self.prompt_history.entries)
        self._draft = ""
        self.vim_enabled = vim_enabled
        self.vim_mode = VimMode.INSERT
        self._vim_pending: str | None = None
        self._vim_goal_column: int | None = None
        self._vim_register = _VimRegister()
        self._vim_undo_stack: list[_VimSnapshot] = []
        self._vim_redo_stack: list[_VimSnapshot] = []
        self._vim_insert_origin = self._vim_snapshot() if vim_enabled else None

    def action_submit(self) -> None:
        self._vim_pending = None
        if macos_shift_pressed():
            self.action_insert_newline()
            return
        value = self.text
        if not value.strip():
            return
        self.prompt_history.append(value)
        self._history_index = len(self.prompt_history.entries)
        self._draft = ""
        self.load_text("")
        self._reset_vim_undo()
        self._set_vim_mode(VimMode.INSERT)
        self.post_message(self.Submitted(value))

    def action_insert_newline(self) -> None:
        self._vim_pending = None
        undo_origin = (
            self._vim_snapshot()
            if self.vim_enabled and self.vim_mode is VimMode.NORMAL
            else None
        )
        self.insert("\n", maintain_selection_offset=False)
        if undo_origin is not None:
            self._record_vim_undo(undo_origin)

    async def _on_key(self, event: events.Key) -> None:
        # Textual currently names xterm modifyOtherKeys Shift+Enter this way.
        if event.key == "shift+\r":
            event.stop()
            event.prevent_default()
            self.action_insert_newline()
            return

        if self.vim_enabled and self.vim_mode is VimMode.INSERT:
            if event.key == "escape":
                event.stop()
                event.prevent_default()
                self._enter_vim_normal()
                return
            await super()._on_key(event)
            return

        if self.vim_enabled:
            event.stop()
            event.prevent_default()
            self._clamp_vim_cursor()
            if event.character is not None and event.character.isprintable():
                self._handle_vim_normal_key(event.character)
                return
            self._vim_pending = None
            if event.key == "ctrl+r":
                self._vim_redo()
                return
            if event.key == "escape":
                self._vim_pending = None
                return
            if event.key == "backspace":
                self._vim_move_horizontal(-1)
                return
            if event.key == "delete":
                self._vim_delete_character()
                return
            if event.key == "left":
                self._vim_move_horizontal(-1)
                return
            if event.key == "right":
                self._vim_move_horizontal(1)
                return
            if event.key == "up":
                self._vim_move_vertical(-1)
                return
            if event.key == "down":
                self._vim_move_vertical(1)
                return
            if event.key in {"home", "ctrl+a"}:
                self._vim_move_line(start=True)
                return
            if event.key in {"end", "ctrl+e"}:
                self._vim_move_line(start=False)
                return
            if event.key == "ctrl+left":
                self._vim_move_word("b")
                return
            if event.key == "ctrl+right":
                self._vim_move_word("w")
                return
            return

        await super()._on_key(event)

    async def _on_paste(self, event: events.Paste) -> None:
        if self.vim_enabled and self.vim_mode is VimMode.NORMAL:
            self._vim_pending = None
            event.stop()
            event.prevent_default()
            return
        await super()._on_paste(event)

    async def _on_mouse_down(self, event: events.MouseDown) -> None:
        self._vim_pending = None
        await super()._on_mouse_down(event)
        self._clamp_vim_cursor()

    def watch_virtual_size(self, old_size: Size, new_size: Size) -> None:
        if old_size.height != new_size.height and self.is_mounted:
            self.post_message(self.VisualHeightChanged())

    def action_paste(self) -> None:
        if self.vim_enabled and self.vim_mode is VimMode.NORMAL:
            self._vim_pending = None
            return
        super().action_paste()

    def action_complete(self) -> None:
        if self.vim_enabled and self.vim_mode is VimMode.NORMAL:
            self._vim_pending = None
            return
        value = self.text
        review_completion = _review_path_completion(value)
        if review_completion is not None:
            self._replace_text(review_completion)
            return
        if not value.startswith("/") or any(char.isspace() for char in value):
            return
        matches = [command for command in COMMANDS if command.startswith(value.lower())]
        if len(matches) == 1:
            self._replace_text(matches[0])

    def action_history_previous(self) -> None:
        if self.vim_enabled and self.vim_mode is VimMode.NORMAL:
            self._vim_pending = None
            self._vim_move_vertical(-1)
            return
        row, _column = self.cursor_location
        if row > 0:
            self.action_cursor_up()
            return
        if not self.prompt_history.entries or self._history_index == 0:
            return
        if self._history_index == len(self.prompt_history.entries):
            self._draft = self.text
        self._history_index -= 1
        self._replace_text(self.prompt_history.entries[self._history_index])

    def action_history_next(self) -> None:
        if self.vim_enabled and self.vim_mode is VimMode.NORMAL:
            self._vim_pending = None
            self._vim_move_vertical(1)
            return
        row, _column = self.cursor_location
        last_row = self.text.count("\n")
        if row < last_row:
            self.action_cursor_down()
            return
        if self._history_index >= len(self.prompt_history.entries):
            return
        self._history_index += 1
        value = (
            self._draft
            if self._history_index == len(self.prompt_history.entries)
            else self.prompt_history.entries[self._history_index]
        )
        self._replace_text(value)

    def _replace_text(self, value: str) -> None:
        self.load_text(value)
        lines = value.split("\n")
        self.move_cursor((len(lines) - 1, len(lines[-1])))

    def clear_vim_pending(self) -> None:
        self._vim_pending = None

    def _set_vim_mode(self, mode: VimMode) -> None:
        if not self.vim_enabled or self.vim_mode is mode:
            return
        self.vim_mode = mode
        self._vim_pending = None
        self._vim_goal_column = None
        self.post_message(self.ModeChanged(mode))

    def _enter_vim_normal(self) -> None:
        self.history.checkpoint()
        self._finish_vim_insert()
        row, column = self.cursor_location
        if column > 0:
            self.move_cursor((row, column - 1))
        self._clamp_vim_cursor()
        self._set_vim_mode(VimMode.NORMAL)

    def _enter_vim_insert(
        self,
        location: tuple[int, int] | None = None,
        *,
        undo_origin: _VimSnapshot | None = None,
    ) -> None:
        self.history.checkpoint()
        self._vim_insert_origin = undo_origin or self._vim_snapshot()
        if location is not None:
            self.move_cursor(location)
        self._set_vim_mode(VimMode.INSERT)

    def _handle_vim_normal_key(self, key: str) -> None:
        pending = self._vim_pending
        if pending is not None:
            self._vim_pending = None
            if pending == "g":
                if key == "g":
                    self._vim_move_document(first=True)
                return
            if key in {pending, "h", "l", "w", "b", "e", "0", "$"}:
                self._vim_apply_operator(pending, key)
            return

        if key in {"d", "c", "y", "g"}:
            self._vim_pending = key
            return

        actions = {
            "h": lambda: self._vim_move_horizontal(-1),
            " ": lambda: self._vim_move_horizontal(1),
            "l": lambda: self._vim_move_horizontal(1),
            "j": lambda: self._vim_move_vertical(1),
            "k": lambda: self._vim_move_vertical(-1),
            "w": lambda: self._vim_move_word("w"),
            "b": lambda: self._vim_move_word("b"),
            "e": lambda: self._vim_move_word("e"),
            "0": lambda: self._vim_move_line(start=True),
            "$": lambda: self._vim_move_line(start=False),
            "G": lambda: self._vim_move_document(first=False),
            "i": self._enter_vim_insert,
            "a": self._vim_append,
            "I": self._vim_insert_at_indent,
            "A": self._vim_append_to_line,
            "o": lambda: self._vim_open_line(above=False),
            "O": lambda: self._vim_open_line(above=True),
            "x": self._vim_delete_character,
            "X": self._vim_delete_character_left,
            "D": lambda: self._vim_apply_operator("d", "$"),
            "C": lambda: self._vim_apply_operator("c", "$"),
            "Y": lambda: self._vim_apply_operator("y", "y"),
            "p": lambda: self._vim_paste(after=True),
            "P": lambda: self._vim_paste(after=False),
            "u": self._vim_undo,
        }
        action = actions.get(key)
        if action is not None:
            action()

    def _vim_move_horizontal(self, offset: int) -> None:
        row, column = self.cursor_location
        line = self.document[row]
        if line:
            column = max(0, min(len(line) - 1, column + offset))
        else:
            column = 0
        self.move_cursor((row, column))
        self._vim_goal_column = None

    def _vim_move_vertical(self, offset: int) -> None:
        row, column = self.cursor_location
        if self._vim_goal_column is None:
            self._vim_goal_column = column
        row = max(0, min(self.document.line_count - 1, row + offset))
        line = self.document[row]
        last_column = max(0, len(line) - 1)
        self.move_cursor((row, min(self._vim_goal_column, last_column)))

    def _vim_move_line(self, *, start: bool) -> None:
        row, _column = self.cursor_location
        column = 0 if start else max(0, len(self.document[row]) - 1)
        self.move_cursor((row, column))
        self._vim_goal_column = None

    def _vim_move_document(self, *, first: bool) -> None:
        row = 0 if first else self.document.line_count - 1
        self.move_cursor((row, self._first_nonblank(row)))
        self._vim_goal_column = None

    def _vim_move_word(self, motion: str) -> None:
        index = self.document.get_index_from_location(self.cursor_location)
        destination = self._vim_word_destination(index, motion)
        self.move_cursor(self.document.get_location_from_index(destination))
        self._clamp_vim_cursor()
        self._vim_goal_column = None

    def _vim_word_destination(self, index: int, motion: str) -> int:
        text = self.text
        if not text:
            return 0
        index = max(0, min(index, len(text) - 1))

        if motion == "w":
            character_class = _vim_character_class(text[index])
            while (
                index < len(text)
                and _vim_character_class(text[index]) == character_class
            ):
                index += 1
            while index < len(text) and text[index].isspace():
                index += 1
            return min(index, len(text))

        if motion == "b":
            if index == 0:
                return 0
            index -= 1
            while index > 0 and text[index].isspace():
                index -= 1
            character_class = _vim_character_class(text[index])
            while (
                index > 0 and _vim_character_class(text[index - 1]) == character_class
            ):
                index -= 1
            return index

        if not text[index].isspace() and (
            index + 1 == len(text)
            or _vim_character_class(text[index + 1])
            != _vim_character_class(text[index])
        ):
            index += 1
        if index == len(text):
            return len(text)
        if text[index].isspace():
            while index < len(text) and text[index].isspace():
                index += 1
            if index == len(text):
                return len(text)
        character_class = _vim_character_class(text[index])
        while (
            index + 1 < len(text)
            and _vim_character_class(text[index + 1]) == character_class
        ):
            index += 1
        return index

    def _vim_append(self) -> None:
        row, column = self.cursor_location
        if self.document[row]:
            column += 1
        self._enter_vim_insert((row, column))

    def _vim_insert_at_indent(self) -> None:
        row, _column = self.cursor_location
        line = self.document[row]
        stripped = line.lstrip()
        column = len(line) - len(stripped) if stripped else len(line)
        self._enter_vim_insert((row, column))

    def _vim_append_to_line(self) -> None:
        row, _column = self.cursor_location
        self._enter_vim_insert((row, len(self.document[row])))

    def _vim_open_line(self, *, above: bool) -> None:
        undo_origin = self._vim_snapshot()
        self.history.checkpoint()
        row, _column = self.cursor_location
        if above:
            self.insert("\n", (row, 0), maintain_selection_offset=False)
            location = (row, 0)
        else:
            self.insert(
                "\n",
                (row, len(self.document[row])),
                maintain_selection_offset=False,
            )
            location = (row + 1, 0)
        self.move_cursor(location)
        self._enter_vim_insert(location, undo_origin=undo_origin)

    def _vim_delete_character(self) -> None:
        row, column = self.cursor_location
        line = self.document[row]
        if not line:
            return
        self._vim_edit((row, column), (row, column + 1), linewise=False)

    def _vim_delete_character_left(self) -> None:
        row, column = self.cursor_location
        if column == 0:
            return
        self._vim_edit((row, column - 1), (row, column), linewise=False)
        self.move_cursor((row, column - 1))
        self._clamp_vim_cursor()

    def _vim_apply_operator(self, operator: str, motion: str) -> None:
        if motion == operator:
            self._vim_apply_line_operator(operator)
            return

        start = self.cursor_location
        if motion == "w" and not self.document[start[0]] and operator in {"d", "y"}:
            self._vim_apply_line_operator(operator)
            return
        start_index = self.document.get_index_from_location(start)
        if operator == "c" and motion == "w":
            end_index = self._vim_change_word_end(start_index)
        elif motion in {"w", "b", "e"}:
            end_index = self._vim_word_destination(start_index, motion)
            if motion == "w":
                row, _column = start
                line_end = self.document.get_index_from_location(
                    (row, len(self.document[row]))
                )
                destination_row, _destination_column = (
                    self.document.get_location_from_index(end_index)
                )
                if destination_row != row and start_index < line_end:
                    end_index = line_end
            if motion == "e" and end_index < len(self.text):
                end_index += 1
        elif motion == "h":
            end_index = max(0, start_index - 1)
        elif motion == "l":
            row, column = start
            end_index = start_index + int(column < len(self.document[row]))
        elif motion == "0":
            row, _column = start
            end_index = self.document.get_index_from_location((row, 0))
        else:
            row, _column = start
            end_index = self.document.get_index_from_location(
                (row, len(self.document[row]))
            )

        low_index, high_index = sorted((start_index, end_index))
        undo_origin = self._vim_snapshot()
        low = self.document.get_location_from_index(low_index)
        high = self.document.get_location_from_index(high_index)
        linewise = low[0] != high[0] and low[1] == high[1] == 0
        if low_index == high_index:
            if operator == "c":
                self._enter_vim_insert(low, undo_origin=undo_origin)
            return
        if operator == "y":
            self._store_vim_register(low, high, linewise=linewise)
            self.move_cursor(low)
            self._clamp_vim_cursor()
            return
        if operator == "c" and linewise:
            self._store_vim_register(low, high, linewise=True)
            self.history.checkpoint()
            self.replace("\n", low, high, maintain_selection_offset=False)
            self._enter_vim_insert(low, undo_origin=undo_origin)
            return
        self._vim_edit(
            low,
            high,
            linewise=linewise,
            record_undo=operator != "c",
        )
        if operator == "c":
            self._enter_vim_insert(low, undo_origin=undo_origin)

    def _vim_change_word_end(self, start_index: int) -> int:
        text = self.text
        if not text or start_index >= len(text):
            return start_index
        if text[start_index].isspace():
            row, _column = self.document.get_location_from_index(start_index)
            line_end = self.document.get_index_from_location(
                (row, len(self.document[row]))
            )
            if start_index == line_end:
                return start_index
            return min(self._vim_word_destination(start_index, "w"), line_end)
        character_class = _vim_character_class(text[start_index])
        end_index = start_index + 1
        while (
            end_index < len(text)
            and _vim_character_class(text[end_index]) == character_class
        ):
            end_index += 1
        return end_index

    def _vim_apply_line_operator(self, operator: str) -> None:
        row, _column = self.cursor_location
        line = self.document[row]
        if operator == "y":
            self._vim_register = _VimRegister(line, linewise=True)
            return
        if operator == "c":
            undo_origin = self._vim_snapshot()
            self._vim_register = _VimRegister(line, linewise=True)
            self.history.checkpoint()
            self.delete((row, 0), (row, len(line)), maintain_selection_offset=False)
            self.move_cursor((row, 0))
            self._enter_vim_insert((row, 0), undo_origin=undo_origin)
            return

        undo_origin = self._vim_snapshot()
        self._vim_register = _VimRegister(line, linewise=True)
        self.history.checkpoint()
        last_row = self.document.line_count - 1
        if last_row == 0:
            self.delete((0, 0), (0, len(line)), maintain_selection_offset=False)
            destination_row = 0
        elif row < last_row:
            self.delete((row, 0), (row + 1, 0), maintain_selection_offset=False)
            destination_row = row
        else:
            previous_length = len(self.document[row - 1])
            self.delete(
                (row - 1, previous_length),
                (row, len(line)),
                maintain_selection_offset=False,
            )
            destination_row = row - 1
        self.move_cursor((destination_row, self._first_nonblank(destination_row)))
        self._clamp_vim_cursor()
        self.history.checkpoint()
        self._record_vim_undo(undo_origin)

    def _vim_edit(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        *,
        linewise: bool,
        record_undo: bool = True,
    ) -> None:
        undo_origin = self._vim_snapshot()
        self._store_vim_register(start, end, linewise=linewise)
        self.history.checkpoint()
        self.delete(start, end, maintain_selection_offset=False)
        self._clamp_vim_cursor()
        self.history.checkpoint()
        if record_undo:
            self._record_vim_undo(undo_origin)

    def _store_vim_register(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        *,
        linewise: bool,
    ) -> None:
        text = self.get_text_range(start, end)
        if linewise and text.endswith("\n"):
            text = text[:-1]
        self._vim_register = _VimRegister(
            text,
            linewise=linewise,
        )

    def _vim_paste(self, *, after: bool) -> None:
        register = self._vim_register
        if not register.text and not register.linewise:
            return
        undo_origin = self._vim_snapshot()
        self.history.checkpoint()
        row, column = self.cursor_location
        if register.linewise:
            if after:
                location = (row, len(self.document[row]))
                inserted = "\n" + register.text
                destination = (row + 1, self._first_nonblank_text(register.text))
            else:
                location = (row, 0)
                inserted = register.text + "\n"
                destination = (row, self._first_nonblank_text(register.text))
        else:
            location = (row, column + int(after and bool(self.document[row])))
            inserted = register.text
            result = self.insert(inserted, location, maintain_selection_offset=False)
            self.move_cursor(result.end_location)
            row, column = self.cursor_location
            if column > 0:
                self.move_cursor((row, column - 1))
            self._clamp_vim_cursor()
            self.history.checkpoint()
            self._record_vim_undo(undo_origin)
            return
        self.insert(inserted, location, maintain_selection_offset=False)
        self.move_cursor(destination)
        self._clamp_vim_cursor()
        self.history.checkpoint()
        self._record_vim_undo(undo_origin)

    def _vim_undo(self) -> None:
        if not self._vim_undo_stack:
            return
        self._vim_redo_stack.append(self._vim_snapshot())
        self._restore_vim_snapshot(self._vim_undo_stack.pop())

    def _vim_redo(self) -> None:
        if not self._vim_redo_stack:
            return
        self._vim_undo_stack.append(self._vim_snapshot())
        self._restore_vim_snapshot(self._vim_redo_stack.pop())

    def _vim_snapshot(self) -> _VimSnapshot:
        return _VimSnapshot(self.text, self.cursor_location)

    def _record_vim_undo(self, snapshot: _VimSnapshot) -> None:
        if snapshot.text == self.text:
            return
        self._vim_undo_stack.append(snapshot)
        self._vim_redo_stack.clear()

    def _finish_vim_insert(self) -> None:
        if self._vim_insert_origin is not None:
            self._record_vim_undo(self._vim_insert_origin)
        self._vim_insert_origin = None

    def _reset_vim_undo(self) -> None:
        self._vim_undo_stack.clear()
        self._vim_redo_stack.clear()
        self._vim_insert_origin = self._vim_snapshot() if self.vim_enabled else None

    def _restore_vim_snapshot(self, snapshot: _VimSnapshot) -> None:
        self.load_text(snapshot.text)
        self.move_cursor(snapshot.cursor)
        self._clamp_vim_cursor()

    def _clamp_vim_cursor(self) -> None:
        if not self.vim_enabled or self.vim_mode is not VimMode.NORMAL:
            return
        row, column = self.cursor_location
        row = max(0, min(row, self.document.line_count - 1))
        line = self.document[row]
        self.move_cursor((row, min(column, max(0, len(line) - 1))))

    def _first_nonblank(self, row: int) -> int:
        return self._first_nonblank_text(self.document[row])

    @staticmethod
    def _first_nonblank_text(line: str) -> int:
        stripped = line.lstrip()
        return len(line) - len(stripped) if stripped else max(0, len(line) - 1)


def _vim_character_class(character: str) -> int:
    if character.isspace():
        return 0
    if character.isalnum() or character == "_":
        return 1
    return 2


def _format_token_count(value: int) -> str:
    if value < 1_000:
        return str(value)
    thousands = value / 1_000
    if thousands < 10 and not thousands.is_integer():
        compact = f"{thousands:.1f}".rstrip("0").rstrip(".")
        return f"{compact}k"
    return f"{thousands:.0f}k"


@dataclass(slots=True)
class _ToolState:
    name: str
    arguments: str
    call_id: str | None
    started_at: float
    status: str = "running"
    detail: str = ""
    finished_at: float | None = None


class TurnView(Vertical):
    """One assistant turn with retained, dynamically hideable details."""

    def __init__(self, label: str, *, details_expanded: bool = False) -> None:
        self.status = Static(Text(label, style="dim"), classes="turn-status")
        self.tool_summaries = Static(classes="tool-summaries")
        self.tool_details = Static(classes="turn-details tool-details")
        self.thinking_detail = Static(classes="turn-details thinking-detail")
        self.heading = Static(
            Text("Ghostwheel", style="bold magenta"),
            classes="assistant-heading",
        )
        self.answer = Static(classes="assistant-answer")
        self.outcome = Static(classes="turn-outcome")
        super().__init__(
            self.status,
            self.tool_summaries,
            self.tool_details,
            self.thinking_detail,
            self.heading,
            self.answer,
            self.outcome,
            classes="turn",
        )
        self.details_expanded = details_expanded
        self._answer = ""
        self._thinking = ""
        self._tools: list[_ToolState] = []
        self.tool_summaries.display = False
        self.tool_details.display = False
        self.thinking_detail.display = False
        self.heading.display = False
        self.answer.display = False
        self.outcome.display = False

    async def apply_event(self, event: AgentEvent) -> None:
        if isinstance(event, ThinkingOutput):
            self._thinking += event.content
            self.status.update(Text("Thinking…", style="dim"))
            self._render_thinking()
        elif isinstance(event, TextOutput):
            self._answer += event.content
            self.status.update(Text("Responding…", style="dim"))
            self.heading.display = True
            self.answer.display = True
            self.answer.update(RichMarkdown(self._answer))
        elif isinstance(event, ToolStarted):
            self._tools.append(
                _ToolState(
                    name=event.name,
                    arguments=event.arguments,
                    call_id=event.call_id,
                    started_at=time.monotonic(),
                )
            )
            self.status.update(Text(f"Running {event.name}…", style="dim"))
            self._render_tools()
        elif isinstance(event, ToolFinished):
            self._finish_tool(event.name, event.call_id, "succeeded", event.result)
            self.status.update(Text("Thinking…", style="dim"))
            self._render_tools()
        elif isinstance(event, ToolFailed):
            self._finish_tool(event.name, event.call_id, "failed", event.error)
            self.status.update(Text(f"{event.name} failed", style="red"))
            self._render_tools()

    def set_details_expanded(self, expanded: bool) -> None:
        self.details_expanded = expanded
        self._render_thinking()
        self._render_tools()

    def finish_turn(self, outcome: TurnOutcome) -> None:
        self.status.display = False
        if isinstance(outcome, TurnSucceeded):
            self._answer = outcome.output
            self.heading.display = True
            self.answer.display = True
            self.answer.update(RichMarkdown(outcome.output))
        elif isinstance(outcome, TurnNoResult):
            self._show_outcome(Text(outcome.message, style="yellow"))
        elif isinstance(outcome, TurnFailed):
            self._show_outcome(_turn_failure(outcome))

    def finish_review(self, outcome: ReviewOutcome, *, width: int) -> None:
        self.status.display = False
        if isinstance(outcome, StructuredReview):
            renderables: list[RenderableType] = []
            if outcome.used_fallback:
                renderables.append(
                    Text(
                        "Structured-output fallback was used for this review.",
                        style="dim",
                    )
                )
            renderables.extend(review_renderables(outcome.review, width=width))
            self._show_outcome(Group(*renderables))
        elif isinstance(outcome, RawReview):
            body = Text()
            body.append("Couldn't produce a structured review.\n", style="yellow")
            body.append("Reason: ", style="dim")
            body.append(outcome.structured_failure, style="dim")
            body.append("\n\nShowing the raw review instead:\n\n", style="bold")
            body.append(outcome.prose)
            self._show_outcome(
                Panel(body, title="Structured Review Failed", border_style="yellow")
            )
        elif isinstance(outcome, ReviewFailed):
            body = Text(outcome.message)
            body.append(
                "\n\nCheck the review model configuration, then use /retry.",
                style="dim",
            )
            self._show_outcome(Panel(body, title="Review Failed", border_style="red"))

    def cancel(self) -> None:
        self.status.display = False
        self._show_outcome(Text("Turn cancelled.", style="yellow"))

    def _show_outcome(self, renderable: RenderableType) -> None:
        self.outcome.update(renderable)
        self.outcome.display = True

    def _finish_tool(
        self,
        name: str,
        call_id: str | None,
        status: str,
        detail: str,
    ) -> None:
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
            activity = _ToolState(name, "", call_id, time.monotonic())
            self._tools.append(activity)
        activity.status = status
        activity.detail = detail
        activity.finished_at = time.monotonic()

    def _render_thinking(self) -> None:
        self.thinking_detail.update(
            Panel(
                Text(self._thinking, style="dim"),
                title=Text("Thinking", style="dim"),
                border_style="dim",
                padding=(0, 1),
            )
        )
        self.thinking_detail.display = bool(self._thinking) and self.details_expanded

    def _render_tools(self) -> None:
        summaries = [_tool_summary(tool) for tool in self._tools]
        details = [_tool_detail(tool) for tool in self._tools]
        self.tool_summaries.update(Group(*summaries))
        self.tool_details.update(Group(*details))
        self.tool_summaries.display = bool(summaries)
        self.tool_details.display = bool(details) and self.details_expanded


class TextualPresenter:
    """Presenter adapter used by the existing command loop inside Textual."""

    def __init__(self, app: GhostwheelApp, app_info: AppInfo) -> None:
        self.app = app
        self.app_info = app_info
        self.current_turn: TurnView | None = None

    async def handle_event(self, event: AgentEvent) -> None:
        turn = self.current_turn
        if turn is None:
            turn = self._start_turn("Thinking…")
        follow = self.app.transcript.is_vertical_scroll_end
        await turn.apply_event(event)
        self.app.follow_output(follow)

    def welcome(self) -> None:
        # The persistent header already carries the application identity.
        return

    def goodbye(self) -> None:
        self.app.exit()

    def help(self) -> None:
        self.app.add_renderable(
            _help_panel(vim_mode=self.app.vim_mode),
            classes="system-message",
        )

    def model_info(self) -> None:
        self.app.add_renderable(
            Text.assemble(
                Text("Model  ", style="bold"),
                Text(
                    f"{self.app_info.provider}/{self.app_info.model}",
                    style="cyan",
                ),
            ),
            classes="system-message",
        )

    def tools_info(self) -> None:
        body = Text.assemble(
            Text("Tool profile  ", style="bold"),
            Text(self.app_info.tool_profile, style="yellow"),
        )
        if self.app_info.tool_profile == "full":
            body.append("\nShell commands run with unrestricted environment access.")
        self.app.add_renderable(
            Panel(body, title="Tools", border_style="yellow"),
            classes="system-message",
        )

    def unknown_command(self, command: str, suggestion: str | None = None) -> None:
        message = Text.assemble(
            Text("Unknown command: ", style="yellow"),
            Text(command),
        )
        if suggestion:
            message.append(f"\nDid you mean {suggestion}?", style="dim")
        message.append("\nType /help to list commands.", style="dim")
        self.app.add_renderable(message, classes="system-message")

    def retry_unavailable(self) -> None:
        self.app.add_renderable(
            Text("Nothing to retry yet.", style="yellow"),
            classes="system-message",
        )

    def history_cleared(self) -> None:
        self.app.update_context()
        self.app.add_renderable(
            Text("Conversation history cleared.", style="dim"),
            classes="system-message",
        )

    def history_compacted(self, before_tokens: int, after_tokens: int) -> None:
        self.app.add_renderable(
            Text(
                "Context compacted: "
                f"{_format_token_count(before_tokens)} → "
                f"~{_format_token_count(after_tokens)}.",
                style="dim",
            ),
            classes="system-message",
        )

    def turn_started(self, label: str = "Thinking…") -> None:
        self._start_turn(label)

    def turn_cancelled(self) -> None:
        if self.current_turn is not None:
            self.current_turn.cancel()
        self.current_turn = None
        self.app.follow_output(True)

    def turn_outcome(self, outcome: TurnOutcome) -> None:
        follow = self.app.transcript.is_vertical_scroll_end
        if self.current_turn is None:
            self._start_turn("Thinking…")
        assert self.current_turn is not None
        self.current_turn.finish_turn(outcome)
        self.current_turn = None
        self.app.update_context()
        self.app.follow_output(follow)

    def review_outcome(self, outcome: ReviewOutcome) -> None:
        follow = self.app.transcript.is_vertical_scroll_end
        if self.current_turn is None:
            self._start_turn("Reviewing…")
        assert self.current_turn is not None
        self.current_turn.finish_review(outcome, width=max(40, self.app.size.width - 4))
        self.current_turn = None
        self.app.follow_output(follow)

    def _start_turn(self, label: str) -> TurnView:
        turn = TurnView(label, details_expanded=self.app.details_expanded)
        self.current_turn = turn
        self.app.add_widget(turn)
        return turn


class GhostwheelApp(App[None]):
    """Full-screen chat UI with a redrawable transcript and composer."""

    TITLE = "Ghostwheel"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("ctrl+o", "toggle_details", show=False, priority=True),
        Binding("ctrl+c", "cancel_or_quit", show=False, priority=True),
        Binding("ctrl+q", "quit", show=False, priority=True),
    ]
    CSS = """
    Screen {
        background: $background;
        color: $text;
    }

    #app-header {
        height: 3;
        padding: 0 1;
        background: $background;
        color: $text;
    }

    #transcript {
        height: 1fr;
        padding: 0 1 1 1;
        background: $background;
        scrollbar-size-vertical: 1;
        scrollbar-color: $primary-darken-2;
        scrollbar-background: $background;
    }

    .user-message {
        height: auto;
        margin-top: 1;
    }

    .system-message {
        height: auto;
        margin: 1 0 0 2;
    }

    TurnView {
        height: auto;
        margin-top: 1;
    }

    TurnView > Static {
        height: auto;
    }

    .turn-status {
        color: $text-muted;
    }

    .tool-summaries, .turn-details, .turn-outcome, .assistant-answer {
        margin-left: 2;
    }

    .assistant-heading {
        margin-top: 1;
    }

    #composer-shell {
        height: 3;
        padding: 0 1;
        background: $surface;
        border-top: solid $primary-darken-3;
    }

    #composer-prompt {
        width: 6;
        height: 1fr;
        color: $accent;
        text-style: bold;
        background: $surface;
    }

    #composer-prompt.vim-mode {
        width: 7;
    }

    #composer {
        width: 1fr;
        height: 1fr;
        border: none;
        padding: 0;
        background: $surface;
        color: $text;
    }

    #context {
        width: 16;
        height: 1fr;
        content-align: right top;
        color: $text-muted;
        background: $surface;
    }
    """

    def __init__(
        self,
        console: Console,
        session: ChatSession,
        reviews: ReviewService,
        *,
        app_info: AppInfo,
        history_path: Path | None = None,
        cancellation: TurnCancellation | None = None,
        vim_mode: bool = True,
    ) -> None:
        super().__init__(driver_class=GhostwheelTerminalDriver, ansi_color=True)
        self._console = console
        self.session = session
        self.reviews = reviews
        self.app_info = app_info
        self.cancellation = cancellation or TurnCancellation()
        self.history = InputHistory(history_path)
        self.input_reader = QueueInputReader()
        self.details_expanded = False
        self.vim_mode = vim_mode
        self._composer_height = COMPOSER_MIN_HEIGHT
        self.header = Static(_header(app_info), id="app-header")
        self.transcript = VerticalScroll(id="transcript")
        self.composer = Composer(self.history, vim_enabled=vim_mode)
        self.composer_prompt = Static(
            self._composer_prompt(),
            id="composer-prompt",
            classes="vim-mode" if vim_mode else "",
        )
        self.context = Static(id="context")
        self.composer_shell = Horizontal(
            self.composer_prompt,
            self.composer,
            self.context,
            id="composer-shell",
        )
        self.presenter = TextualPresenter(self, app_info)

    def compose(self) -> ComposeResult:
        yield self.header
        yield self.transcript
        yield self.composer_shell

    def on_mount(self) -> None:
        from ghostwheel.cli import run_cli

        self.update_context()
        self.composer.focus()
        self.call_after_refresh(self._resize_composer)
        self.run_worker(
            run_cli(
                self._console,
                self.session,
                self.reviews,
                presenter=self.presenter,  # type: ignore[arg-type]
                input_reader=self.input_reader,
                cancellation=self.cancellation,
            ),
            name="command-loop",
            exit_on_error=True,
        )

    async def on_composer_submitted(self, message: Composer.Submitted) -> None:
        value = message.value
        user = Text("You › ", style="bold cyan")
        user.append(value)
        await self.transcript.mount(Static(user, classes="user-message"))
        self.follow_output(True)
        self.input_reader.submit(value)

    def on_composer_mode_changed(self, _message: Composer.ModeChanged) -> None:
        self.composer_prompt.update(self._composer_prompt())

    def on_composer_visual_height_changed(
        self,
        _message: Composer.VisualHeightChanged,
    ) -> None:
        self._resize_composer()

    def on_text_area_changed(self, message: TextArea.Changed) -> None:
        if message.text_area is self.composer:
            self._resize_composer()

    def on_resize(self, _event: events.Resize) -> None:
        self.call_after_refresh(self._resize_composer)

    def action_toggle_details(self) -> None:
        self.composer.clear_vim_pending()
        follow = self.transcript.is_vertical_scroll_end
        self.details_expanded = not self.details_expanded
        turns = list(self.query(TurnView))
        current_turn = self.presenter.current_turn
        if current_turn is not None and current_turn not in turns:
            turns.append(current_turn)
        for turn in turns:
            turn.set_details_expanded(self.details_expanded)
        self.follow_output(follow)

    def action_cancel_or_quit(self) -> None:
        self.composer.clear_vim_pending()
        if not self.cancellation.cancel():
            self.input_reader.submit("/quit")

    def add_widget(self, widget: Static | TurnView) -> None:
        follow = self.transcript.is_vertical_scroll_end
        self.transcript.mount(widget)
        self.follow_output(follow)

    def add_renderable(self, renderable: RenderableType, *, classes: str) -> None:
        self.add_widget(Static(renderable, classes=classes))

    def follow_output(self, should_follow: bool) -> None:
        if should_follow:
            self.call_after_refresh(
                self.transcript.scroll_end,
                animate=False,
                immediate=True,
            )

    def update_context(self) -> None:
        estimated_tokens = getattr(self.session, "estimated_context_tokens", 0)
        context_window = getattr(self.session, "context_window_tokens", 0)
        is_estimate = getattr(self.session, "context_tokens_estimated", True)
        compaction_enabled = getattr(self.session, "compaction_enabled", True)
        estimate_marker = "~" if is_estimate else ""
        compaction_marker = "" if compaction_enabled else " · off"
        label = (
            f"ctx {estimate_marker}{_format_token_count(estimated_tokens)}/"
            f"{_format_token_count(context_window)}{compaction_marker}"
            if context_window
            else ""
        )
        self.context.update(Text(label, style="dim"))

    def _resize_composer(self) -> None:
        if not self.composer_shell.is_mounted:
            return
        visual_lines = max(1, self.composer.wrapped_document.height)
        header_height = self.header.region.height or 3
        maximum_height = max(
            COMPOSER_MIN_HEIGHT,
            self.size.height - header_height - TRANSCRIPT_MIN_HEIGHT,
        )
        target_height = min(
            maximum_height,
            max(COMPOSER_MIN_HEIGHT, visual_lines + COMPOSER_BORDER_HEIGHT),
        )
        if target_height != self._composer_height:
            self._composer_height = target_height
            self.composer_shell.styles.height = target_height
            self.call_after_refresh(self._resize_composer)

    def _composer_prompt(self) -> str:
        if not self.vim_mode:
            return "You ›"
        mode = "I" if self.composer.vim_mode is VimMode.INSERT else "N"
        return f"You {mode}›"


def _header(app_info: AppInfo) -> Group:
    title = Text("Ghostwheel", style="bold magenta")
    details = Text()
    details.append(f"{app_info.provider}/{app_info.model}", style="cyan")
    details.append("  ·  ")
    details.append(app_info.workspace)
    details.append("  ·  tools: ")
    profile_style = "bold yellow" if app_info.tool_profile == "full" else "green"
    details.append(app_info.tool_profile.upper(), style=profile_style)
    return Group(title, details)


def _help_panel(*, vim_mode: bool = False) -> Panel:
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
    body.append("Ctrl+O              toggle thinking and tool details\n", style="dim")
    body.append("↑/↓                 recall earlier prompts\n", style="dim")
    body.append("Tab                 complete commands and review paths", style="dim")
    if vim_mode:
        body.append("\n\nVim prompt editing\n", style="bold")
        body.append("Esc / i a I A       switch Normal / Insert mode\n", style="dim")
        body.append("h j k l · w b e · 0 $   move the cursor\n", style="dim")
        body.append("x X · d c y + motion    edit or copy text\n", style="dim")
        body.append("o O · p P · u Ctrl+R    open, paste, undo/redo", style="dim")
    return Panel(body, title="Commands", border_style="cyan")


def _review_path_completion(value: str) -> str | None:
    prefix = "/review "
    if not value.lower().startswith(prefix):
        return None
    path_text = value[len(prefix) :]
    candidate_path = Path(path_text or ".").expanduser()
    directory = candidate_path.parent if path_text else candidate_path
    name_prefix = candidate_path.name if path_text else ""
    try:
        matches = sorted(
            child for child in directory.iterdir() if child.name.startswith(name_prefix)
        )
    except OSError:
        return None
    if not matches:
        return None
    common_name = os.path.commonprefix([match.name for match in matches])
    if len(matches) > 1 and common_name == name_prefix:
        return None
    parent_text = str(Path(path_text).parent) if path_text else ""
    completed_path = (
        str(Path(parent_text) / common_name) if parent_text else common_name
    )
    if len(matches) == 1 and matches[0].is_dir():
        completed_path += os.sep
    return prefix + completed_path


def _tool_summary(activity: _ToolState) -> Text:
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
        line.append(_duration(activity.finished_at - activity.started_at), style="dim")
    if activity.status == "failed" and activity.detail:
        line.append("  ·  ", style="red")
        line.append(
            _preview(" ".join(activity.detail.split()), 100),
            style="red",
        )
    return line


def _tool_detail(activity: _ToolState) -> Panel:
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


def _turn_failure(outcome: TurnFailed) -> Panel:
    title, hint = {
        FailureKind.PROVIDER: (
            "Provider Error",
            "Check that the configured model server is running; use /model to inspect it.",
        ),
        FailureKind.CONFIGURATION: (
            "Configuration Error",
            "Check the model, context-window, and compaction settings.",
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
    return Panel(body, title=title, border_style="red")
