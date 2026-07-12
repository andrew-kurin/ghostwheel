"""Textual prompt composer, completion, and optional Vim editing."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from textual import events
from textual.binding import Binding
from textual.geometry import Size
from textual.message import Message
from textual.widgets import TextArea

from ghostwheel.input_ui import COMMANDS, InputHistory
from ghostwheel.keyboard import macos_shift_pressed

__all__ = ["Composer", "VimMode"]


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
        # Textual dispatches default handlers for every class in the MRO. Since
        # the base handler was invoked explicitly above, prevent it from being
        # dispatched a second time after this override returns.
        event.stop()
        event.prevent_default()

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
