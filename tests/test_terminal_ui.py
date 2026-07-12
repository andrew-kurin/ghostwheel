from __future__ import annotations

import asyncio
import os
import select
import signal
import stat
import subprocess
import sys
import termios
import textwrap
import time
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.data_structures import Size
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.output.vt100 import Vt100_Output
from prompt_toolkit.key_binding.vi_state import InputMode
from rich.console import Console

from ghostwheel.app_info import AppInfo, ToolInfo
from ghostwheel.events import TextOutput, ToolFailed, ToolFinished, ToolStarted
from ghostwheel.runtime_contracts import TurnSucceeded
from ghostwheel.terminal_ui import TerminalUI, default_history_path


class FakeSession:
    history: tuple[object, ...] = ()
    last_compaction = None
    estimated_context_tokens = 1_250
    context_window_tokens = 16_384
    context_tokens_estimated = True
    compaction_enabled = True

    async def send(self, _prompt: str) -> TurnSucceeded[str]:
        return TurnSucceeded("done", ())

    def clear(self) -> None:
        self.history = ()


class RecordingOutput(DummyOutput):
    def __init__(self) -> None:
        self.alternate_screen_entries = 0
        self.mouse_support_entries = 0

    def enter_alternate_screen(self) -> None:
        self.alternate_screen_entries += 1

    def enable_mouse_support(self) -> None:
        self.mouse_support_entries += 1


def make_ui(
    *,
    workspace: Path,
    app_info: AppInfo | None = None,
    output: StringIO | None = None,
    force_terminal: bool = False,
    width: int = 80,
    **kwargs: object,
) -> TerminalUI:
    return TerminalUI(
        Console(
            file=output or StringIO(),
            color_system=None,
            force_terminal=force_terminal,
            width=width,
        ),
        session=FakeSession(),
        app_info=app_info or AppInfo(str(workspace), "provider", "model", "read-only"),
        **kwargs,
    )


async def wait_for_prompt(ui: TerminalUI, task: asyncio.Task[str]) -> None:
    for _attempt in range(100):
        if ui._get_prompt_session().app.is_running:
            return
        if task.done():
            await task
            raise AssertionError("prompt stopped before accepting input")
        await asyncio.sleep(0.01)
    raise AssertionError("prompt did not start")


async def feed_prompt(ui: TerminalUI, pipe_input: object, value: str) -> str:
    task = asyncio.create_task(ui.read())
    await wait_for_prompt(ui, task)
    pipe_input.send_text(value)  # type: ignore[attr-defined]
    return await asyncio.wait_for(task, 1)


async def wait_for_buffer_text(ui: TerminalUI, expected: str) -> None:
    for _attempt in range(100):
        if ui._get_prompt_session().default_buffer.text == expected:
            return
        await asyncio.sleep(0.01)
    actual = ui._get_prompt_session().default_buffer.text
    raise AssertionError(f"buffer did not become {expected!r}; got {actual!r}")


async def wait_for_cursor_row(ui: TerminalUI, expected: int) -> None:
    for _attempt in range(100):
        row = ui._get_prompt_session().default_buffer.document.cursor_position_row
        if row == expected:
            return
        await asyncio.sleep(0.01)
    actual = ui._get_prompt_session().default_buffer.document.cursor_position_row
    raise AssertionError(f"cursor row did not become {expected}; got {actual}")


async def wait_for_input_mode(ui: TerminalUI, expected: InputMode) -> None:
    for _attempt in range(100):
        if ui._get_prompt_session().app.vi_state.input_mode is expected:
            return
        await asyncio.sleep(0.01)
    actual = ui._get_prompt_session().app.vi_state.input_mode
    raise AssertionError(f"input mode did not become {expected}; got {actual}")


def read_until(descriptor: int, marker: bytes, timeout: float = 2) -> bytes:
    deadline = time.monotonic() + timeout
    output = bytearray()
    while marker not in output:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([descriptor], [], [], remaining)[0]:
            raise AssertionError(f"child output did not contain {marker!r}: {output!r}")
        output.extend(os.read(descriptor, 4096))
    return bytes(output)


def test_prompt_is_inline_mouse_free_and_distinguishes_submit_from_newline(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        output = RecordingOutput()
        with create_pipe_input() as pipe_input:
            ui = make_ui(
                workspace=tmp_path,
                interactive=True,
                prompt_input=pipe_input,
                prompt_output=output,
                live=False,
            )
            try:
                assert await feed_prompt(ui, pipe_input, "first\nsecond\r") == (
                    "first\nsecond"
                )
                session = ui._get_prompt_session()
                assert session.app.full_screen is False
                assert session.app.erase_when_done is False
                assert session.app.mouse_support() is False
                assert session.enable_suspend is True
                assert output.alternate_screen_entries == 0
                assert output.mouse_support_entries == 0

                with pytest.raises(EOFError):
                    await feed_prompt(ui, pipe_input, "\x11")
            finally:
                ui.close()

    asyncio.run(scenario())


def test_bracketed_paste_is_inserted_exactly_once(tmp_path: Path) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe_input:
            ui = make_ui(
                workspace=tmp_path,
                interactive=True,
                prompt_input=pipe_input,
                prompt_output=DummyOutput(),
                live=False,
            )
            try:
                pasted = "inserted\nonce"
                encoded = f"\x1b[200~{pasted}\x1b[201~\r"
                assert await feed_prompt(ui, pipe_input, encoded) == pasted
            finally:
                ui.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("vim_mode", [True, False])
def test_up_and_down_cycle_all_history_then_restore_the_draft(
    tmp_path: Path,
    vim_mode: bool,
) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe_input:
            ui = make_ui(
                workspace=tmp_path,
                interactive=True,
                prompt_input=pipe_input,
                prompt_output=DummyOutput(),
                vim_mode=vim_mode,
                live=False,
            )
            try:
                assert await feed_prompt(ui, pipe_input, "first prompt\r") == (
                    "first prompt"
                )
                assert await feed_prompt(ui, pipe_input, "second prompt\r") == (
                    "second prompt"
                )

                prompt_task = asyncio.create_task(ui.read())
                await wait_for_prompt(ui, prompt_task)
                pipe_input.send_text("unmatched draft")
                await wait_for_buffer_text(ui, "unmatched draft")

                pipe_input.send_text("\x1b[A")
                await wait_for_buffer_text(ui, "second prompt")
                pipe_input.send_text("\x1b[A")
                await wait_for_buffer_text(ui, "first prompt")
                pipe_input.send_text("\x1b[B")
                await wait_for_buffer_text(ui, "second prompt")
                pipe_input.send_text("\x1b[B")
                await wait_for_buffer_text(ui, "unmatched draft")

                pipe_input.send_text("\r")
                assert await asyncio.wait_for(prompt_task, 1) == "unmatched draft"
            finally:
                ui.close()

    asyncio.run(scenario())


def test_history_arrows_move_within_multiline_input_before_recall(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe_input:
            ui = make_ui(
                workspace=tmp_path,
                interactive=True,
                prompt_input=pipe_input,
                prompt_output=DummyOutput(),
                live=False,
            )
            try:
                assert await feed_prompt(ui, pipe_input, "historic prompt\r") == (
                    "historic prompt"
                )

                prompt_task = asyncio.create_task(ui.read())
                await wait_for_prompt(ui, prompt_task)
                pipe_input.send_text("top\nbottom")
                await wait_for_buffer_text(ui, "top\nbottom")
                await wait_for_cursor_row(ui, 1)

                pipe_input.send_text("\x1b[A")
                await wait_for_cursor_row(ui, 0)
                assert ui._get_prompt_session().default_buffer.text == "top\nbottom"

                pipe_input.send_text("\x1b[A")
                await wait_for_buffer_text(ui, "historic prompt")
                pipe_input.send_text("\x1b[B")
                await wait_for_buffer_text(ui, "top\nbottom")
                await wait_for_cursor_row(ui, 1)

                pipe_input.send_text("\x1b[A")
                await wait_for_cursor_row(ui, 0)
                pipe_input.send_text("\x1b[B")
                await wait_for_cursor_row(ui, 1)
                assert ui._get_prompt_session().default_buffer.text == "top\nbottom"

                pipe_input.send_text("\r")
                assert await asyncio.wait_for(prompt_task, 1) == "top\nbottom"
            finally:
                ui.close()

    asyncio.run(scenario())


def test_history_arrows_only_move_vertically_in_vim_navigation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe_input:
            ui = make_ui(
                workspace=tmp_path,
                interactive=True,
                prompt_input=pipe_input,
                prompt_output=DummyOutput(),
                vim_mode=True,
                live=False,
            )
            try:
                assert await feed_prompt(ui, pipe_input, "historic prompt\r") == (
                    "historic prompt"
                )

                prompt_task = asyncio.create_task(ui.read())
                await wait_for_prompt(ui, prompt_task)
                pipe_input.send_text("top\nbottom\x1b")
                await wait_for_input_mode(ui, InputMode.NAVIGATION)
                await wait_for_cursor_row(ui, 1)

                pipe_input.send_text("\x1b[A")
                await wait_for_cursor_row(ui, 0)
                pipe_input.send_text("\x1b[A")
                await asyncio.sleep(0.01)
                assert ui._get_prompt_session().default_buffer.text == "top\nbottom"

                pipe_input.send_text("\x1b[B")
                await wait_for_cursor_row(ui, 1)
                pipe_input.send_text("\x1b[B")
                await asyncio.sleep(0.01)
                assert ui._get_prompt_session().default_buffer.text == "top\nbottom"

                pipe_input.send_text("\r")
                assert await asyncio.wait_for(prompt_task, 1) == "top\nbottom"
            finally:
                ui.close()

    asyncio.run(scenario())


def test_prompt_preserves_unicode_submission(tmp_path: Path) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe_input:
            ui = make_ui(
                workspace=tmp_path,
                interactive=True,
                prompt_input=pipe_input,
                prompt_output=DummyOutput(),
                live=False,
            )
            try:
                value = "漢字 · é · 👩‍💻 · 🇺🇸 · 👍🏽"
                assert await feed_prompt(ui, pipe_input, value + "\r") == value
            finally:
                ui.close()

    asyncio.run(scenario())


def test_prompt_emits_no_alternate_screen_or_mouse_protocols(tmp_path: Path) -> None:
    async def scenario() -> str:
        terminal_bytes = StringIO()
        prompt_output = Vt100_Output(
            terminal_bytes,
            lambda: Size(rows=15, columns=60),
            term="xterm-256color",
            enable_cpr=False,
        )
        with create_pipe_input() as pipe_input:
            ui = make_ui(
                workspace=tmp_path,
                interactive=True,
                prompt_input=pipe_input,
                prompt_output=prompt_output,
                live=False,
            )
            try:
                assert await feed_prompt(ui, pipe_input, "hello\r") == "hello"
            finally:
                ui.close()
        return terminal_bytes.getvalue()

    output = asyncio.run(scenario())

    assert "\x1b[?2004h" in output
    assert "\x1b[?2004l" in output
    assert "~1.2k/16k" in output
    assert "\x1b[3J" not in output
    for mode in (9, 47, 1047, 1049, 1000, 1002, 1003, 1006, 1015, 1016):
        assert f"\x1b[?{mode}h" not in output


def test_redirected_input_uses_the_same_ui_without_terminal_control(
    tmp_path: Path,
) -> None:
    ui = make_ui(
        workspace=tmp_path,
        interactive=False,
        input_stream=StringIO("first\nsecond\n"),
        live=True,
    )

    async def scenario() -> None:
        assert await ui.read() == "first"
        assert await ui.read() == "second"
        with pytest.raises(EOFError):
            await ui.read()

    asyncio.run(scenario())
    assert ui.live_enabled is False


def test_redirected_pipe_input_is_async_and_preserves_buffered_lines(
    tmp_path: Path,
) -> None:
    read_descriptor, write_descriptor = os.pipe()
    input_stream = os.fdopen(read_descriptor, "r", encoding="utf-8")
    ui = make_ui(
        workspace=tmp_path,
        interactive=False,
        input_stream=input_stream,
        live=False,
    )

    async def scenario() -> None:
        first_line = asyncio.create_task(ui.read())
        await asyncio.sleep(0.01)
        os.write(write_descriptor, b"first")
        await asyncio.sleep(0.01)
        assert not first_line.done()
        os.write(write_descriptor, b" line\nsecond line\n")
        assert await asyncio.wait_for(first_line, 1) == "first line"
        assert await ui.read() == "second line"
        os.close(write_descriptor)
        with pytest.raises(EOFError):
            await ui.read()

    try:
        asyncio.run(scenario())
    finally:
        ui.close()
        input_stream.close()
        try:
            os.close(write_descriptor)
        except OSError:
            pass


def test_private_history_round_trips_multiline_entries(tmp_path: Path) -> None:
    history_path = tmp_path / "state" / "ghostwheel" / "history"
    ui = make_ui(
        workspace=tmp_path,
        history_path=history_path,
        interactive=False,
        live=False,
    )

    ui._history.append_string("first")
    ui._history.append_string("more\nthan one line")

    assert stat.S_IMODE(history_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(history_path.stat().st_mode) == 0o600
    restored = make_ui(
        workspace=tmp_path,
        history_path=history_path,
        interactive=False,
        live=False,
    )
    assert restored._history_store.entries == ["first", "more\nthan one line"]
    assert "+more\n+than one line" in history_path.read_text(encoding="utf-8")


def test_in_memory_history_and_xdg_default_create_no_unrequested_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert default_history_path() == tmp_path / "ghostwheel" / "input-history"

    ui = make_ui(
        workspace=tmp_path,
        history_path=None,
        interactive=False,
        live=False,
    )
    ui._history.append_string("remember for this run")
    assert ui._history_store.entries == ["remember for this run"]
    assert not (tmp_path / "ghostwheel").exists()


def test_completion_covers_commands_files_and_directories(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("read me", encoding="utf-8")
    (tmp_path / "src").mkdir()
    ui = make_ui(workspace=tmp_path, interactive=False, live=False)
    event = CompleteEvent(completion_requested=True)

    command_completions = list(ui._completer.get_completions(Document("/ret"), event))
    file_completions = list(
        ui._completer.get_completions(Document("/review REA"), event)
    )
    directory_completions = list(
        ui._completer.get_completions(Document("/review s"), event)
    )

    assert [completion.text for completion in command_completions] == ["/retry"]
    assert [completion.text for completion in file_completions] == ["DME.md"]
    assert [completion.text for completion in directory_completions] == ["rc/"]
    assert (
        list(
            ui._completer.get_completions(
                Document("/revX", cursor_position=4),
                event,
            )
        )
        == []
    )


def test_custom_edit_bindings_are_inert_in_vim_navigation(tmp_path: Path) -> None:
    async def wait_for_navigation(ui: TerminalUI) -> None:
        for _attempt in range(100):
            if ui._get_prompt_session().app.vi_state.input_mode is InputMode.NAVIGATION:
                return
            await asyncio.sleep(0.01)
        raise AssertionError("prompt did not enter Vi navigation mode")

    async def scenario() -> None:
        with create_pipe_input() as pipe_input:
            ui = make_ui(
                workspace=tmp_path,
                interactive=True,
                prompt_input=pipe_input,
                prompt_output=DummyOutput(),
                vim_mode=True,
                live=False,
            )
            try:
                newline_task = asyncio.create_task(ui.read())
                await wait_for_prompt(ui, newline_task)
                pipe_input.send_text("abc\x1b")
                await wait_for_navigation(ui)
                assert ui._status_text().endswith(" · N")
                pipe_input.send_text("\n")
                await asyncio.sleep(0.01)
                assert ui._get_prompt_session().default_buffer.text == "abc"
                assert not newline_task.done()
                pipe_input.send_text("\r")
                assert await asyncio.wait_for(newline_task, 1) == "abc"

                completion_task = asyncio.create_task(ui.read())
                await wait_for_prompt(ui, completion_task)
                assert (
                    ui._get_prompt_session().app.vi_state.input_mode is InputMode.INSERT
                )
                assert ui._status_text().endswith(" · I")
                pipe_input.send_text("/rev")
                await asyncio.sleep(0.01)
                assert ui._get_prompt_session().default_buffer.text == "/rev"
                pipe_input.send_text("\x1b")
                await wait_for_navigation(ui)
                pipe_input.send_text("\t")
                await asyncio.sleep(0.01)
                assert ui._get_prompt_session().default_buffer.text == "/rev"
                assert not completion_task.done()
                pipe_input.send_text("\r")
                assert await asyncio.wait_for(completion_task, 1) == "/rev"
            finally:
                ui.close()

    asyncio.run(scenario())


def test_active_turn_silences_and_discards_terminal_typeahead(
    tmp_path: Path,
) -> None:
    master, slave = os.openpty()
    input_stream = os.fdopen(os.dup(slave), "r", encoding="utf-8")
    attributes = termios.tcgetattr(slave)
    local_flags = 3
    attributes[local_flags] |= termios.ECHO | termios.ECHONL | termios.ISIG
    termios.tcsetattr(slave, termios.TCSANOW, attributes)
    ui = make_ui(
        workspace=tmp_path,
        interactive=True,
        input_stream=input_stream,
        live=False,
    )

    try:
        ui.turn_started()
        active_attributes = termios.tcgetattr(slave)
        assert active_attributes[local_flags] & termios.ISIG
        assert not active_attributes[local_flags] & termios.ECHO
        assert not active_attributes[local_flags] & termios.ECHONL
        assert not active_attributes[local_flags] & termios.ICANON

        os.write(master, b"discard me\n")
        assert select.select([master], [], [], 0.05)[0] == []

        ui.turn_outcome(TurnSucceeded("done", ()))
        restored_attributes = termios.tcgetattr(slave)
        assert restored_attributes[local_flags] & termios.ECHO
        assert restored_attributes[local_flags] & termios.ECHONL
        assert select.select([slave], [], [], 0)[0] == []

        os.write(master, b"next\n")
        assert select.select([slave], [], [], 0.2)[0] == [slave]
        assert os.read(slave, 1024) == b"next\n"
    finally:
        ui.close()
        input_stream.close()
        os.close(master)
        os.close(slave)


def test_redirected_read_exits_promptly_on_sigint(tmp_path: Path) -> None:
    child = textwrap.dedent(
        f"""
        import asyncio
        import sys
        from io import StringIO

        from rich.console import Console

        from ghostwheel.app_info import AppInfo
        from ghostwheel.terminal_ui import TerminalUI

        ui = TerminalUI(
            Console(file=StringIO()),
            session=object(),
            app_info=AppInfo({str(tmp_path)!r}, "provider", "model", "read-only"),
            interactive=False,
            input_stream=sys.stdin,
            live=False,
        )

        async def main():
            asyncio.get_running_loop().call_later(
                0.05,
                lambda: print("READY", flush=True),
            )
            try:
                await ui.read()
            finally:
                ui.close()

        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            pass
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", child],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        assert process.stdout is not None
        assert select.select([process.stdout], [], [], 2)[0] == [process.stdout]
        assert process.stdout.readline() == "READY\n"
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=2) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


def test_sigterm_restores_active_turn_terminal_echo(tmp_path: Path) -> None:
    master, slave = os.openpty()
    local_flags = 3
    attributes = termios.tcgetattr(slave)
    attributes[local_flags] |= (
        termios.ECHO | termios.ECHONL | termios.ICANON | termios.ISIG
    )
    termios.tcsetattr(slave, termios.TCSANOW, attributes)
    child = textwrap.dedent(
        f"""
        import signal
        import sys

        from rich.console import Console

        from ghostwheel.app_info import AppInfo
        from ghostwheel.terminal_ui import TerminalUI

        ui = TerminalUI(
            Console(file=sys.stdout),
            session=object(),
            app_info=AppInfo({str(tmp_path)!r}, "provider", "model", "read-only"),
            interactive=True,
            input_stream=sys.stdin,
            live=False,
        )
        ui.turn_started()
        print("READY", flush=True)
        signal.pause()
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", child],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
    )

    try:
        read_until(master, b"READY")
        active_flags = termios.tcgetattr(slave)[local_flags]
        assert not active_flags & termios.ECHO
        assert not active_flags & termios.ICANON
        assert active_flags & termios.ISIG
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=2) == -signal.SIGTERM
        restored_flags = termios.tcgetattr(slave)[local_flags]
        assert restored_flags & termios.ECHO
        assert restored_flags & termios.ECHONL
        assert restored_flags & termios.ICANON
        assert restored_flags & termios.ISIG
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        os.close(master)
        os.close(slave)


def test_sigterm_restores_prompt_terminal_modes(tmp_path: Path) -> None:
    master, slave = os.openpty()
    local_flags = 3
    input_flags = 0
    local_mask = termios.ECHO | termios.ICANON | termios.IEXTEN | termios.ISIG
    input_mask = (
        termios.IXON | termios.IXOFF | termios.ICRNL | termios.INLCR | termios.IGNCR
    )
    attributes = termios.tcgetattr(slave)
    attributes[local_flags] |= local_mask
    attributes[input_flags] |= termios.IXON | termios.ICRNL
    termios.tcsetattr(slave, termios.TCSANOW, attributes)
    original = termios.tcgetattr(slave)
    child = textwrap.dedent(
        f"""
        import asyncio
        import os
        import sys

        from rich.console import Console

        from ghostwheel.app_info import AppInfo
        from ghostwheel.terminal_ui import TerminalUI

        ui = TerminalUI(
            Console(file=sys.stdout),
            session=object(),
            app_info=AppInfo({str(tmp_path)!r}, "provider", "model", "read-only"),
            interactive=True,
            input_stream=sys.stdin,
            live=False,
        )

        async def main():
            asyncio.get_running_loop().call_later(
                0.1,
                lambda: os.write(1, b"READY\\n"),
            )
            try:
                await ui.read()
            finally:
                ui.close()

        asyncio.run(main())
        """
    )
    environment = os.environ.copy()
    environment.update(
        TERM="xterm-256color",
        PROMPT_TOOLKIT_NO_CPR="1",
        PYTHONUNBUFFERED="1",
    )
    process = subprocess.Popen(
        [sys.executable, "-c", child],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
        env=environment,
    )

    try:
        read_until(master, b"READY")
        assert not termios.tcgetattr(slave)[local_flags] & termios.ECHO
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=2) == -signal.SIGTERM
        restored = termios.tcgetattr(slave)
        assert restored[local_flags] & local_mask == original[local_flags] & local_mask
        assert restored[input_flags] & input_mask == original[input_flags] & input_mask
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        os.close(master)
        os.close(slave)


def test_context_and_vim_mode_are_exposed_in_the_ruled_status(tmp_path: Path) -> None:
    ui = make_ui(
        workspace=tmp_path,
        interactive=True,
        prompt_input=None,
        prompt_output=DummyOutput(),
        vim_mode=True,
        live=False,
    )

    assert ui.context_status == "~1.2k/16k"
    assert ui._status_text() == "~1.2k/16k · I"
    toolbar = str(ui._bottom_toolbar())
    assert "─" in toolbar
    assert "~1.2k/16k · I" in toolbar
    assert ui._get_prompt_session().app.editing_mode is EditingMode.VI

    ui.session.estimated_context_tokens = 948  # type: ignore[attr-defined]
    ui.session.context_tokens_estimated = False  # type: ignore[attr-defined]
    assert ui._status_text() == "948/16k · I"


def test_events_require_an_active_turn_and_dynamic_text_stays_literal(
    tmp_path: Path,
) -> None:
    output = StringIO()
    ui = make_ui(
        workspace=tmp_path,
        output=output,
        interactive=False,
        live=False,
    )

    with pytest.raises(RuntimeError, match="outside an active turn"):
        asyncio.run(ui.handle_event(TextOutput("orphaned")))

    ui.turn_started()
    asyncio.run(ui.handle_event(TextOutput("# Heading\n\n[bold]literal[/bold]")))
    asyncio.run(ui.handle_event(ToolStarted("read", "{'path': '[/]'}", call_id="1")))
    asyncio.run(ui.handle_event(ToolFinished("read", "[/]", call_id="1")))
    asyncio.run(ui.handle_event(ToolFailed("grep", "[bad]", call_id="missing")))
    ui.turn_outcome(
        TurnSucceeded("# Heading\n\n**strong** and [bold]literal[/bold]", ())
    )

    rendered = output.getvalue()
    assert "Heading" in rendered
    assert "strong" in rendered
    assert "[bold]literal[/bold]" in rendered
    assert "read" in rendered and "[/]" in rendered
    assert "grep" in rendered and "[bad]" in rendered
    assert rendered.count("[bold]literal[/bold]") == 1


def test_help_and_compaction_reflect_the_new_terminal_ui(tmp_path: Path) -> None:
    output = StringIO()
    ui = make_ui(
        workspace=tmp_path,
        output=output,
        interactive=False,
        live=False,
    )

    ui.help()
    ui.history_compacted(12_000, 4_200)

    rendered = output.getvalue()
    assert "Ctrl+J" in rendered
    assert "Ctrl+Q" in rendered
    assert "Ctrl+O" not in rendered
    assert "list available tools and the active profile" in rendered
    assert "Context compacted: 12k → ~4.2k." in rendered


def test_tools_info_lists_every_available_tool_and_description(tmp_path: Path) -> None:
    output = StringIO()
    tools = (
        ToolInfo("read", "Read a text file."),
        ToolInfo("ls", "List a directory."),
        ToolInfo("grep", "Search text files."),
        ToolInfo("bash", "Run a shell command."),
    )
    ui = make_ui(
        workspace=tmp_path,
        app_info=AppInfo(
            str(tmp_path),
            "provider",
            "model",
            "full",
            tools,
        ),
        output=output,
        width=120,
        interactive=False,
        live=False,
    )

    ui.tools_info()

    rendered = output.getvalue()
    assert "Tool profile  full" in rendered
    assert "Available tools (4)" in rendered
    for tool in tools:
        assert tool.name in rendered
        assert tool.description in rendered
    assert "Shell commands run with unrestricted environment access." in rendered


def test_tools_info_handles_a_profile_with_no_tools(tmp_path: Path) -> None:
    output = StringIO()
    ui = make_ui(
        workspace=tmp_path,
        output=output,
        interactive=False,
        live=False,
    )

    ui.tools_info()

    rendered = output.getvalue()
    assert "Available tools (0)" in rendered
    assert "None for this profile." in rendered
    assert "unrestricted environment access" not in rendered


@pytest.mark.parametrize("width", [20, 40, 80])
def test_live_region_is_bounded_to_six_visual_rows(tmp_path: Path, width: int) -> None:
    ui = make_ui(
        workspace=tmp_path,
        force_terminal=True,
        width=width,
        interactive=True,
        live=True,
    )
    ui._turn.apply(ToolStarted("first-tool-with-a-long-name", "x" * 300))
    ui._turn.apply(ToolStarted("second-tool-with-a-long-name", "y" * 300))
    ui._turn.apply(ToolStarted("third-tool-with-a-long-name", "z" * 300))
    ui._turn.apply(TextOutput("\n".join("answer " + "q" * 300 for _ in range(20))))

    rendered_lines = ui.console.render_lines(ui._active_renderable(), pad=False)

    assert len(rendered_lines) <= 6
    rendered = "".join(segment.text for line in rendered_lines for segment in line)
    assert "~1.2k/16k · I" in rendered
