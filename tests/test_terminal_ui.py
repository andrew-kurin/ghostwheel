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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.keys import Keys
from rich.console import Console

import ghostwheel.terminal_io as terminal_io
import ghostwheel.terminal_ui as terminal_ui_module
from ghostwheel.app_info import AppInfo, ModelInfo, ToolInfo, ToolSetInfo
from ghostwheel.events import TextOutput, ToolFailed, ToolFinished, ToolStarted
from ghostwheel.review import ReviewFailed
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


class FakeReviews:
    async def review(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("review was not expected")


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
    session: FakeSession | None = None,
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
        session=session or FakeSession(),
        app_info=app_info
        or AppInfo(
            str(workspace),
            ModelInfo("provider", "model"),
            ModelInfo("provider", "model"),
            ToolSetInfo("read-only"),
            ToolSetInfo("read-only"),
        ),
        **kwargs,
    )


async def wait_for_prompt[ResultT](
    ui: TerminalUI,
    task: asyncio.Task[ResultT],
) -> None:
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


def start_fallback_tty_reader(
    tmp_path: Path,
    *,
    controlling: bool = True,
    no_flush: bool = False,
    render_prompt: bool = False,
    verify_sigint_handler: bool = False,
    line_delimiter_slot: str | None = None,
    line_delimiter: bytes = b";",
) -> tuple[subprocess.Popen[bytes], int, int]:
    """Start a fallback reader with a real controlling pseudo-terminal."""

    master, slave = os.openpty()
    if line_delimiter_slot is not None:
        attributes = termios.tcgetattr(slave)
        attributes[6][getattr(termios, line_delimiter_slot)] = line_delimiter
        termios.tcsetattr(slave, termios.TCSANOW, attributes)
    child = textwrap.dedent(
        f"""
        import asyncio
        import fcntl
        import os
        import signal
        import sys
        import termios
        from io import StringIO

        from rich.console import Console

        from ghostwheel.app_info import AppInfo, ModelInfo, ToolSetInfo
        from ghostwheel.terminal_ui import TerminalUI

        if {controlling!r}:
            os.setsid()
            fcntl.ioctl(sys.stdin.fileno(), termios.TIOCSCTTY, 0)
        if {no_flush!r}:
            attributes = termios.tcgetattr(sys.stdin.fileno())
            attributes[3] |= termios.NOFLSH
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, attributes)
        console_output = sys.stdout if {render_prompt!r} else StringIO()
        ui = TerminalUI(
            Console(file=console_output, color_system=None),
            session=object(),
            app_info=AppInfo(
                {str(tmp_path)!r},
                ModelInfo("provider", "model"),
                ModelInfo("provider", "model"),
                ToolSetInfo("read-only"),
                ToolSetInfo("read-only"),
            ),
            interactive=False,
            input_stream=sys.stdin,
            live=False,
        )

        async def main():
            loop = asyncio.get_running_loop()
            sigint_received = asyncio.Event()
            if {verify_sigint_handler!r}:
                loop.add_signal_handler(signal.SIGINT, sigint_received.set)
            loop.call_later(
                0.05,
                lambda: os.write(1, b"READY\\n"),
            )
            try:
                value = await ui.read()
            except EOFError:
                os.write(1, b"EOF\\n")
            else:
                os.write(1, f"VALUE:{{value!r}}\\n".encode())
                if {verify_sigint_handler!r}:
                    os.kill(os.getpid(), signal.SIGINT)
                    await asyncio.wait_for(sigint_received.wait(), 1)
                    os.write(1, b"SIGINT-PRESERVED\\n")
            finally:
                ui.close()

        asyncio.run(main())
        """
    )
    environment = os.environ.copy()
    environment.update(TERM="dumb", PYTHONUNBUFFERED="1")
    process = subprocess.Popen(
        [sys.executable, "-c", child],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
        env=environment,
    )
    return process, master, slave


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
                assert (
                    await feed_prompt(
                        ui,
                        pipe_input,
                        "third\x1b[27;2;13~fourth\r",
                    )
                    == "third\nfourth"
                )
                session = ui._get_prompt_session()
                assert session.app.full_screen is False
                assert session.app.erase_when_done is False
                assert session.app.mouse_support() is False
                assert session.enable_suspend is True
                assert output.alternate_screen_entries == 0
                assert output.mouse_support_entries == 0

            finally:
                ui.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("vim_mode", [True, False])
def test_prompt_shortcuts_clear_the_draft_ignore_ctrl_q_and_quit(
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
                prompt_task = asyncio.create_task(ui.read())
                await wait_for_prompt(ui, prompt_task)
                pipe_input.send_text("discard this\x03")
                await wait_for_buffer_text(ui, "")
                assert not prompt_task.done()
                if vim_mode:
                    assert (
                        ui._get_prompt_session().app.vi_state.input_mode
                        is InputMode.INSERT
                    )

                pipe_input.send_text("unfinished\x11")
                await wait_for_buffer_text(ui, "unfinished")
                assert not prompt_task.done()

                pipe_input.send_text("\n")
                await asyncio.sleep(0.01)
                assert ui._get_prompt_session().default_buffer.text == "unfinished"
                assert not prompt_task.done()

                pipe_input.send_text("\x04")
                with pytest.raises(EOFError):
                    await asyncio.wait_for(prompt_task, 1)
            finally:
                ui.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("input_mode", [InputMode.NAVIGATION, InputMode.REPLACE])
def test_shift_enter_inserts_a_newline_in_every_vim_mode(
    tmp_path: Path,
    input_mode: InputMode,
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
                prompt_task = asyncio.create_task(ui.read())
                await wait_for_prompt(ui, prompt_task)
                pipe_input.send_text("abc")
                await wait_for_buffer_text(ui, "abc")
                ui._get_prompt_session().app.vi_state.input_mode = input_mode

                pipe_input.send_text("\x1b[27;2;13~")
                for _attempt in range(100):
                    value = ui._get_prompt_session().default_buffer.text
                    if "\n" in value:
                        break
                    await asyncio.sleep(0.01)
                else:
                    raise AssertionError("Shift+Enter did not insert a newline")
                assert not prompt_task.done()

                expected = ui._get_prompt_session().default_buffer.text
                pipe_input.send_text("\r")
                assert await asyncio.wait_for(prompt_task, 1) == expected
            finally:
                ui.close()

    asyncio.run(scenario())


def test_active_turn_uses_escape_to_cancel_and_ctrl_d_to_quit(
    tmp_path: Path,
) -> None:
    class BlockingSession(FakeSession):
        def __init__(self) -> None:
            self.started: asyncio.Queue[str] = asyncio.Queue()
            self.cancelled: list[str] = []

        async def send(self, prompt: str) -> TurnSucceeded[str]:
            await self.started.put(prompt)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.append(prompt)
                raise
            raise AssertionError("unreachable")

    async def wait_for_cancelled_output(output: StringIO, count: int) -> None:
        for _attempt in range(100):
            if output.getvalue().count("Turn cancelled.") == count:
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"turn cancellation count did not reach {count}")

    async def scenario() -> None:
        output = StringIO()
        session = BlockingSession()
        with create_pipe_input() as pipe_input:
            ui = make_ui(
                workspace=tmp_path,
                session=session,
                output=output,
                interactive=True,
                prompt_input=pipe_input,
                prompt_output=DummyOutput(),
                live=False,
            )
            run_task = asyncio.create_task(ui.run(FakeReviews()))

            await wait_for_prompt(ui, run_task)
            ui._get_prompt_session().app.ttimeoutlen = 0.05
            pipe_input.send_text("first\r")
            assert await asyncio.wait_for(session.started.get(), 1) == "first"

            # Ctrl+C, Meta/Alt keys, and complete arrow sequences are discarded.
            pipe_input.send_text("\x03\x1ba\x1b[A")
            await asyncio.sleep(0.1)
            assert session.cancelled == []
            assert not run_task.done()

            # A split arrow prefix must be allowed to finish before the VT
            # parser decides whether Escape was standalone.
            pipe_input.send_text("\x1b")
            await asyncio.sleep(0.01)
            pipe_input.send_text("[A")
            await asyncio.sleep(0.1)
            assert session.cancelled == []

            # Ordinary input and a trailing Escape can share one read; the
            # pending Escape is recognized after ttimeoutlen.
            pipe_input.send_text("x\x1b")
            await wait_for_cancelled_output(output, 1)
            assert session.cancelled == ["first"]
            await wait_for_prompt(ui, run_task)

            # Escape can arrive in the same terminal read as prompt submission.
            pipe_input.send_text("second\r\x1b")
            assert await asyncio.wait_for(session.started.get(), 1) == "second"
            await wait_for_cancelled_output(output, 2)
            await wait_for_prompt(ui, run_task)

            pipe_input.send_text("third\r")
            assert await asyncio.wait_for(session.started.get(), 1) == "third"
            pipe_input.send_text("\x04")
            await asyncio.wait_for(run_task, 1)

            assert session.cancelled[-1] == "third"
            assert output.getvalue().count("Turn cancelled.") == 2
            assert "Goodbye!" in output.getvalue()

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
                pipe_input.send_text("top\x1b[27;2;13~bottom")
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
                pipe_input.send_text("top\x1b[27;2;13~bottom\x1b")
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
    output = StringIO()
    ui = make_ui(
        workspace=tmp_path,
        output=output,
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
    assert output.getvalue() == ""


def test_fallback_prompt_requires_tty_input_and_output(tmp_path: Path) -> None:
    input_master, input_slave = os.openpty()
    input_stream = os.fdopen(os.dup(input_slave), "r", encoding="utf-8")
    redirected_output = StringIO()
    tty_input_ui = make_ui(
        workspace=tmp_path,
        output=redirected_output,
        interactive=False,
        input_stream=input_stream,
        live=False,
    )

    async def read_tty_input() -> None:
        task = asyncio.create_task(tty_input_ui.read())
        await asyncio.sleep(0)
        os.write(input_master, b"from tty\n")
        assert await asyncio.wait_for(task, 1) == "from tty"

    try:
        asyncio.run(read_tty_input())
        assert redirected_output.getvalue() == ""
    finally:
        tty_input_ui.close()
        input_stream.close()
        os.close(input_master)
        os.close(input_slave)

    output_master, output_slave = os.openpty()
    output_stream = os.fdopen(os.dup(output_slave), "w", encoding="utf-8")
    non_tty_input_ui = TerminalUI(
        Console(file=output_stream, color_system=None),
        session=FakeSession(),
        app_info=AppInfo(
            str(tmp_path),
            ModelInfo("provider", "model"),
            ModelInfo("provider", "model"),
            ToolSetInfo("read-only"),
            ToolSetInfo("read-only"),
        ),
        interactive=False,
        input_stream=StringIO("from stream\n"),
        live=False,
    )
    try:
        assert asyncio.run(non_tty_input_ui.read()) == "from stream"
        assert select.select([output_master], [], [], 0.05)[0] == []
    finally:
        non_tty_input_ui.close()
        output_stream.close()
        os.close(output_master)
        os.close(output_slave)


@pytest.mark.parametrize(
    "term",
    ["", "   ", "dumb", "DUMB", "unknown", "UnKnOwN"],
)
def test_terminal_ui_auto_detection_rejects_unsupported_terminals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    term: str,
) -> None:
    class TTYStringIO(StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setenv("TERM", term)
    input_stream = TTYStringIO()
    console = Console(file=TTYStringIO(), color_system=None)
    ui = TerminalUI(
        console,
        session=FakeSession(),
        app_info=AppInfo(
            str(tmp_path),
            ModelInfo("provider", "model"),
            ModelInfo("provider", "model"),
            ToolSetInfo("read-only"),
            ToolSetInfo("read-only"),
        ),
        input_stream=input_stream,
        live=False,
    )

    try:
        assert ui.interactive is False
        assert not ui._use_prompt_toolkit()
    finally:
        ui.close()


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


def test_stream_input_does_not_initialize_persistent_history(tmp_path: Path) -> None:
    history_path = tmp_path / "unwritable-state" / "input-history"
    ui = make_ui(
        workspace=tmp_path,
        history_path=history_path,
        interactive=False,
        input_stream=StringIO("streamed prompt\n"),
        live=False,
    )

    assert asyncio.run(ui.read()) == "streamed prompt"
    assert ui._composer.history is None
    assert ui._composer.history_store is None
    assert not history_path.parent.exists()


def test_history_write_failure_keeps_prompt_history_in_memory_and_warns_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        history_path = tmp_path / "input-history"
        output = StringIO()
        original_open = Path.open
        with create_pipe_input() as pipe_input:
            ui = make_ui(
                workspace=tmp_path,
                output=output,
                history_path=history_path,
                interactive=True,
                prompt_input=pipe_input,
                prompt_output=DummyOutput(),
                live=False,
            )
            try:
                first_prompt = asyncio.create_task(ui.read())
                await wait_for_prompt(ui, first_prompt)

                def deny_history_append(
                    path: Path,
                    mode: str = "r",
                    *args: object,
                    **kwargs: object,
                ) -> object:
                    if path == history_path and "a" in mode:
                        raise PermissionError("write denied")
                    return original_open(path, mode, *args, **kwargs)

                monkeypatch.setattr(Path, "open", deny_history_append)
                pipe_input.send_text("first prompt\r")
                assert await asyncio.wait_for(first_prompt, 1) == "first prompt"

                history_store = ui._composer.history_store
                assert history_store is not None
                assert history_store.path is None
                assert history_store.entries == ["first prompt"]
                assert output.getvalue().count("History Warning") == 1
                assert "using in-memory history" in output.getvalue()
                assert "write denied" in output.getvalue()

                assert await feed_prompt(ui, pipe_input, "second prompt\r") == (
                    "second prompt"
                )
                assert history_store.entries == ["first prompt", "second prompt"]
                assert output.getvalue().count("History Warning") == 1
            finally:
                ui.close()

    asyncio.run(scenario())


def test_close_restores_terminal_and_closes_composer_when_live_stop_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ThrowingLive:
        def stop(self) -> None:
            calls.append("live")
            raise RuntimeError("stop failed")

    calls: list[str] = []
    ui = make_ui(workspace=tmp_path, interactive=False, live=False)
    ui._live = ThrowingLive()  # type: ignore[assignment]
    monkeypatch.setattr(
        ui._terminal_guard,
        "restore",
        lambda: calls.append("restore"),
    )
    monkeypatch.setattr(ui._composer, "close", lambda: calls.append("close"))

    with pytest.raises(RuntimeError, match="stop failed"):
        ui.close()

    assert calls == ["live", "restore", "close"]
    assert ui._live is None


def test_reset_turn_restores_terminal_and_state_when_live_stop_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ThrowingLive:
        def stop(self) -> None:
            raise RuntimeError("stop failed")

    restored: list[bool] = []
    ui = make_ui(workspace=tmp_path, interactive=False, live=False)
    ui._active = True
    ui._turn_uses_live = True
    ui._turn.answer = "stale"
    ui._live = ThrowingLive()  # type: ignore[assignment]
    monkeypatch.setattr(
        ui._terminal_guard,
        "restore",
        lambda: restored.append(True),
    )

    with pytest.raises(RuntimeError, match="stop failed"):
        ui._reset_turn("Ready")

    assert restored == [True]
    assert ui._active is False
    assert ui._turn_uses_live is False
    assert ui._turn.status == "Ready"
    assert ui._turn.answer == ""
    assert ui._live is None


@pytest.mark.parametrize("failure", ["live", "output"])
def test_finish_activity_restores_terminal_on_rendering_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    class ThrowingLive:
        def update(self, *_args: object, **_kwargs: object) -> None:
            pass

        def stop(self) -> None:
            if failure == "live":
                raise RuntimeError("render failed")

    restored: list[bool] = []
    ui = make_ui(workspace=tmp_path, interactive=False, live=False)
    ui._active = True
    ui._turn_uses_live = failure == "live"
    ui._live = ThrowingLive() if failure == "live" else None  # type: ignore[assignment]
    ui._stream_at_line_start = failure != "output"
    monkeypatch.setattr(
        ui._terminal_guard,
        "restore",
        lambda: restored.append(True),
    )
    if failure == "output":

        def fail_output(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("render failed")

        monkeypatch.setattr(ui.console, "print", fail_output)

    with pytest.raises(RuntimeError, match="render failed"):
        ui._finish_activity()

    assert restored == [True]
    assert ui._live is None


def test_finish_activity_keeps_turn_input_guarded_until_success_is_rendered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restored: list[bool] = []
    ui = make_ui(workspace=tmp_path, interactive=False, live=False)
    ui._active = True
    monkeypatch.setattr(
        ui._terminal_guard,
        "restore",
        lambda: restored.append(True),
    )

    ui._finish_activity()

    assert restored == []


@pytest.mark.parametrize("failure", ["live", "output"])
def test_turn_started_rolls_back_terminal_state_on_presentation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    class ThrowingLive:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def start(self, *, refresh: bool) -> None:
            assert refresh is True
            raise RuntimeError("start failed")

        def stop(self) -> None:
            pass

    calls: list[str] = []
    ui = make_ui(workspace=tmp_path, interactive=False, live=False)
    ui.live_enabled = failure == "live"
    monkeypatch.setattr(
        ui._terminal_guard,
        "silence",
        lambda: calls.append("silence"),
    )
    monkeypatch.setattr(
        ui._terminal_guard,
        "restore",
        lambda: calls.append("restore"),
    )
    if failure == "live":
        monkeypatch.setattr(terminal_ui_module, "Live", ThrowingLive)
    else:

        def fail_output(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("start failed")

        monkeypatch.setattr(ui.console, "print", fail_output)

    with pytest.raises(RuntimeError, match="start failed"):
        ui.turn_started()

    assert calls == ["restore", "silence", "restore"]
    assert ui._active is False
    assert ui._turn_uses_live is False
    assert ui._live is None


@pytest.mark.parametrize("outcome", ["cancelled", "chat", "review"])
def test_outcome_rendering_failure_resets_terminal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    restored: list[bool] = []
    ui = make_ui(workspace=tmp_path, interactive=False, live=False)
    ui._active = True
    monkeypatch.setattr(
        ui._terminal_guard,
        "restore",
        lambda: restored.append(True),
    )

    def fail_output(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("outcome failed")

    monkeypatch.setattr(ui.console, "print", fail_output)

    with pytest.raises(RuntimeError, match="outcome failed"):
        if outcome == "cancelled":
            ui.turn_cancelled()
        elif outcome == "chat":
            ui.turn_outcome(TurnSucceeded("answer", ()))
        else:
            ui.review_outcome(ReviewFailed("review failed"))

    assert restored == [True]
    assert ui._active is False
    assert ui._turn_uses_live is False


def test_private_history_round_trips_multiline_entries(tmp_path: Path) -> None:
    history_path = tmp_path / "state" / "ghostwheel" / "history"
    ui = make_ui(
        workspace=tmp_path,
        history_path=history_path,
        interactive=False,
        live=False,
    )

    ui._get_prompt_history().append_string("first")
    ui._get_prompt_history().append_string("more\nthan one line")

    assert stat.S_IMODE(history_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(history_path.stat().st_mode) == 0o600
    restored = make_ui(
        workspace=tmp_path,
        history_path=history_path,
        interactive=False,
        live=False,
    )
    restored._get_prompt_history()
    assert restored._composer.history_store is not None
    assert restored._composer.history_store.entries == [
        "first",
        "more\nthan one line",
    ]
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
    ui._get_prompt_history().append_string("remember for this run")
    assert ui._composer.history_store is not None
    assert ui._composer.history_store.entries == ["remember for this run"]
    assert not (tmp_path / "ghostwheel").exists()


def test_completion_covers_commands_files_and_directories(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("read me", encoding="utf-8")
    (tmp_path / "src").mkdir()
    ui = make_ui(workspace=tmp_path, interactive=False, live=False)
    event = CompleteEvent(completion_requested=True)

    completer = ui._composer.completer
    command_completions = list(completer.get_completions(Document("/ret"), event))
    file_completions = list(completer.get_completions(Document("/review REA"), event))
    directory_completions = list(
        completer.get_completions(Document("/review s"), event)
    )

    assert [completion.text for completion in command_completions] == ["/retry"]
    assert [completion.text for completion in file_completions] == ["DME.md"]
    assert [completion.text for completion in directory_completions] == ["rc/"]
    assert (
        list(
            completer.get_completions(
                Document("/revX", cursor_position=4),
                event,
            )
        )
        == []
    )


def test_completion_binding_is_inert_in_vim_navigation(tmp_path: Path) -> None:
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
        assert not active_attributes[local_flags] & termios.ISIG
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


def test_fallback_tty_monitors_escape_and_ctrl_d_without_composer(
    tmp_path: Path,
) -> None:
    class RecordingCancellation:
        def __init__(self) -> None:
            self.requested = asyncio.Event()

        def cancel(self) -> bool:
            self.requested.set()
            return True

    master, slave = os.openpty()
    input_stream = os.fdopen(os.dup(slave), "r", encoding="utf-8")
    ui = make_ui(
        workspace=tmp_path,
        interactive=False,
        input_stream=input_stream,
        live=False,
    )

    async def scenario() -> None:
        assert not ui._use_prompt_toolkit()

        escape_cancellation = RecordingCancellation()
        ui.turn_started()
        with ui._capture_turn_input(escape_cancellation):
            os.write(master, b"\x1b")
            await asyncio.wait_for(escape_cancellation.requested.wait(), 1)
        assert not ui._quit_requested
        ui.turn_cancelled()

        quit_cancellation = RecordingCancellation()
        ui.turn_started()
        with ui._capture_turn_input(quit_cancellation):
            os.write(master, b"\x04")
            await asyncio.wait_for(quit_cancellation.requested.wait(), 1)
        assert ui._quit_requested
        ui.turn_cancelled()

    try:
        asyncio.run(scenario())
        restored_flags = termios.tcgetattr(slave)[3]
        assert restored_flags & termios.ECHO
        assert restored_flags & termios.ICANON
        assert restored_flags & termios.ISIG
    finally:
        ui.close()
        input_stream.close()
        os.close(master)
        os.close(slave)


def test_detached_turn_input_callback_cannot_consume_the_next_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class QueuedCallbackInput:
        closed = False

        def __init__(self) -> None:
            self.callback: Callable[[], None] | None = None
            self.read_count = 0

        @contextmanager
        def raw_mode(self) -> Iterator[None]:
            yield

        @contextmanager
        def attach(self, callback: Callable[[], None]) -> Iterator[None]:
            self.callback = callback
            yield

        def read_keys(self) -> list[KeyPress]:
            self.read_count += 1
            return [KeyPress(Keys.ControlD)]

        def flush_keys(self) -> list[KeyPress]:
            return []

    class RecordingCancellation:
        def __init__(self) -> None:
            self.calls = 0

        def cancel(self) -> bool:
            self.calls += 1
            return True

    async def scenario() -> None:
        queued_input = QueuedCallbackInput()
        cancellation = RecordingCancellation()
        monkeypatch.setattr(terminal_io, "get_typeahead", lambda _input: ())
        ui = make_ui(
            workspace=tmp_path,
            interactive=True,
            prompt_input=queued_input,
            prompt_output=DummyOutput(),
            live=False,
        )
        try:
            ui.turn_started()
            with ui._capture_turn_input(cancellation):
                assert queued_input.callback is not None
                queued_callback = queued_input.callback
            ui.turn_cancelled()

            # Reproduce an event-loop reader callback that was queued before
            # attach() detached it, but runs after the active turn has ended.
            queued_callback()
            await asyncio.sleep(0)

            assert queued_input.read_count == 0
            assert cancellation.calls == 0
            assert not ui._quit_requested
        finally:
            ui.close()

    asyncio.run(scenario())


def test_redirected_read_exits_promptly_on_sigint(tmp_path: Path) -> None:
    child = textwrap.dedent(
        f"""
        import asyncio
        import sys
        from io import StringIO

        from rich.console import Console

        from ghostwheel.app_info import AppInfo, ModelInfo, ToolSetInfo
        from ghostwheel.terminal_ui import TerminalUI

        ui = TerminalUI(
            Console(file=StringIO()),
            session=object(),
            app_info=AppInfo(
                {str(tmp_path)!r},
                ModelInfo("provider", "model"),
                ModelInfo("provider", "model"),
                ToolSetInfo("read-only"),
                ToolSetInfo("read-only"),
            ),
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
        # Startup imports can be slow on a loaded CI runner; the two-second
        # responsiveness bound begins only after the child reports readiness.
        assert select.select([process.stdout], [], [], 10)[0] == [process.stdout]
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


def test_fallback_tty_ctrl_d_discards_draft_and_quits_immediately(
    tmp_path: Path,
) -> None:
    process, master, slave = start_fallback_tty_reader(tmp_path)

    try:
        read_until(master, b"READY")
        os.write(master, b"discard this\x04")
        output = read_until(master, b"EOF")
        assert b"VALUE:" not in output
        assert process.wait(timeout=2) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        os.close(master)
        os.close(slave)


@pytest.mark.parametrize(
    "line_delimiter_slot",
    [name for name in ("VEOL", "VEOL2") if hasattr(termios, name)],
)
def test_fallback_tty_accepts_configured_canonical_line_delimiters(
    tmp_path: Path,
    line_delimiter_slot: str,
) -> None:
    process, master, slave = start_fallback_tty_reader(
        tmp_path,
        line_delimiter_slot=line_delimiter_slot,
    )

    try:
        read_until(master, b"READY")
        os.write(master, b"custom delimiter;")
        output = read_until(master, b"VALUE:'custom delimiter'")
        assert b"VALUE:'custom delimiter;'" not in output
        assert process.wait(timeout=2) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        os.close(master)
        os.close(slave)


def test_fallback_tty_renders_prompt_and_redraws_after_ctrl_c(
    tmp_path: Path,
) -> None:
    process, master, slave = start_fallback_tty_reader(
        tmp_path,
        render_prompt=True,
    )

    try:
        initial_output = read_until(master, b"READY")
        assert b"\r\n> " in initial_output

        os.write(master, b"discard this")
        os.write(master, b"\x03")
        redrawn_output = read_until(master, b"\r\n> ")
        assert redrawn_output.endswith(b"\r\n> ")

        os.write(master, b"keep this\n")
        output = read_until(master, b"VALUE:'keep this'")
        assert b"VALUE:'discard this" not in output
        assert process.wait(timeout=2) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        os.close(master)
        os.close(slave)


def test_fallback_tty_preserves_event_loop_sigint_handler(tmp_path: Path) -> None:
    process, master, slave = start_fallback_tty_reader(
        tmp_path,
        verify_sigint_handler=True,
    )

    try:
        read_until(master, b"READY")
        os.write(master, b"complete read\n")
        output = read_until(master, b"SIGINT-PRESERVED")
        assert b"VALUE:'complete read'" in output
        assert process.wait(timeout=2) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        os.close(master)
        os.close(slave)


def test_fallback_tty_ctrl_c_clears_draft_without_exiting(tmp_path: Path) -> None:
    process, master, slave = start_fallback_tty_reader(tmp_path, no_flush=True)

    try:
        read_until(master, b"READY")
        # Replacement input can already be queued when Python handles SIGINT;
        # clearing the draft must not flush bytes following Ctrl+C.
        os.write(master, b"discard this\x03keep this\n")
        output = read_until(master, b"VALUE:'keep this'")
        assert b"VALUE:'discard this" not in output
        assert process.wait(timeout=2) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        os.close(master)
        os.close(slave)


def test_fallback_tty_sigterm_restores_inherited_no_flush(tmp_path: Path) -> None:
    process, master, slave = start_fallback_tty_reader(
        tmp_path,
        controlling=False,
        no_flush=True,
    )

    try:
        read_until(master, b"READY")
        active_flags = termios.tcgetattr(slave)[3]
        assert not active_flags & termios.NOFLSH

        process.send_signal(signal.SIGTERM)

        assert process.wait(timeout=2) == -signal.SIGTERM
        restored_flags = termios.tcgetattr(slave)[3]
        assert restored_flags & termios.NOFLSH
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        os.close(master)
        os.close(slave)


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

        from ghostwheel.app_info import AppInfo, ModelInfo, ToolSetInfo
        from ghostwheel.terminal_ui import TerminalUI

        ui = TerminalUI(
            Console(file=sys.stdout),
            session=object(),
            app_info=AppInfo(
                {str(tmp_path)!r},
                ModelInfo("provider", "model"),
                ModelInfo("provider", "model"),
                ToolSetInfo("read-only"),
                ToolSetInfo("read-only"),
            ),
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
        assert not active_flags & termios.ISIG
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

        from ghostwheel.app_info import AppInfo, ModelInfo, ToolSetInfo
        from ghostwheel.terminal_ui import TerminalUI

        ui = TerminalUI(
            Console(file=sys.stdout),
            session=object(),
            app_info=AppInfo(
                {str(tmp_path)!r},
                ModelInfo("provider", "model"),
                ModelInfo("provider", "model"),
                ToolSetInfo("read-only"),
                ToolSetInfo("read-only"),
            ),
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
        width=20,
        interactive=False,
        live=False,
    )

    with pytest.raises(RuntimeError, match="outside an active turn"):
        asyncio.run(ui.handle_event(TextOutput("orphaned")))

    raw_answer = (
        "# Heading\n\n**strong** and [bold]literal[/bold] "
        "with-a-single-very-long-unwrapped-line"
    )
    ui.turn_started()
    asyncio.run(ui.handle_event(TextOutput("# Heading\n\n", starts_part=True)))
    asyncio.run(ui.handle_event(TextOutput("**strong** and ")))
    asyncio.run(
        ui.handle_event(
            TextOutput("[bold]literal[/bold] with-a-single-very-long-unwrapped-line")
        )
    )
    streamed = output.getvalue()
    assert streamed.endswith("\nGhostwheel\n" + raw_answer)
    asyncio.run(ui.handle_event(ToolStarted("read", "{'path': '[/]'}", call_id="1")))
    asyncio.run(ui.handle_event(ToolFinished("read", "[/]", call_id="1")))
    asyncio.run(ui.handle_event(ToolFailed("grep", "[bad]", call_id="missing")))
    ui.turn_outcome(TurnSucceeded(raw_answer, ()))

    rendered = output.getvalue()
    assert "# Heading" in rendered
    assert "**strong**" in rendered
    assert "[bold]literal[/bold]" in rendered
    assert "read" in rendered and "[/]" in rendered
    assert "grep" in rendered and "[bad]" in rendered
    assert rendered.count("# Heading") == 1
    assert rendered.count("**strong**") == 1
    assert rendered.count("[bold]literal[/bold]") == 1
    assert rendered.count("with-a-single-very-long-unwrapped-line") == 1
    lines = rendered.splitlines()
    streamed_answer_line = next(
        index
        for index, line in enumerate(lines)
        if "with-a-single-very-long-unwrapped-line" in line
    )
    tool_output = lines[streamed_answer_line + 1 :]
    assert tool_output[0].lstrip().startswith("▸ read")
    assert all(line for line in tool_output)


def test_non_live_success_without_text_events_prints_literal_fallback(
    tmp_path: Path,
) -> None:
    output = StringIO()
    ui = make_ui(
        workspace=tmp_path,
        output=output,
        width=20,
        interactive=False,
        live=False,
    )
    raw_answer = "**literal-markdown** with-a-single-very-long-unwrapped-line"

    ui.turn_started()
    ui.turn_outcome(TurnSucceeded(raw_answer, ()))

    rendered = output.getvalue()
    assert rendered.endswith("\nGhostwheel\n" + raw_answer + "\n")
    assert rendered.count(raw_answer) == 1


def test_non_live_streamed_success_finishes_line_before_followup(
    tmp_path: Path,
) -> None:
    output = StringIO()
    ui = make_ui(
        workspace=tmp_path,
        output=output,
        interactive=False,
        live=False,
    )

    ui.turn_started()
    asyncio.run(ui.handle_event(TextOutput("partial answer", starts_part=True)))
    ui.turn_outcome(TurnSucceeded("partial answer", ()))
    ui.history_compacted(1_200, 600)

    assert "partial answer\nContext compacted" in output.getvalue()


def test_non_live_tools_start_on_the_line_after_streamed_text_without_gaps(
    tmp_path: Path,
) -> None:
    output = StringIO()
    ui = make_ui(
        workspace=tmp_path,
        output=output,
        width=120,
        interactive=False,
        live=False,
    )

    ui.turn_started()
    asyncio.run(ui.handle_event(TextOutput("partial answer", starts_part=True)))
    asyncio.run(ui.handle_event(ToolStarted("read", "file.py", call_id="1")))
    asyncio.run(ui.handle_event(ToolFinished("read", "done", call_id="1")))

    rendered = output.getvalue()
    assert "partial answer\n  ▸ read" in rendered
    assert "partial answer\n\n" not in rendered
    lines = rendered.splitlines()
    answer_line = lines.index("partial answer")
    assert lines[answer_line + 1].lstrip().startswith("▸ read")
    assert lines[answer_line + 2].lstrip().startswith("✓ read")


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
    assert "Shift+Enter" in rendered
    assert "Ctrl+C" in rendered
    assert "Ctrl+D" in rendered
    assert "Esc" in rendered
    assert "Ctrl+J" not in rendered
    assert "Ctrl+Q" not in rendered
    assert "Ctrl+O" not in rendered
    assert "list available tools and active profiles" in rendered
    assert "Context compacted: 12k → ~4.2k." in rendered


def test_tools_info_lists_mixed_chat_and_review_capabilities(tmp_path: Path) -> None:
    output = StringIO()
    chat_tools = (
        ToolInfo("read", "Read a text file."),
        ToolInfo("ls", "List a directory."),
    )
    review_tools = (
        ToolInfo("read", "Read a text file."),
        ToolInfo("grep", "Search text files."),
        ToolInfo("bash", "Run a shell command."),
    )
    ui = make_ui(
        workspace=tmp_path,
        app_info=AppInfo(
            str(tmp_path),
            ModelInfo("provider", "model"),
            ModelInfo("provider", "model"),
            ToolSetInfo("read-only", chat_tools),
            ToolSetInfo("full", review_tools, True),
        ),
        output=output,
        width=120,
        interactive=False,
        live=False,
    )

    ui.tools_info()

    rendered = output.getvalue()
    assert "Chat" in rendered
    assert "Review" in rendered
    assert "Tool profile  read-only" in rendered
    assert "Tool profile  full" in rendered
    assert "Available tools (2)" in rendered
    assert "Available tools (3)" in rendered
    for tool in (*chat_tools, *review_tools):
        assert tool.name in rendered
        assert tool.description in rendered
    assert rendered.count("unrestricted environment access") == 1


def test_welcome_labels_mixed_chat_and_review_profiles(tmp_path: Path) -> None:
    output = StringIO()
    ui = make_ui(
        workspace=tmp_path,
        app_info=AppInfo(
            str(tmp_path),
            ModelInfo("chat-provider", "chat-model"),
            ModelInfo("review-provider", "review-model"),
            ToolSetInfo("read-only"),
            ToolSetInfo("full", has_shell_access=True),
        ),
        output=output,
        width=160,
        interactive=False,
        live=False,
    )

    ui.welcome()

    rendered = output.getvalue()
    assert "chat chat-provider/chat-model" in rendered
    assert "review review-provider/review-model" in rendered
    assert "chat READ-ONLY" in rendered
    assert "review FULL" in rendered
    assert "(unrestricted shell)" in rendered


def test_model_info_lists_distinct_chat_and_review_models(tmp_path: Path) -> None:
    output = StringIO()
    ui = make_ui(
        workspace=tmp_path,
        app_info=AppInfo(
            str(tmp_path),
            ModelInfo("chat-provider", "chat-model"),
            ModelInfo("review-provider", "review-model"),
            ToolSetInfo("read-only"),
            ToolSetInfo("read-only"),
        ),
        output=output,
        interactive=False,
        live=False,
    )

    ui.model_info()

    rendered = output.getvalue()
    assert "Chat model    chat-provider/chat-model" in rendered
    assert "Review model  review-provider/review-model" in rendered


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
    assert rendered.count("Available tools (0)") == 2
    assert rendered.count("None for this profile.") == 2
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
