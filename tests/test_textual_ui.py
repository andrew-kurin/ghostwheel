from __future__ import annotations

import asyncio
from io import StringIO

from rich.console import Console
from textual import events
from textual._xterm_parser import XTermParser
from textual.drivers.linux_driver import LinuxDriver

import ghostwheel.textual_ui as textual_ui
from ghostwheel.events import ThinkingOutput, ToolFinished, ToolStarted
from ghostwheel.rich_ui import AppInfo
from ghostwheel.session import TurnSucceeded
from ghostwheel.textual_ui import (
    MODIFY_OTHER_KEYS_ENABLE,
    MODIFY_OTHER_KEYS_RESET,
    GhostwheelApp,
    GhostwheelTerminalDriver,
    TurnView,
    VimMode,
    _help_panel,
)


class FakeSession:
    def __init__(self) -> None:
        self.history: tuple[object, ...] = ()
        self.last_compacted_turns = 0
        self.sent: list[str] = []
        self.sent_event = asyncio.Event()

    @property
    def turn_count(self) -> int:
        return len(self.sent)

    async def send(self, prompt: str) -> TurnSucceeded[str]:
        self.sent.append(prompt)
        self.sent_event.set()
        return TurnSucceeded(f"Received {prompt}", ())

    def clear(self) -> None:
        self.history = ()


class FakeReviews:
    async def review(self, *_args, **_kwargs):
        raise AssertionError("review was not expected")


class BlockingSession(FakeSession):
    def __init__(self) -> None:
        super().__init__()
        self.cancelled = asyncio.Event()

    async def send(self, prompt: str) -> TurnSucceeded[str]:
        self.sent.append(prompt)
        self.sent_event.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


def make_app(
    session: FakeSession | None = None,
    *,
    vim_mode: bool = True,
) -> GhostwheelApp:
    return GhostwheelApp(
        Console(),
        session or FakeSession(),
        FakeReviews(),  # type: ignore[arg-type]
        app_info=AppInfo(
            workspace="/tmp/workspace",
            provider="ollama",
            model="test-model",
            tool_profile="full",
        ),
        max_turns=20,
        vim_mode=vim_mode,
    )


def test_shift_enter_inserts_a_newline_and_plain_enter_submits() -> None:
    async def scenario() -> None:
        session = FakeSession()
        app = make_app(session, vim_mode=False)
        async with app.run_test(size=(100, 30)) as pilot:
            assert app.composer.vim_enabled is False
            assert str(app.composer_prompt.render()) == "You ›"
            await pilot.press(
                "t", "e", "s", "t", "shift+enter", "s", "t", "i", "l", "l"
            )
            await pilot.pause()

            assert app.composer.text == "test\nstill"
            assert session.sent == []

            await pilot.press("enter")
            await asyncio.wait_for(session.sent_event.wait(), 1)
            await pilot.pause()
            assert session.sent == ["test\nstill"]
            assert app.composer.text == ""

            await pilot.press("a", "shift+\r", "b")
            assert app.composer.text == "a\nb"

            app.composer.load_text("")
            await pilot.press(*"mapped", "ctrl+j", *"newline")
            assert app.composer.text == "mapped\nnewline"
            app.exit()

    asyncio.run(scenario())


def test_composer_grows_upward_and_shrinks_after_submit() -> None:
    async def scenario() -> None:
        session = FakeSession()
        app = make_app(session)
        async with app.run_test(size=(100, 30)) as pilot:
            initial_region = app.composer_shell.region
            initial_visible_rows = app.composer.region.intersection(
                app.composer_shell.content_region
            ).height
            assert initial_region.height == 3
            assert initial_visible_rows == 2

            await pilot.press(*"one", "shift+enter", *"two", "shift+enter", *"three")
            await pilot.pause()

            expanded_region = app.composer_shell.region
            visible_rows = app.composer.region.intersection(
                app.composer_shell.content_region
            ).height
            assert app.composer.text == "one\ntwo\nthree"
            assert expanded_region.height == 4
            assert visible_rows == 3
            assert expanded_region.bottom == initial_region.bottom
            assert expanded_region.y < initial_region.y

            await pilot.press("enter")
            await asyncio.wait_for(session.sent_event.wait(), 1)
            await pilot.pause()

            assert app.composer.text == ""
            assert app.composer_shell.region.height == 3
            assert (
                app.composer.region.intersection(
                    app.composer_shell.content_region
                ).height
                == initial_visible_rows
            )
            app.exit()

    asyncio.run(scenario())


def test_composer_resizes_for_multiline_prompt_history() -> None:
    async def scenario() -> None:
        app = make_app()
        app.history.append("short")
        app.history.append("one\ntwo\nthree\nfour")
        app.composer._history_index = len(app.history.entries)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("up")
            await pilot.pause()
            assert app.composer.text == "one\ntwo\nthree\nfour"
            assert app.composer_shell.region.height == 5

            app.composer.move_cursor((0, 0))
            await pilot.press("up")
            await pilot.pause()
            assert app.composer.text == "short"
            assert app.composer_shell.region.height == 3

            await pilot.press("down")
            await pilot.pause()
            assert app.composer.text == "one\ntwo\nthree\nfour"
            assert app.composer_shell.region.height == 5

            await pilot.press("down")
            await pilot.pause()
            assert app.composer.text == ""
            assert app.composer_shell.region.height == 3
            app.exit()

    asyncio.run(scenario())


def test_composer_tracks_soft_wrap_and_keeps_transcript_visible() -> None:
    async def scenario() -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            app.composer.load_text("x" * 60)
            await pilot.pause()
            assert app.composer.wrapped_document.height == 1
            assert app.composer_shell.region.height == 3

            await pilot.resize_terminal(35, 30)
            await pilot.pause()
            wrapped_height = app.composer.wrapped_document.height
            assert wrapped_height > 2
            assert app.composer_shell.region.height == wrapped_height + 1

            await pilot.resize_terminal(100, 30)
            await pilot.pause()
            assert app.composer.wrapped_document.height == 1
            assert app.composer_shell.region.height == 3

            app.composer.load_text("\n".join(f"line {index}" for index in range(20)))
            app.composer.move_cursor(app.composer.document.end)
            await pilot.resize_terminal(40, 12)
            await pilot.pause()
            visible_rows = app.composer.region.intersection(
                app.composer_shell.content_region
            ).height
            assert visible_rows == 4
            assert app.transcript.region.height == 4
            assert app.composer.wrapped_document.height > visible_rows
            assert app.composer.scroll_offset.y > 0
            app.exit()

    asyncio.run(scenario())


def test_vim_mode_starts_in_insert_and_resets_after_submit() -> None:
    async def scenario() -> None:
        session = FakeSession()
        app = make_app(session, vim_mode=True)
        async with app.run_test(size=(100, 30)) as pilot:
            assert app.composer.vim_mode is VimMode.INSERT
            assert str(app.composer_prompt.render()) == "You I›"

            await pilot.press(*"hello", "escape")
            assert app.composer.vim_mode is VimMode.NORMAL
            assert app.composer.cursor_location == (0, 4)
            assert str(app.composer_prompt.render()) == "You N›"

            await pilot.press("h", "q")
            assert app.composer.text == "hello"
            assert app.composer.cursor_location == (0, 3)

            await pilot.press("i", "!", "enter")
            await asyncio.wait_for(session.sent_event.wait(), 1)
            await pilot.pause()
            assert session.sent == ["hel!lo"]
            assert app.composer.text == ""
            assert app.composer.vim_mode is VimMode.INSERT
            assert str(app.composer_prompt.render()) == "You I›"
            app.exit()

    asyncio.run(scenario())


def test_vim_motions_use_logical_lines_and_word_boundaries() -> None:
    async def scenario() -> None:
        app = make_app(vim_mode=True)
        async with app.run_test(size=(30, 20)) as pilot:
            app.composer.load_text("one two\nx\nthree four")
            app.composer.move_cursor((0, 1))
            await pilot.press("escape")

            await pilot.press("w")
            assert app.composer.cursor_location == (0, 4)
            await pilot.press("e")
            assert app.composer.cursor_location == (0, 6)
            await pilot.press("b")
            assert app.composer.cursor_location == (0, 4)
            await pilot.press("$")
            assert app.composer.cursor_location == (0, 6)

            await pilot.press("j", "j")
            assert app.composer.cursor_location == (2, 6)
            await pilot.press("0")
            assert app.composer.cursor_location == (2, 0)
            await pilot.press("right", "left", "end", "home")
            assert app.composer.cursor_location == (2, 0)
            await pilot.press("g", "g")
            assert app.composer.cursor_location == (0, 0)
            await pilot.press("G")
            assert app.composer.cursor_location == (2, 0)
            app.exit()

    asyncio.run(scenario())


def test_vim_insert_commands_open_and_position_text() -> None:
    async def scenario() -> None:
        app = make_app(vim_mode=True)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press(*"  cat", "escape", "I", "X", "escape", "A", "!")
            assert app.composer.text == "  Xcat!"

            await pilot.press("escape", "o", *"below", "escape", "O", *"above")
            assert app.composer.text == "  Xcat!\nabove\nbelow"
            assert app.composer.vim_mode is VimMode.INSERT

            app.composer.load_text("   ")
            app.composer.move_cursor((0, 3))
            await pilot.press("escape", "I", "X")
            assert app.composer.text == "   X"
            app.exit()

    asyncio.run(scenario())


def test_vim_delete_yank_paste_undo_and_redo() -> None:
    async def scenario() -> None:
        app = make_app(vim_mode=True)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press(*"one two", "escape", "0", "d", "w")
            assert app.composer.text == "two"

            await pilot.press("u")
            assert app.composer.text == "one two"
            await pilot.press("ctrl+r")
            assert app.composer.text == "two"

            app.composer.load_text("aa\nbb\ncc")
            app.composer.move_cursor((1, 0))
            await pilot.press("d", "d")
            assert app.composer.text == "aa\ncc"
            assert app.composer.cursor_location == (1, 0)

            await pilot.press("y", "y", "p")
            assert app.composer.text == "aa\ncc\ncc"

            app.composer.load_text("aa\nbb")
            app.composer.move_cursor((1, 1))
            await pilot.press("d", "d")
            assert app.composer.text == "aa"
            assert app.composer.cursor_location == (0, 0)

            app.composer.load_text("foo\nbar")
            app.composer.move_cursor((0, 0))
            await pilot.press("d", "w")
            assert app.composer.text == "\nbar"
            await pilot.press("u")
            assert app.composer.text == "foo\nbar"

            app.composer.move_cursor((1, 0))
            await pilot.press("y", "b", "p")
            assert app.composer.text == "foo\nfoo\nbar"
            app.exit()

    asyncio.run(scenario())


def test_vim_change_and_open_line_are_single_undo_units() -> None:
    async def scenario() -> None:
        app = make_app(vim_mode=True)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press(*"one two", "escape", "0", "c", "w", *"new", "escape")
            assert app.composer.text == "new two"
            await pilot.press("u")
            assert app.composer.text == "one two"
            await pilot.press("ctrl+r")
            assert app.composer.text == "new two"
            app.exit()

        app = make_app(vim_mode=True)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press(*"one", "escape", "o", *"two", "escape")
            assert app.composer.text == "one\ntwo"
            await pilot.press("u")
            assert app.composer.text == "one"
            await pilot.press("ctrl+r")
            assert app.composer.text == "one\ntwo"
            app.exit()

    asyncio.run(scenario())


def test_vim_empty_line_register_repeated_e_and_empty_change() -> None:
    async def scenario() -> None:
        app = make_app(vim_mode=True)
        async with app.run_test(size=(100, 30)) as pilot:
            app.composer.load_text("one two")
            app.composer.move_cursor((0, 1))
            await pilot.press("escape", "e")
            assert app.composer.cursor_location == (0, 2)
            await pilot.press("e")
            assert app.composer.cursor_location == (0, 6)

            app.composer.load_text("aa\n\nbb")
            app.composer.move_cursor((1, 0))
            await pilot.press("y", "y", "p")
            assert app.composer.text == "aa\n\n\nbb"

            app.composer.load_text("")
            app.composer.move_cursor((0, 0))
            await pilot.press("C", "x", "escape", "u")
            assert app.composer.text == ""
            app.exit()

    asyncio.run(scenario())


def test_vim_change_word_preserves_logical_line_boundaries() -> None:
    async def scenario() -> None:
        app = make_app(vim_mode=True)
        async with app.run_test(size=(100, 30)) as pilot:
            app.composer.load_text("aa  \nbb")
            app.composer.move_cursor((0, 2))
            app.composer._set_vim_mode(VimMode.NORMAL)
            await pilot.press("c", "w", "X")
            assert app.composer.text == "aaX\nbb"

            await pilot.press("escape")
            app.composer.load_text("aa\n\nbb")
            app.composer.move_cursor((1, 0))
            await pilot.press("c", "w", "X")
            assert app.composer.text == "aa\nX\nbb"
            app.exit()

    asyncio.run(scenario())


def test_vim_unfinished_operator_is_cleared_by_global_and_priority_keys() -> None:
    async def scenario() -> None:
        app = make_app(vim_mode=True)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press(*"one two", "escape", "0", "d", "ctrl+o", "w")
            assert app.composer.text == "one two"
            assert app.composer.cursor_location == (0, 4)

            await pilot.press("d", "shift+enter", "u")
            assert app.composer.text == "one two"
            app.exit()

    asyncio.run(scenario())


def test_vim_normal_mode_blocks_terminal_paste_but_keeps_newline_shortcuts() -> None:
    async def scenario() -> None:
        app = make_app(vim_mode=True)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("a", "b", "escape")
            app.composer.post_message(events.Paste("not inserted"))
            await pilot.pause()
            assert app.composer.text == "ab"

            await pilot.press("shift+enter")
            assert app.composer.text == "a\nb"
            await pilot.press("ctrl+j")
            assert app.composer.text == "a\n\nb"
            assert app.composer.vim_mode is VimMode.NORMAL
            app.exit()

    asyncio.run(scenario())


def test_vim_shortcuts_only_appear_in_vim_help() -> None:
    def rendered(vim_mode: bool) -> str:
        output = StringIO()
        console = Console(file=output, color_system=None, width=100)
        console.print(_help_panel(vim_mode=vim_mode))
        return output.getvalue()

    assert "Vim prompt editing" not in rendered(False)
    assert "Vim prompt editing" in rendered(True)
    assert "Esc / i a I A" in rendered(True)


def test_textual_parser_distinguishes_shift_enter_protocol_sequences() -> None:
    kitty_events = list(XTermParser().feed("\x1b[13;2u"))
    xterm_events = list(XTermParser().feed("\x1b[27;2;13~"))
    mapped_events = list(XTermParser().feed("\n"))

    assert len(kitty_events) == 1
    assert kitty_events[0].key == "shift+enter"  # type: ignore[attr-defined]
    assert len(xterm_events) == 1
    assert xterm_events[0].key == "shift+\r"  # type: ignore[attr-defined]
    assert len(mapped_events) == 1
    assert mapped_events[0].key == "ctrl+j"  # type: ignore[attr-defined]


def test_terminal_driver_enables_and_resets_modify_other_keys(monkeypatch) -> None:
    calls: list[str] = []
    driver = object.__new__(GhostwheelTerminalDriver)
    driver._writer_thread = object()  # type: ignore[assignment]
    driver.write = calls.append  # type: ignore[method-assign]
    monkeypatch.setattr(
        LinuxDriver,
        "start_application_mode",
        lambda _self: calls.append("start"),
    )
    monkeypatch.setattr(
        LinuxDriver,
        "stop_application_mode",
        lambda _self: calls.append("stop"),
    )

    driver.start_application_mode()
    driver.stop_application_mode()

    assert calls == [
        "start",
        MODIFY_OTHER_KEYS_ENABLE,
        MODIFY_OTHER_KEYS_RESET,
        "stop",
    ]


def test_macos_shift_state_recovers_a_bare_enter(monkeypatch) -> None:
    async def scenario() -> None:
        session = FakeSession()
        app = make_app(session)
        monkeypatch.setattr(textual_ui, "macos_shift_pressed", lambda: True)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("a", "enter", "b")
            await pilot.pause()

            assert app.composer.text == "a\nb"
            assert session.sent == []
            app.exit()

    asyncio.run(scenario())


def test_ctrl_o_reflows_existing_details_on_expand_and_collapse() -> None:
    async def scenario() -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            app.presenter.turn_started()
            await app.presenter.handle_event(ThinkingOutput("retained thinking"))
            await app.presenter.handle_event(
                ToolStarted("read", "{'path': 'README.md'}", call_id="call-1")
            )
            await app.presenter.handle_event(
                ToolFinished("read", "retained result", call_id="call-1")
            )
            app.presenter.turn_outcome(TurnSucceeded("Done", ()))
            await pilot.pause()

            turn = app.query_one(TurnView)
            assert turn.tool_summaries.display is True
            assert turn.thinking_detail.display is False
            assert turn.tool_details.display is False

            await pilot.press("ctrl+o")
            await pilot.pause()
            assert turn.thinking_detail.display is True
            assert turn.tool_details.display is True

            await pilot.press("ctrl+o")
            await pilot.pause()
            assert turn.thinking_detail.display is False
            assert turn.tool_details.display is False
            assert turn._thinking == "retained thinking"
            assert turn._tools[0].detail == "retained result"
            assert len(app.query(".system-message")) == 0
            app.exit()

    asyncio.run(scenario())


def test_history_navigation_keeps_multiline_cursor_navigation_local() -> None:
    async def scenario() -> None:
        app = make_app()
        app.history.append("older prompt")
        app.composer._history_index = len(app.history.entries)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("up")
            assert app.composer.text == "older prompt"
            await pilot.press("down")
            assert app.composer.text == ""

            app.composer.load_text("first\nsecond")
            app.composer.move_cursor((1, 3))
            await pilot.press("up")
            assert app.composer.text == "first\nsecond"
            assert app.composer.cursor_location[0] == 0
            app.exit()

    asyncio.run(scenario())


def test_tab_completes_commands_and_review_paths(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "README.md").write_text("test")
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("/", "r", "e", "t", "tab")
            assert app.composer.text == "/retry"

            app.composer.load_text("/review READ")
            app.composer.move_cursor((0, len(app.composer.text)))
            await pilot.press("tab")
            assert app.composer.text == "/review README.md"
            app.exit()

    asyncio.run(scenario())


def test_ctrl_c_cancels_the_active_turn_without_exiting() -> None:
    async def scenario() -> None:
        session = BlockingSession()
        app = make_app(session)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("w", "a", "i", "t", "enter")
            await asyncio.wait_for(session.sent_event.wait(), 1)
            assert app.cancellation.active is True

            await pilot.press("ctrl+c")
            await asyncio.wait_for(session.cancelled.wait(), 1)
            await pilot.pause()

            assert app.cancellation.active is False
            assert app.presenter.current_turn is None
            assert app.is_running is True
            app.exit()

    asyncio.run(scenario())
