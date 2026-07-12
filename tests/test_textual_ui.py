from __future__ import annotations

import asyncio

import pytest
from rich.text import Text
from textual import events
from textual._xterm_parser import XTermParser
from textual.drivers.headless_driver import HeadlessDriver
from textual.drivers.linux_driver import LinuxDriver
from textual.pilot import Pilot
from textual.widget import Widget
from textual.worker import WorkerFailed
from textual.widgets import Static

from ghostwheel.app_info import AppInfo
from ghostwheel.events import ThinkingOutput, ToolFinished, ToolStarted
from ghostwheel.runtime_contracts import TurnNoResult, TurnSucceeded
from ghostwheel.textual_ui import (
    MODIFY_OTHER_KEYS_ENABLE,
    MODIFY_OTHER_KEYS_RESET,
    GhostwheelApp,
    GhostwheelTerminalDriver,
    TurnView,
)
from tests.textual_support import FakeReviews, FakeSession, make_app


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


async def right_click(
    app: GhostwheelApp,
    pilot: Pilot[None],
    widget: Widget,
) -> None:
    x = widget.region.x
    y = widget.region.y
    for event_type in (events.MouseDown, events.MouseUp):
        app.post_message(
            event_type(
                None,
                x,
                y,
                0,
                0,
                3,
                False,
                False,
                False,
                screen_x=x,
                screen_y=y,
            )
        )
        await pilot.pause()


def test_context_status_distinguishes_exact_and_estimated_token_usage() -> None:
    async def scenario() -> None:
        session = FakeSession()
        session.estimated_context_tokens = 1_250
        app = make_app(session)
        async with app.run_test(size=(100, 30)) as pilot:
            assert str(app.context.render()) == "~1.2k/16k · I"
            assert app.context.region.right == app.composer_shell.content_region.right

            session.context_tokens_estimated = False
            app.update_context()
            assert str(app.context.render()) == "1.2k/16k · I"

            await pilot.resize_terminal(35, 30)
            session.compaction_enabled = False
            app.update_context()
            await pilot.pause()
            assert str(app.context.render()) == "1.2k/16k · off · I"
            assert app.context.region.right == app.composer_shell.content_region.right

            session.context_window_tokens = 0
            app.update_context()
            assert str(app.context.render()) == "I"
            app.exit()

    asyncio.run(scenario())


def test_textual_compaction_message_shows_before_and_after_tokens() -> None:
    async def scenario() -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            app.presenter.history_compacted(12_000, 4_200)
            await pilot.pause()

            messages = list(app.query(".system-message"))
            assert str(messages[-1].render()) == ("Context compacted: 12k → ~4.2k.")
            app.exit()

    asyncio.run(scenario())


def test_textual_parser_recognizes_command_c() -> None:
    events = list(XTermParser().feed("\x1b[99;9u"))

    assert len(events) == 1
    assert events[0].key == "super+c"  # type: ignore[attr-defined]


def test_textual_app_supports_legacy_and_ui_neutral_constructors() -> None:
    app_info = AppInfo("/workspace", "provider", "model", "read-only")
    console = object()
    legacy_session = FakeSession()
    legacy_reviews = FakeReviews()

    legacy = GhostwheelApp(
        console,
        legacy_session,
        legacy_reviews,  # type: ignore[arg-type]
        app_info=app_info,
    )
    neutral_session = FakeSession()
    neutral_reviews = FakeReviews()
    neutral = GhostwheelApp(
        session=neutral_session,
        reviews=neutral_reviews,  # type: ignore[arg-type]
        app_info=app_info,
    )

    assert legacy._console is console
    assert legacy.session is legacy_session
    assert legacy.reviews is legacy_reviews
    assert neutral._console is None
    assert neutral.session is neutral_session
    assert neutral.reviews is neutral_reviews


def test_direct_exit_waits_for_active_turn_cancellation_cleanup() -> None:
    class DelayedCleanupSession(FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.cleaned = asyncio.Event()

        async def send(self, prompt: str) -> TurnNoResult:
            self.sent.append(prompt)
            self.sent_event.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0.05)
                self.cleaned.set()
            return TurnNoResult()

    async def scenario() -> None:
        session = DelayedCleanupSession()
        app = make_app(session)

        async def exit_during_turn(pilot: Pilot[None]) -> None:
            await pilot.pause()
            app.input_reader.submit("wait")
            await asyncio.wait_for(session.sent_event.wait(), 1)
            app.exit()

        await app.run_async(headless=True, auto_pilot=exit_during_turn)

        assert session.cleaned.is_set()

    asyncio.run(scenario())


def test_command_worker_failure_is_raised_after_driver_cleanup(monkeypatch) -> None:
    class FailingSession(FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.error = RuntimeError("command loop failed")

        async def send(self, prompt: str) -> TurnSucceeded[str]:
            self.sent.append(prompt)
            self.sent_event.set()
            raise self.error

    async def scenario() -> None:
        session = FailingSession()
        app = make_app(session)
        driver_closed = asyncio.Event()
        original_close = HeadlessDriver.close

        def close(driver: HeadlessDriver) -> None:
            original_close(driver)
            driver_closed.set()

        monkeypatch.setattr(HeadlessDriver, "close", close)

        async def trigger_failure(_pilot: Pilot[None]) -> None:
            app.input_reader.submit("fail")
            await asyncio.wait_for(session.sent_event.wait(), 1)

        with pytest.raises(WorkerFailed) as exc_info:
            await app.run_async(headless=True, auto_pilot=trigger_failure)

        assert driver_closed.is_set()
        assert exc_info.value.error is session.error

    asyncio.run(scenario())


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


def test_transcript_mouse_wheel_does_not_navigate_prompt_history() -> None:
    async def scenario() -> None:
        app = make_app(vim_mode=False)
        app.history.append("older prompt")
        app.composer._history_index = len(app.history.entries)
        async with app.run_test(size=(60, 20)) as pilot:
            app.transcript.mount(
                Static(
                    Text("\n".join(f"transcript line {index}" for index in range(100)))
                )
            )
            app.composer.load_text("draft prompt")
            await pilot.pause()
            app.transcript.scroll_end(animate=False, immediate=True)
            await pilot.pause()

            initial_scroll = app.transcript.scroll_y
            initial_history_index = app.composer._history_index
            assert initial_scroll == app.transcript.max_scroll_y

            await pilot._post_mouse_events(
                [events.MouseScrollUp],
                app.transcript,
                offset=(10, 5),
            )
            await pilot.pause()

            assert app.transcript.scroll_y < initial_scroll
            assert app.composer.text == "draft prompt"
            assert app.composer._history_index == initial_history_index

            await pilot._post_mouse_events(
                [events.MouseScrollDown],
                app.transcript,
                offset=(10, 5),
            )
            await pilot.pause()
            assert app.transcript.scroll_y == initial_scroll
            assert app.composer.text == "draft prompt"
            app.exit()

    asyncio.run(scenario())


def test_transcript_scrollbar_track_and_thumb_are_mouse_operable() -> None:
    async def scenario() -> None:
        app = make_app(vim_mode=False)
        async with app.run_test(size=(60, 20)) as pilot:
            app.transcript.mount(
                Static(
                    Text("\n".join(f"transcript line {index}" for index in range(100)))
                )
            )
            await pilot.pause()
            scrollbar = app.transcript.vertical_scrollbar
            app.transcript.scroll_home(animate=False, immediate=True)
            await pilot.pause()

            assert scrollbar.display is True
            assert scrollbar.window_virtual_size > scrollbar.window_size
            assert await pilot.click(
                scrollbar,
                offset=(0, scrollbar.region.height - 1),
            )
            await pilot.pause()
            assert app.transcript.scroll_y > 0
            assert app.screen.get_selected_text() is None

            app.transcript.scroll_home(animate=False, immediate=True)
            await pilot.pause()
            assert await pilot.mouse_down(scrollbar, offset=(0, 0))
            assert app.mouse_captured is scrollbar
            assert await pilot._post_mouse_events(
                [events.MouseMove],
                scrollbar,
                offset=(0, scrollbar.region.height // 2),
                button=1,
            )
            await pilot.pause()
            assert app.transcript.scroll_y > 0
            assert await pilot.mouse_up(
                scrollbar,
                offset=(0, scrollbar.region.height // 2),
            )
            await pilot.pause()
            assert app.mouse_captured is None
            assert app.screen.get_selected_text() is None
            app.exit()

    asyncio.run(scenario())


def test_command_c_copies_mouse_selected_transcript_text() -> None:
    async def scenario() -> None:
        app = make_app(vim_mode=False)
        async with app.run_test(size=(60, 20)) as pilot:
            app.presenter.turn_started()
            app.presenter.turn_outcome(
                TurnSucceeded("selectable **transcript** text", ())
            )
            await pilot.pause()
            message = app.query_one(TurnView).answer

            assert await pilot.mouse_down(message, offset=(0, 0))
            assert await pilot._post_mouse_events(
                [events.MouseMove],
                message,
                offset=(10, 0),
                button=1,
            )
            assert await pilot.mouse_up(message, offset=(10, 0))
            await pilot.pause()

            assert app.screen.get_selected_text() == "selectable "
            selected_segment, unselected_segment = list(message.render_line(0))[:2]
            assert selected_segment.style is not None
            assert unselected_segment.style is not None
            assert selected_segment.style.bgcolor != unselected_segment.style.bgcolor
            assert selected_segment.style.color != selected_segment.style.bgcolor
            await pilot.press("super+c")
            assert app.clipboard == "selectable "
            app.exit()

    asyncio.run(scenario())


def test_right_click_copies_and_preserves_transcript_selection() -> None:
    async def scenario() -> None:
        app = make_app(vim_mode=False)
        async with app.run_test(size=(60, 20)) as pilot:
            app.presenter.turn_started()
            app.presenter.turn_outcome(TurnSucceeded("selectable transcript text", ()))
            await pilot.pause()
            message = app.query_one(TurnView).answer

            assert await pilot.mouse_down(message, offset=(0, 0))
            assert await pilot._post_mouse_events(
                [events.MouseMove],
                message,
                offset=(10, 0),
                button=1,
            )
            assert await pilot.mouse_up(message, offset=(10, 0))
            await pilot.pause()

            await right_click(app, pilot, message)

            assert app.clipboard == "selectable "
            assert app.screen.get_selected_text() == "selectable "
            app.exit()

    asyncio.run(scenario())


def test_right_click_copies_and_preserves_composer_selection() -> None:
    async def scenario() -> None:
        app = make_app(vim_mode=False)
        async with app.run_test(size=(60, 20)) as pilot:
            app.composer.load_text("copy this")
            app.composer.move_cursor((0, 0))
            app.composer.move_cursor((0, 4), select=True)

            await right_click(app, pilot, app.composer)

            assert app.clipboard == "copy"
            assert app.composer.selected_text == "copy"
            app.exit()

    asyncio.run(scenario())


def test_right_click_without_selection_does_not_copy() -> None:
    async def scenario() -> None:
        app = make_app(vim_mode=False)
        async with app.run_test(size=(60, 20)) as pilot:
            app.copy_to_clipboard("unchanged")

            await right_click(app, pilot, app.composer)

            assert app.clipboard == "unchanged"
            app.exit()

    asyncio.run(scenario())


def test_command_c_without_selection_does_not_quit() -> None:
    async def scenario() -> None:
        app = make_app(vim_mode=False)
        async with app.run_test(size=(60, 20)) as pilot:
            await pilot.press("super+c")

            assert app.is_running is True
            assert app.input_reader._queue.empty()
            app.exit()

    asyncio.run(scenario())


def test_ctrl_c_copies_composer_selection_before_cancel_or_quit() -> None:
    async def scenario() -> None:
        app = make_app(vim_mode=False)
        async with app.run_test(size=(60, 20)) as pilot:
            app.composer.load_text("copy this")
            app.composer.move_cursor((0, 0))
            app.composer.move_cursor((0, 4), select=True)

            await pilot.press("ctrl+c")

            assert app.clipboard == "copy"
            assert app.is_running is True
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
