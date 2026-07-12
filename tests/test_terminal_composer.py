from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.output import DummyOutput

import ghostwheel.terminal_composer as terminal_composer


async def _wait_until(predicate: Callable[[], bool]) -> None:
    for _attempt in range(100):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not reached")


@pytest.mark.parametrize(
    "input_mode",
    [
        InputMode.INSERT,
        InputMode.NAVIGATION,
        InputMode.REPLACE,
        InputMode.INSERT_MULTIPLE,
    ],
)
@pytest.mark.parametrize(
    "shift_enter",
    ["\n", "\x1b[27;2;13~"],
    ids=["mapped-lf", "xterm-modified-cr"],
)
def test_shift_enter_encodings_insert_newline_in_every_vim_mode(
    input_mode: InputMode,
    shift_enter: str,
) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe_input:
            session: PromptSession[str] = PromptSession(
                multiline=True,
                editing_mode=EditingMode.VI,
                key_bindings=terminal_composer._key_bindings(),
                input=pipe_input,
                output=DummyOutput(),
            )
            prompt_task = asyncio.create_task(session.prompt_async())
            await _wait_until(lambda: session.app.is_running)
            session.default_buffer.insert_text("before")
            session.app.vi_state.input_mode = input_mode

            pipe_input.send_text(shift_enter)
            await _wait_until(lambda: "\n" in session.default_buffer.text)

            assert session.default_buffer.text == "before\n"
            assert not prompt_task.done()

            session.app.vi_state.input_mode = InputMode.INSERT
            pipe_input.send_text("\r")
            assert await asyncio.wait_for(prompt_task, 1) == "before\n"

    asyncio.run(scenario())


def test_macos_shift_state_recovers_bare_enter(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe_input:
            session: PromptSession[str] = PromptSession(
                multiline=True,
                key_bindings=terminal_composer._key_bindings(),
                input=pipe_input,
                output=DummyOutput(),
            )
            prompt_task = asyncio.create_task(session.prompt_async())
            await _wait_until(lambda: session.app.is_running)
            session.default_buffer.insert_text("before")
            monkeypatch.setattr(
                terminal_composer,
                "_macos_shift_pressed",
                lambda: True,
            )

            pipe_input.send_text("\r")
            await _wait_until(lambda: "\n" in session.default_buffer.text)

            assert session.default_buffer.text == "before\n"
            assert not prompt_task.done()

            monkeypatch.setattr(
                terminal_composer,
                "_macos_shift_pressed",
                lambda: False,
            )
            pipe_input.send_text("\r")
            assert await asyncio.wait_for(prompt_task, 1) == "before\n"

    asyncio.run(scenario())


class _FakeFlagsReader:
    argtypes = None
    restype = None

    def __call__(self, _state: int) -> int:
        return terminal_composer._MACOS_SHIFT_FLAG


class _FakeCoreGraphics:
    CGEventSourceFlagsState = _FakeFlagsReader()


def test_macos_shift_state_reads_core_graphics_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal_composer._macos_flags_reader.cache_clear()
    monkeypatch.setattr(terminal_composer.sys, "platform", "darwin")
    monkeypatch.setattr(
        terminal_composer.ctypes,
        "CDLL",
        lambda _path: _FakeCoreGraphics(),
    )

    assert terminal_composer._macos_shift_pressed() is True
    terminal_composer._macos_flags_reader.cache_clear()
