from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.formatted_text import FormattedText
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
def test_shift_enter_inserts_newline_in_every_vim_mode(
    input_mode: InputMode,
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

            pipe_input.send_text("\x1b[27;2;13~")
            await _wait_until(lambda: "\n" in session.default_buffer.text)

            assert session.default_buffer.text == "before\n"
            assert not prompt_task.done()

            session.app.vi_state.input_mode = InputMode.INSERT
            pipe_input.send_text("\r")
            assert await asyncio.wait_for(prompt_task, 1) == "before\n"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "input_mode",
    [
        InputMode.INSERT,
        InputMode.NAVIGATION,
        InputMode.REPLACE,
        InputMode.INSERT_MULTIPLE,
    ],
)
def test_bare_carriage_return_always_submits(input_mode: InputMode) -> None:
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

            pipe_input.send_text("\r")
            assert await asyncio.wait_for(prompt_task, 1) == "before"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "input_mode",
    [
        InputMode.INSERT,
        InputMode.NAVIGATION,
        InputMode.REPLACE,
        InputMode.INSERT_MULTIPLE,
    ],
)
def test_control_j_is_inert_in_every_vim_mode(input_mode: InputMode) -> None:
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

            pipe_input.send_text("\n")
            await asyncio.sleep(0.01)

            assert session.default_buffer.text == "before"
            assert not prompt_task.done()
            pipe_input.send_text("\r")
            assert await asyncio.wait_for(prompt_task, 1) == "before"

    asyncio.run(scenario())


def _composer(history_path: Path | None) -> terminal_composer.TerminalComposer:
    return terminal_composer.TerminalComposer(
        workspace=Path.cwd(),
        history_path=history_path,
        vim_mode=False,
        prompt_input=None,
        prompt_output=None,
        bottom_toolbar=FormattedText,
        rprompt=FormattedText,
    )


def test_history_create_error_falls_back_to_memory_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "blocked" / "input-history"
    attempts = 0

    def deny_create(_history: terminal_composer.InputHistory) -> None:
        nonlocal attempts
        attempts += 1
        raise PermissionError("create denied")

    monkeypatch.setattr(terminal_composer.InputHistory, "_ensure_file", deny_create)
    composer = _composer(history_path)

    history = composer.get_history()

    assert history.store.path is None
    assert history.store.entries == []
    warning = composer.take_warning()
    assert warning == terminal_composer.ComposerWarning(
        code="history_unavailable",
        message="Prompt history is unavailable; using in-memory history.",
        path=history_path,
        detail="create denied",
    )
    assert composer.take_warning() is None

    history.store.append("kept for this session")

    assert history.store.entries == ["kept for this session"]
    assert attempts == 1
    assert composer.take_warning() is None


def test_history_read_error_falls_back_to_memory_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "input-history"
    history_path.touch()
    original_read_text = Path.read_text

    def deny_history_read(path: Path, *args: object, **kwargs: object) -> str:
        if path == history_path:
            raise PermissionError("read denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny_history_read)
    composer = _composer(history_path)

    history = composer.get_history()

    assert history.store.path is None
    assert history.store.entries == []
    warning = composer.take_warning()
    assert warning is not None
    assert warning.code == "history_unavailable"
    assert warning.path == history_path
    assert warning.detail == "read denied"
    assert composer.take_warning() is None
