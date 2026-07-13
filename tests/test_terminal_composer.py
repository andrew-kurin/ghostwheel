from __future__ import annotations

import asyncio
import os
import stat
import subprocess
import sys
from collections.abc import Callable
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.output import DummyOutput

import ghostwheel.terminal_composer as terminal_composer


@pytest.mark.parametrize(
    "state_home",
    [None, "", "relative-state", "~/.state"],
)
def test_default_history_path_ignores_missing_or_relative_xdg_state_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    state_home: str | None,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    if state_home is None:
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    else:
        monkeypatch.setenv("XDG_STATE_HOME", state_home)

    assert terminal_composer.default_history_path() == (
        home / ".local/state/ghostwheel/input-history"
    )


def test_default_history_path_accepts_absolute_xdg_state_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    state_home = "/var/tmp/ghostwheel-state"
    monkeypatch.setenv("XDG_STATE_HOME", state_home)

    assert terminal_composer.default_history_path() == (
        Path(state_home) / "ghostwheel/input-history"
    )


def test_default_history_path_disables_persistence_without_absolute_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("HOME", "relative-home")

    assert terminal_composer.default_history_path() is None


async def _wait_until(predicate: Callable[[], bool]) -> None:
    for _attempt in range(100):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not reached")


@pytest.mark.parametrize(
    ("editing_mode", "input_mode"),
    [
        (EditingMode.EMACS, None),
        (EditingMode.VI, InputMode.INSERT),
        (EditingMode.VI, InputMode.NAVIGATION),
        (EditingMode.VI, InputMode.REPLACE),
        (EditingMode.VI, InputMode.INSERT_MULTIPLE),
    ],
)
@pytest.mark.parametrize(
    "sequence",
    [
        "\x1b[27;2;13~",
        "\x1b[13;2u",
    ],
    ids=["legacy-xterm", "csi-u"],
)
def test_shift_enter_inserts_newline_in_emacs_and_every_vim_mode(
    editing_mode: EditingMode,
    input_mode: InputMode | None,
    sequence: str,
) -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe_input:
            session: PromptSession[str] = PromptSession(
                multiline=True,
                editing_mode=editing_mode,
                key_bindings=terminal_composer._key_bindings(),
                input=pipe_input,
                output=DummyOutput(),
            )
            prompt_task = asyncio.create_task(session.prompt_async())
            await _wait_until(lambda: session.app.is_running)
            session.default_buffer.insert_text("before")
            if input_mode is not None:
                session.app.vi_state.input_mode = input_mode

            pipe_input.send_text(sequence)
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
        prompt_message=lambda: FormattedText([("class:prompt", "> ")]),
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


def test_history_expanduser_error_falls_back_to_memory_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_path = Path("~/input-history")

    def fail_expansion(path: Path) -> Path:
        assert path == history_path
        raise RuntimeError("home directory unavailable")

    monkeypatch.setattr(Path, "expanduser", fail_expansion)
    composer = _composer(history_path)

    history = composer.get_history()

    assert history.store.path is None
    assert history.store.entries == []
    warning = composer.take_warning()
    assert warning is not None
    assert warning.code == "history_unavailable"
    assert warning.path == history_path
    assert warning.detail == "home directory unavailable"
    assert composer.take_warning() is None

    history.store.append("kept for this session")

    assert history.store.entries == ["kept for this session"]
    assert composer.take_warning() is None


def test_history_read_error_falls_back_to_memory_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "input-history"
    history_path.touch()
    original_open = os.open

    def deny_history_read(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
    ) -> int:
        if Path(path) == history_path and not flags & (os.O_WRONLY | os.O_RDWR):
            raise PermissionError("read denied")
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", deny_history_read)
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


def test_history_fifo_falls_back_without_blocking(tmp_path: Path) -> None:
    history_path = tmp_path / "input-history"
    os.mkfifo(history_path)
    script = """
from pathlib import Path
from ghostwheel.terminal_composer import InputHistory

errors = []
history = InputHistory(Path(__import__('sys').argv[1]), on_error=lambda path, error: errors.append(error))
assert history.path is None
assert len(errors) == 1
assert 'not a regular file' in str(errors[0])
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(history_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=2,
    )

    assert result.returncode == 0, result.stderr


def test_history_symlink_does_not_modify_target(tmp_path: Path) -> None:
    target = tmp_path / "unrelated"
    target.write_text("original", encoding="utf-8")
    target.chmod(0o644)
    history_path = tmp_path / "input-history"
    history_path.symlink_to(target)
    errors: list[OSError | RuntimeError | UnicodeError] = []

    history = terminal_composer.InputHistory(
        history_path,
        on_error=lambda path, error: errors.append(error),
    )
    history.append("kept only in memory")

    assert history.path is None
    assert history.entries == ["kept only in memory"]
    assert len(errors) == 1
    assert target.read_text(encoding="utf-8") == "original"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_history_append_rejects_symlink_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "input-history"
    target = tmp_path / "unrelated"
    target.write_text("original", encoding="utf-8")
    target.chmod(0o644)
    errors: list[OSError | RuntimeError | UnicodeError] = []
    history = terminal_composer.InputHistory(
        history_path,
        on_error=lambda path, error: errors.append(error),
    )
    original_open = os.open
    swapped = False

    def swap_before_open(
        path: os.PathLike[str] | str,
        flags: int,
        mode: int = 0o777,
    ) -> int:
        nonlocal swapped
        if Path(path) == history_path and not swapped:
            swapped = True
            history_path.unlink()
            history_path.symlink_to(target)
        return original_open(path, flags, mode)

    monkeypatch.setattr(os, "open", swap_before_open)

    history.append("kept only in memory")

    assert swapped
    assert history.path is None
    assert history.entries == ["kept only in memory"]
    assert len(errors) == 1
    assert target.read_text(encoding="utf-8") == "original"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_history_round_trips_surrogateescaped_terminal_bytes(tmp_path: Path) -> None:
    history_path = tmp_path / "input-history"
    value = "review src/bad-\udcff-name.py"

    terminal_composer.InputHistory(history_path).append(value)

    assert b"bad-\xff-name.py" in history_path.read_bytes()
    assert terminal_composer.InputHistory(history_path).entries == [value]


@pytest.mark.parametrize(
    "separator",
    [
        "\r",
        "\v",
        "\f",
        "\x85",
        "\u2028",
        "\u2029",
    ],
    ids=[
        "carriage-return",
        "vertical-tab",
        "form-feed",
        "next-line",
        "line-separator",
        "paragraph-separator",
    ],
)
def test_history_restart_preserves_non_lf_line_separators(
    tmp_path: Path,
    separator: str,
) -> None:
    history_path = tmp_path / "input-history"
    value = f"before{separator}after\nsecond line"

    terminal_composer.InputHistory(history_path).append(value)

    assert terminal_composer.InputHistory(history_path).entries == [value]


def test_prompt_session_persists_surrogateescaped_terminal_bytes(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        history_path = tmp_path / "input-history"
        with create_pipe_input() as pipe_input:
            composer = terminal_composer.TerminalComposer(
                workspace=tmp_path,
                history_path=history_path,
                vim_mode=False,
                prompt_input=pipe_input,
                prompt_output=DummyOutput(),
                prompt_message=lambda: FormattedText([("class:prompt", "> ")]),
                bottom_toolbar=lambda: FormattedText(),
                rprompt=lambda: FormattedText(),
            )
            session = composer.get_session(StringIO())
            prompt_task = asyncio.create_task(session.prompt_async())
            await _wait_until(lambda: session.app.is_running)

            pipe_input.send_bytes(b"review src/bad-\xff-name.py\r")
            value = await asyncio.wait_for(prompt_task, 1)

            assert value == "review src/bad-\udcff-name.py"
            assert composer.get_history().store.entries == [value]
            assert composer.take_warning() is None

        assert terminal_composer.InputHistory(history_path).entries == [value]

    asyncio.run(scenario())


def test_unencodable_history_value_falls_back_to_memory_once(
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "input-history"
    composer = _composer(history_path)
    history = composer.get_history()
    history.store.append("existing prompt")
    persisted_before_failure = history_path.read_bytes()

    history.store.append("valid first line\nunsupported surrogate: \ud800")

    assert history.store.path is None
    assert history.store.entries == [
        "existing prompt",
        "valid first line\nunsupported surrogate: \ud800",
    ]
    assert history_path.read_bytes() == persisted_before_failure
    assert terminal_composer.InputHistory(history_path).entries == ["existing prompt"]
    warning = composer.take_warning()
    assert warning is not None
    assert warning.code == "history_unavailable"
    assert warning.path == history_path
    assert "surrogates not allowed" in (warning.detail or "")
    assert composer.take_warning() is None

    history.store.append("kept for this session")

    assert history.store.entries[-1] == "kept for this session"
    assert composer.take_warning() is None
