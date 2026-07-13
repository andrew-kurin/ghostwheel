from __future__ import annotations

import asyncio
import os

import pytest
from prompt_toolkit.input import DummyInput
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

from ghostwheel.terminal_io import ActiveTurnInputMonitor, RedirectedLineReader


class _Cancellation:
    def cancel(self) -> bool:
        return True


class _TerminalGuard:
    def restore(self) -> None:
        pass


def _key_data(keys: list[KeyPress]) -> list[str]:
    return [key.data for key in keys]


def test_active_turn_discards_bytes_unread_at_teardown() -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe_input:
            monitor = ActiveTurnInputMonitor(
                get_input=lambda: pipe_input,
                get_timeout=lambda _input: 0.01,
                terminal_guard=_TerminalGuard(),  # type: ignore[arg-type]
            )

            with monitor.capture(_Cancellation()):
                # Do not yield to the registered input callback before the
                # capture closes; teardown itself must drain these bytes.
                pipe_input.send_bytes(b"ACTIVE")

            pipe_input.send_bytes(b"NEXT\r")
            keys = pipe_input.read_keys() + pipe_input.flush_keys()

        assert _key_data(keys) == ["N", "E", "X", "T", "\r"]

    asyncio.run(scenario())


def test_active_turn_accepts_input_without_pollable_descriptor() -> None:
    async def scenario() -> None:
        prompt_input = DummyInput()
        monitor = ActiveTurnInputMonitor(
            get_input=lambda: prompt_input,
            get_timeout=lambda _input: 0.01,
            terminal_guard=_TerminalGuard(),  # type: ignore[arg-type]
        )

        with monitor.capture(_Cancellation()):
            pass

    asyncio.run(scenario())


def test_active_turn_discards_incomplete_bracketed_paste() -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe_input:
            monitor = ActiveTurnInputMonitor(
                get_input=lambda: pipe_input,
                get_timeout=lambda _input: 0.01,
                terminal_guard=_TerminalGuard(),  # type: ignore[arg-type]
            )

            with monitor.capture(_Cancellation()):
                pipe_input.send_bytes(b"\x1b[200~ACTIVE")
                await asyncio.sleep(0.05)

            pipe_input.send_bytes(b"\x1b[200~NEXT\x1b[201~\r")
            keys = pipe_input.read_keys() + pipe_input.flush_keys()

        assert [(key.key, key.data) for key in keys] == [
            (Keys.BracketedPaste, "NEXT"),
            (Keys.ControlM, "\r"),
        ]

    asyncio.run(scenario())


def test_active_turn_discards_incomplete_multibyte_character() -> None:
    async def scenario() -> None:
        with create_pipe_input() as pipe_input:
            monitor = ActiveTurnInputMonitor(
                get_input=lambda: pipe_input,
                get_timeout=lambda _input: 0.01,
                terminal_guard=_TerminalGuard(),  # type: ignore[arg-type]
            )

            with monitor.capture(_Cancellation()):
                pipe_input.send_bytes(b"\xc3")
                await asyncio.sleep(0.05)

            pipe_input.send_bytes(b"NEXT\r")
            keys = pipe_input.read_keys() + pipe_input.flush_keys()

        assert _key_data(keys) == ["N", "E", "X", "T", "\r"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("encoding", "payload", "expected"),
    [
        (
            "utf-16-le",
            "café\nsecond\n".encode("utf-16-le"),
            ("café", "second"),
        ),
        (
            "iso2022_jp",
            b"\x1b$BF|K\\\n8l\x1b(B\n",
            ("日本", "語"),
        ),
    ],
)
def test_redirected_pipe_decodes_before_splitting_lines(
    encoding: str,
    payload: bytes,
    expected: tuple[str, str],
) -> None:
    read_descriptor, write_descriptor = os.pipe()
    input_stream = os.fdopen(read_descriptor, "r", encoding=encoding)
    reader = RedirectedLineReader(input_stream)

    async def scenario() -> None:
        os.write(write_descriptor, payload)
        assert await reader.read() == expected[0]
        assert await reader.read() == expected[1]
        os.close(write_descriptor)
        with pytest.raises(EOFError):
            await reader.read()

    try:
        asyncio.run(scenario())
    finally:
        input_stream.close()
        try:
            os.close(write_descriptor)
        except OSError:
            pass
