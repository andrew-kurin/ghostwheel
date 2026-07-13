"""Terminal input lifecycle helpers for the inline UI.

These collaborators isolate the platform-sensitive parts of terminal input:
raw-mode restoration, active-turn key monitoring, and redirected line reads.
"""

from __future__ import annotations

import asyncio
import os
import signal
import stat
import termios
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from types import FrameType
from typing import Protocol, TextIO

from prompt_toolkit.input import Input
from prompt_toolkit.input.typeahead import get_typeahead
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys

_TERMIOS_LOCAL_FLAGS = 3
_TERMIOS_CONTROL_CHARACTERS = 6
_TERMINAL_GUARDED_SIGNALS = (
    signal.SIGHUP,
    signal.SIGQUIT,
    signal.SIGTERM,
    signal.SIGTSTP,
)


class Cancellation(Protocol):
    """Minimal cancellation behavior required by active-turn monitoring."""

    def cancel(self) -> bool: ...


class RawTerminalGuard:
    """Own raw tty state and restore it across process signals."""

    def __init__(
        self,
        input_stream: TextIO,
        *,
        externally_managed_input: bool,
        is_active: Callable[[], bool],
        flush_input_on_restore: bool = True,
    ) -> None:
        self._input_stream = input_stream
        self._externally_managed_input = externally_managed_input
        self._is_active = is_active
        self._flush_input_on_restore = flush_input_on_restore
        self._descriptor: int | None = None
        self._attributes: list[object] | None = None
        self._signal_handlers: dict[int, object] = {}

    def silence(self) -> None:
        """Disable echo, canonical input, and signals during an active turn."""

        if self._externally_managed_input or self._attributes is not None:
            return
        terminal = self._terminal_attributes()
        if terminal is None:
            return
        descriptor, attributes = terminal
        quiet_attributes = attributes.copy()
        quiet_attributes[_TERMIOS_LOCAL_FLAGS] &= ~(
            termios.ECHO | termios.ECHONL | termios.ICANON | termios.ISIG
        )
        if not self._install_signal_handlers():
            return
        self._descriptor = descriptor
        self._attributes = attributes
        try:
            termios.tcsetattr(descriptor, termios.TCSANOW, quiet_attributes)
        except OSError, termios.error:
            self.restore()

    def guard_prompt(self) -> None:
        """Remember cooked tty state while prompt-toolkit temporarily owns it."""

        if self._externally_managed_input or self._attributes is not None:
            return
        terminal = self._terminal_attributes()
        if terminal is None:
            return
        descriptor, attributes = terminal
        if not self._install_signal_handlers():
            return
        self._descriptor = descriptor
        self._attributes = attributes

    def clear_local_flags(self, flags: int) -> bool:
        """Temporarily clear tty local flags with signal-safe restoration."""

        if self._externally_managed_input or self._attributes is not None:
            return False
        terminal = self._terminal_attributes()
        if terminal is None:
            return False
        descriptor, attributes = terminal
        updated_attributes = attributes.copy()
        updated_attributes[_TERMIOS_LOCAL_FLAGS] &= ~flags
        if updated_attributes == attributes:
            return True
        if not self._install_signal_handlers():
            return False
        self._descriptor = descriptor
        self._attributes = attributes
        try:
            termios.tcsetattr(descriptor, termios.TCSANOW, updated_attributes)
        except OSError, termios.error:
            self.restore()
            return False
        return True

    def restore(self) -> None:
        """Restore saved tty attributes and process signal handlers."""

        descriptor = self._descriptor
        attributes = self._attributes
        self._descriptor = None
        self._attributes = None
        if descriptor is not None and attributes is not None:
            if self._flush_input_on_restore:
                try:
                    termios.tcflush(descriptor, termios.TCIFLUSH)
                except OSError, termios.error:
                    pass
            try:
                termios.tcsetattr(descriptor, termios.TCSANOW, attributes)
            except OSError, termios.error:
                pass
        self._restore_signal_handlers()

    def _terminal_attributes(self) -> tuple[int, list[object]] | None:
        try:
            descriptor = self._input_stream.fileno()
            if not os.isatty(descriptor):
                return None
            return descriptor, termios.tcgetattr(descriptor)
        except AttributeError, OSError, termios.error:
            return None

    def _install_signal_handlers(self) -> bool:
        if self._signal_handlers:
            return True

        installed: dict[int, object] = {}
        try:
            for signum in _TERMINAL_GUARDED_SIGNALS:
                installed[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle_signal)
        except OSError, RuntimeError, ValueError:
            for signum, previous in installed.items():
                try:
                    signal.signal(signum, previous)
                except OSError, RuntimeError, ValueError:
                    pass
            return False
        self._signal_handlers = installed
        return True

    def _restore_signal_handlers(self) -> None:
        handlers = self._signal_handlers
        self._signal_handlers = {}
        for signum, previous in handlers.items():
            try:
                signal.signal(signum, previous)
            except OSError, RuntimeError, ValueError:
                pass

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        """Restore the tty before honoring termination or job-control signals."""

        previous = self._signal_handlers.get(signum, signal.SIG_DFL)
        descriptor = self._descriptor
        original_attributes = self._attributes
        active_attributes: list[object] | None = None
        if descriptor is not None:
            try:
                active_attributes = termios.tcgetattr(descriptor)
            except OSError, termios.error:
                pass
        self.restore()

        if previous == signal.SIG_IGN:
            self._resume(descriptor, original_attributes, active_attributes)
            return
        if previous == signal.SIG_DFL or previous is None:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            # SIGTSTP returns here after SIGCONT; termination signals do not.
            self._resume(descriptor, original_attributes, active_attributes)
            return
        if callable(previous):
            previous(signum, frame)
            self._resume(descriptor, original_attributes, active_attributes)

    def _resume(
        self,
        descriptor: int | None,
        original_attributes: list[object] | None,
        active_attributes: list[object] | None,
    ) -> None:
        if (
            not self._is_active()
            or descriptor is None
            or original_attributes is None
            or active_attributes is None
            or not self._install_signal_handlers()
        ):
            return
        self._descriptor = descriptor
        self._attributes = original_attributes
        try:
            termios.tcsetattr(descriptor, termios.TCSANOW, active_attributes)
        except OSError, termios.error:
            self.restore()


class ActiveTurnInputMonitor:
    """Drain active-turn input and translate Esc/Ctrl+D into cancellation."""

    def __init__(
        self,
        *,
        get_input: Callable[[], Input | None],
        get_timeout: Callable[[Input], float],
        terminal_guard: RawTerminalGuard,
    ) -> None:
        self._get_input = get_input
        self._get_timeout = get_timeout
        self._terminal_guard = terminal_guard
        self.quit_requested = False

    @contextmanager
    def capture(self, cancellation: Cancellation) -> Iterator[None]:
        """Monitor keys for the lifetime of one cancellable active turn."""

        prompt_input = self._get_input()
        if prompt_input is None:
            # The turn may already have disabled canonical input and ISIG.
            # Restore a normal tty if monitoring cannot be attached.
            self._terminal_guard.restore()
            yield
            return

        loop = asyncio.get_running_loop()
        flush_handle: asyncio.TimerHandle | None = None
        monitor_active = True
        cancellation_requested = False

        def cancel_if_active() -> None:
            nonlocal cancellation_requested
            if monitor_active and not cancellation.cancel():
                cancellation_requested = False

        def request_cancellation() -> None:
            nonlocal cancellation_requested
            if cancellation_requested:
                return
            cancellation_requested = True
            # The controller starts presentation immediately before its task
            # is cancellable. Defer to avoid losing same-chunk input.
            loop.call_soon(cancel_if_active)

        def handle_keys(keys: Iterable[KeyPress]) -> None:
            key_presses = tuple(keys)
            for key_press in key_presses:
                if key_press.key == Keys.ControlD:
                    self.quit_requested = True
                    request_cancellation()
                    return
            # Meta combinations decode as Escape followed by the modified key.
            # Only a trailing Escape represents a standalone cancel request.
            if key_presses and key_presses[-1].key == Keys.Escape:
                request_cancellation()

        def flush_pending_escape() -> None:
            nonlocal flush_handle
            flush_handle = None
            if not monitor_active:
                return
            handle_keys(prompt_input.flush_keys())

        def input_ready() -> None:
            nonlocal flush_handle
            # attach() can have queued this callback before its context was
            # detached. Never consume keys or mutate state after the turn.
            if not monitor_active:
                return
            if flush_handle is not None:
                flush_handle.cancel()
                flush_handle = None
            keys = prompt_input.read_keys()
            handle_keys(keys)
            if prompt_input.closed:
                self.quit_requested = True
                request_cancellation()
            elif monitor_active:
                flush_handle = loop.call_later(
                    self._get_timeout(prompt_input),
                    flush_pending_escape,
                )

        stack = ExitStack()
        try:
            stack.enter_context(prompt_input.raw_mode())
            stack.enter_context(prompt_input.attach(input_ready))
        except EOFError, NotImplementedError, OSError, RuntimeError:
            stack.close()
            self._terminal_guard.restore()
            yield
            return

        try:
            # Retained typeahead can include keys read with prompt submission.
            # Defer a pending VT prefix so split arrows/Meta sequences finish.
            handle_keys(get_typeahead(prompt_input))
            flush_handle = loop.call_later(
                self._get_timeout(prompt_input),
                flush_pending_escape,
            )
            yield
        finally:
            monitor_active = False
            if flush_handle is not None:
                flush_handle.cancel()
            stack.close()
            prompt_input.flush_keys()
            get_typeahead(prompt_input)


class RedirectedLineReader:
    """Read stream or cooked-terminal lines without worker-thread leakage.

    A fallback TTY deliberately remains in canonical mode so the kernel keeps
    providing the user's normal line editing.  Its EOF delimiter is handled
    as an unconditional quit, and its interrupt character clears the pending
    kernel line without propagating ``KeyboardInterrupt``.
    """

    def __init__(self, input_stream: TextIO) -> None:
        self._input_stream = input_stream
        self._buffer = bytearray()
        self._eof = False
        self._terminal_read_active = False
        self._terminal_guard = RawTerminalGuard(
            input_stream,
            externally_managed_input=False,
            is_active=lambda: self._terminal_read_active,
            flush_input_on_restore=False,
        )

    async def read(self) -> str:
        try:
            descriptor = self._input_stream.fileno()
            descriptor_mode = os.fstat(descriptor).st_mode
        except AttributeError, OSError, TypeError, ValueError:
            return self._read_synchronously()
        if stat.S_ISREG(descriptor_mode):
            return self._read_synchronously()
        if os.isatty(descriptor):
            return await self._read_terminal_line(descriptor)

        while b"\n" not in self._buffer and not self._eof:
            chunk = await self._read_ready_chunk(descriptor)
            if chunk:
                self._buffer.extend(chunk)
            else:
                self._eof = True

        newline = self._buffer.find(b"\n")
        if newline >= 0:
            raw_value = bytes(self._buffer[: newline + 1])
            del self._buffer[: newline + 1]
        elif self._buffer:
            raw_value = bytes(self._buffer)
            self._buffer.clear()
        else:
            raise EOFError

        encoding = getattr(self._input_stream, "encoding", None) or "utf-8"
        errors = getattr(self._input_stream, "errors", None) or "strict"
        return raw_value.decode(encoding, errors).rstrip("\r\n")

    async def _read_terminal_line(self, descriptor: int) -> str:
        self._terminal_read_active = True
        try:
            if not self._terminal_guard.clear_local_flags(termios.NOFLSH):
                raise RuntimeError("fallback terminal input could not guard tty state")
            line_delimiters = self._terminal_line_delimiters(descriptor)
            while True:
                try:
                    raw_value = await self._read_terminal_chunk(descriptor)
                except _TerminalLineCleared:
                    continue

                # In canonical mode a read which does not end in a line delimiter
                # was released by VEOF.  Discard any pending draft so one Ctrl+D
                # always means quit, even when the kernel line buffer was nonempty.
                delimiter = next(
                    (
                        candidate
                        for candidate in line_delimiters
                        if raw_value.endswith(candidate)
                    ),
                    None,
                )
                if delimiter is None:
                    raise EOFError

                encoding = getattr(self._input_stream, "encoding", None) or "utf-8"
                errors = getattr(self._input_stream, "errors", None) or "strict"
                return raw_value[: -len(delimiter)].decode(encoding, errors)
        finally:
            self._terminal_read_active = False
            self._terminal_guard.restore()

    @staticmethod
    def _terminal_line_delimiters(descriptor: int) -> tuple[bytes, ...]:
        """Return enabled delimiters which release a canonical terminal read."""

        delimiters = [b"\n"]
        try:
            attributes = termios.tcgetattr(descriptor)
            disabled_character = os.fpathconf(descriptor, "PC_VDISABLE")
        except OSError, ValueError, termios.error:
            return tuple(delimiters)

        control_characters = attributes[_TERMIOS_CONTROL_CHARACTERS]
        for name in ("VEOL", "VEOL2"):
            index = getattr(termios, name, None)
            if index is None:
                continue
            value = control_characters[index]
            if isinstance(value, int):
                character = bytes((value,))
            elif isinstance(value, bytes) and len(value) == 1:
                character = value
            else:
                continue
            if character[0] == disabled_character or character in delimiters:
                continue
            delimiters.append(character)
        return tuple(delimiters)

    async def _read_terminal_chunk(self, descriptor: int) -> bytes:
        loop = asyncio.get_running_loop()
        readable: asyncio.Future[bytes] = loop.create_future()

        def clear_pending_line(_signum: int, _frame: FrameType | None) -> None:
            if not readable.done():
                readable.set_exception(_TerminalLineCleared())

        previous_sigint = signal.getsignal(signal.SIGINT)
        try:
            signal.signal(signal.SIGINT, clear_pending_line)
        except (OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(
                "fallback terminal input must run on the main thread"
            ) from error

        def read_ready() -> None:
            if readable.done():
                return
            try:
                chunk = os.read(descriptor, 65_536)
            except BlockingIOError:
                return
            except OSError as error:
                readable.set_exception(error)
            else:
                readable.set_result(chunk)

        try:
            loop.add_reader(descriptor, read_ready)
        except (NotImplementedError, OSError) as error:
            signal.signal(signal.SIGINT, previous_sigint)
            raise RuntimeError(
                "fallback terminal input requires a pollable POSIX file descriptor"
            ) from error
        try:
            return await readable
        finally:
            loop.remove_reader(descriptor)
            signal.signal(signal.SIGINT, previous_sigint)

    async def _read_ready_chunk(self, descriptor: int) -> bytes:
        loop = asyncio.get_running_loop()
        readable: asyncio.Future[bytes] = loop.create_future()

        def read_ready() -> None:
            if readable.done():
                return
            try:
                chunk = os.read(descriptor, 65_536)
            except BlockingIOError:
                return
            except OSError as error:
                readable.set_exception(error)
            else:
                readable.set_result(chunk)

        try:
            loop.add_reader(descriptor, read_ready)
        except (NotImplementedError, OSError) as error:
            raise RuntimeError(
                "redirected input requires a pollable POSIX file descriptor"
            ) from error
        try:
            return await readable
        finally:
            loop.remove_reader(descriptor)

    def _read_synchronously(self) -> str:
        value = self._input_stream.readline()
        if value == "":
            raise EOFError
        return value.rstrip("\r\n")


class _TerminalLineCleared(Exception):
    """Internal wakeup used when Ctrl+C clears a fallback terminal line."""
