"""Cancellation lifecycle for one in-flight Ghostwheel turn."""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Awaitable
from typing import TypeVar


class _Cancelled:
    """Sentinel returned when the active turn is cancelled."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "CANCELLED"


CANCELLED = _Cancelled()
"""The result of a turn cancelled through :class:`TurnCancellation`."""

ResultT = TypeVar("ResultT")


class TurnCancellation:
    """Own cancellation and optional SIGINT handling for one active turn.

    A controller is reusable across turns, but only one call to :meth:`run` may
    be active at a time.  Cancelling the controller returns ``CANCELLED`` from
    that call; cancelling the task which called ``run`` still propagates an
    ``asyncio.CancelledError`` to its caller.
    """

    def __init__(self, *, handle_sigint: bool = False) -> None:
        self._handle_sigint = handle_sigint
        self._task: asyncio.Future[object] | None = None

    @property
    def active(self) -> bool:
        """Whether a turn is currently owned by this controller."""

        return self._task is not None

    def cancel(self) -> bool:
        """Request cancellation of the active turn, if one exists."""

        task = self._task
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def run(self, awaitable: Awaitable[ResultT]) -> ResultT | _Cancelled:
        """Run an awaitable as the active turn and normalize its cancellation."""

        if self._task is not None:
            raise RuntimeError("a turn is already active")

        task = asyncio.ensure_future(awaitable)
        self._task = task
        loop = asyncio.get_running_loop()
        previous_sigint: signal.Handlers | None = None
        handler_installed = False

        if self._handle_sigint:
            previous_sigint = signal.getsignal(signal.SIGINT)
            try:
                loop.add_signal_handler(signal.SIGINT, self.cancel)
                handler_installed = True
            except NotImplementedError, RuntimeError, ValueError:
                # Signal handlers are only available on supported loops in the
                # main thread. Programmatic cancellation remains available.
                pass

        try:
            try:
                return await task
            except asyncio.CancelledError:
                caller = asyncio.current_task()
                if caller is not None and caller.cancelling():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    raise
                return CANCELLED
        finally:
            if handler_installed:
                loop.remove_signal_handler(signal.SIGINT)
                signal.signal(signal.SIGINT, previous_sigint)
            self._task = None
