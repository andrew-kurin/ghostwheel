from __future__ import annotations

import asyncio
import os
import signal

import pytest

from ghostwheel.cancellation import CANCELLED, TurnCancellation


def test_run_tracks_active_turn_and_returns_result() -> None:
    async def scenario() -> None:
        controller = TurnCancellation()
        release = asyncio.Event()

        async def turn() -> str:
            assert controller.active is True
            await release.wait()
            return "finished"

        running = asyncio.create_task(controller.run(turn()))
        await asyncio.sleep(0)
        assert controller.active is True

        release.set()
        assert await running == "finished"
        assert controller.active is False
        assert controller.cancel() is False

    asyncio.run(scenario())


def test_cancel_normalizes_active_turn_cancellation() -> None:
    async def scenario() -> None:
        controller = TurnCancellation()
        started = asyncio.Event()
        was_cancelled = asyncio.Event()

        async def turn() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                was_cancelled.set()
                raise

        running = asyncio.create_task(controller.run(turn()))
        await started.wait()
        assert controller.cancel() is True
        assert await running is CANCELLED
        assert was_cancelled.is_set()
        assert controller.active is False

    asyncio.run(scenario())


def test_cancelling_run_cancels_child_and_propagates() -> None:
    async def scenario() -> None:
        controller = TurnCancellation()
        started = asyncio.Event()
        was_cancelled = asyncio.Event()

        async def turn() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                was_cancelled.set()

        running = asyncio.create_task(controller.run(turn()))
        await started.wait()
        running.cancel()

        with pytest.raises(asyncio.CancelledError):
            await running
        assert was_cancelled.is_set()
        assert controller.active is False

    asyncio.run(scenario())


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal handling required")
def test_sigint_cancels_turn_and_restores_previous_handler() -> None:
    previous_sigint = signal.getsignal(signal.SIGINT)

    async def scenario() -> None:
        controller = TurnCancellation(handle_sigint=True)
        started = asyncio.Event()

        async def turn() -> None:
            started.set()
            await asyncio.Event().wait()

        running = asyncio.create_task(controller.run(turn()))
        await started.wait()
        os.kill(os.getpid(), signal.SIGINT)

        assert await asyncio.wait_for(running, 1) is CANCELLED
        assert controller.active is False

    asyncio.run(scenario())
    assert signal.getsignal(signal.SIGINT) is previous_sigint
