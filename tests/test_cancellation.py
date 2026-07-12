from __future__ import annotations

import asyncio

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
