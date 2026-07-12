import asyncio

import pytest

from ghostwheel.event_dispatcher import EventDeliveryError, EventDispatcher
from ghostwheel.events import TextOutput


def test_event_dispatcher_delivers_to_a_late_bound_sink() -> None:
    dispatcher = EventDispatcher()
    delivered: list[object] = []
    dispatcher.bind(delivered.append)

    asyncio.run(dispatcher.emit(TextOutput("hello")))

    assert delivered == [TextOutput("hello")]
    assert dispatcher.is_bound is True


def test_event_dispatcher_awaits_async_sinks() -> None:
    dispatcher = EventDispatcher()
    delivered: list[object] = []

    async def sink(event: object) -> None:
        await asyncio.sleep(0)
        delivered.append(event)

    dispatcher.bind(sink)
    asyncio.run(dispatcher(TextOutput("hello")))

    assert delivered == [TextOutput("hello")]


def test_event_dispatcher_rejects_missing_or_duplicate_bindings() -> None:
    dispatcher = EventDispatcher()

    with pytest.raises(RuntimeError, match="not bound"):
        asyncio.run(dispatcher.emit(TextOutput("hello")))

    dispatcher.bind(lambda _event: None)
    with pytest.raises(RuntimeError, match="already bound"):
        dispatcher.bind(lambda _event: None)


def test_event_dispatcher_preserves_sink_failures_as_delivery_errors() -> None:
    dispatcher = EventDispatcher()
    cause = RuntimeError("presenter exploded")

    def broken_sink(_event: object) -> None:
        raise cause

    event = TextOutput("partial response")
    dispatcher.bind(broken_sink)

    with pytest.raises(EventDeliveryError) as raised:
        asyncio.run(dispatcher.emit(event))

    assert raised.value.event is event
    assert raised.value.cause is cause
    assert raised.value.__cause__ is cause
