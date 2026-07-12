"""Late-bound delivery for runtime events."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable

from ghostwheel.events import AgentEvent

EventSink = Callable[[AgentEvent], Awaitable[None] | None]


class EventDeliveryError(RuntimeError):
    """An event sink failed while presenting a runtime event."""

    def __init__(self, event: AgentEvent, cause: Exception) -> None:
        self.event = event
        self.cause = cause
        super().__init__(f"Could not deliver {type(event).__name__}: {cause}")


async def deliver_event(sink: EventSink, event: AgentEvent) -> None:
    """Deliver one event while preserving presentation as its own error domain."""

    try:
        delivered = sink(event)
        if inspect.isawaitable(delivered):
            await delivered
    except EventDeliveryError:
        # A dispatcher may itself be used as a sink. Preserve the original
        # event, cause, and exception chain instead of wrapping it a second time.
        raise
    except Exception as error:
        raise EventDeliveryError(event, error) from error


class EventDispatcher:
    """Connect a runtime created before its presenter to exactly one event sink."""

    def __init__(self) -> None:
        self._sink: EventSink | None = None

    @property
    def is_bound(self) -> bool:
        return self._sink is not None

    def bind(self, sink: EventSink) -> None:
        if self._sink is not None:
            raise RuntimeError("Event dispatcher is already bound")
        self._sink = sink

    async def emit(self, event: AgentEvent) -> None:
        sink = self._sink
        if sink is None:
            raise RuntimeError("Event dispatcher is not bound")
        await deliver_event(sink, event)

    async def __call__(self, event: AgentEvent) -> None:
        await self.emit(event)
