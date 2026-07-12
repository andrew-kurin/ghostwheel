"""Shared application fixtures for Textual integration tests."""

import asyncio

from ghostwheel.app_info import AppInfo
from ghostwheel.runtime_contracts import TurnSucceeded
from ghostwheel.textual_ui import GhostwheelApp


class FakeSession:
    def __init__(self) -> None:
        self.history: tuple[object, ...] = ()
        self.last_compacted_turns = 0
        self.last_compaction = None
        self.sent: list[str] = []
        self.sent_event = asyncio.Event()
        self.estimated_context_tokens = 0
        self.context_tokens_estimated = True
        self.context_window_tokens = 16_384
        self.compaction_enabled = True

    @property
    def turn_count(self) -> int:
        return len(self.sent)

    async def send(self, prompt: str) -> TurnSucceeded[str]:
        self.sent.append(prompt)
        self.sent_event.set()
        return TurnSucceeded(f"Received {prompt}", ())

    def clear(self) -> None:
        self.history = ()
        self.estimated_context_tokens = 0


class FakeReviews:
    async def review(self, *_args, **_kwargs):
        raise AssertionError("review was not expected")


def make_app(
    session: FakeSession | None = None,
    *,
    vim_mode: bool = True,
) -> GhostwheelApp:
    return GhostwheelApp(
        session or FakeSession(),
        FakeReviews(),  # type: ignore[arg-type]
        app_info=AppInfo(
            workspace="/tmp/workspace",
            provider="ollama",
            model="test-model",
            tool_profile="full",
        ),
        vim_mode=vim_mode,
    )
