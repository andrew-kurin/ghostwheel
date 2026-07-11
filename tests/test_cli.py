import asyncio
from io import StringIO
from typing import Any

from rich.console import Console

from ghostwheel.cli import CommandKind, parse_command, run_cli
from ghostwheel.events import TextOutput, ToolFailed, ToolFinished
from ghostwheel.review import ReviewFailed
from ghostwheel.rich_ui import RichPresenter
from ghostwheel.session import TurnNoResult


class FakeConsole:
    def __init__(self, inputs: list[str]) -> None:
        self.inputs = iter(inputs)

    def input(self, _prompt: str) -> str:
        return next(self.inputs)


class FakeSession:
    def __init__(self) -> None:
        self.history = ("chat-history",)
        self.last_compacted_turns = 0
        self.sent: list[str] = []
        self.cleared = 0

    async def send(self, prompt: str) -> TurnNoResult:
        self.sent.append(prompt)
        return TurnNoResult()

    def clear(self) -> None:
        self.cleared += 1
        self.history = ()


class FakeReviews:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def review(
        self,
        paths: str,
        *,
        chat_history: tuple[object, ...],
    ) -> ReviewFailed:
        self.calls.append((paths, chat_history))
        return ReviewFailed("not available")


class FakePresenter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def welcome(self) -> None:
        self.calls.append(("welcome", None))

    def goodbye(self) -> None:
        self.calls.append(("goodbye", None))

    def history_cleared(self) -> None:
        self.calls.append(("cleared", None))

    def history_compacted(self, count: int) -> None:
        self.calls.append(("compacted", count))

    def turn_outcome(self, outcome: object) -> None:
        self.calls.append(("turn", outcome))

    def review_outcome(self, outcome: object) -> None:
        self.calls.append(("review", outcome))


def test_parse_command_requires_a_command_boundary() -> None:
    assert parse_command(" /review src ").kind is CommandKind.REVIEW
    assert parse_command("/review").value == "."
    assert parse_command("/reviewer src").kind is CommandKind.CHAT
    assert parse_command("   ").kind is CommandKind.EMPTY


def test_cli_routes_commands_without_adding_review_to_chat_history() -> None:
    session = FakeSession()
    reviews = FakeReviews()
    presenter = FakePresenter()

    asyncio.run(
        run_cli(
            FakeConsole(["/review src", "hello", "/clear", "/quit"]),  # type: ignore[arg-type]
            session,  # type: ignore[arg-type]
            reviews,  # type: ignore[arg-type]
            presenter=presenter,  # type: ignore[arg-type]
        )
    )

    assert reviews.calls == [("src", ("chat-history",))]
    assert session.sent == ["hello"]
    assert session.cleared == 1
    assert [name for name, _value in presenter.calls] == [
        "welcome",
        "review",
        "turn",
        "cleared",
        "goodbye",
    ]


def test_rich_presenter_treats_model_and_tool_content_as_plain_text() -> None:
    output = StringIO()
    presenter = RichPresenter(
        Console(file=output, color_system=None, force_terminal=False, width=120)
    )

    asyncio.run(presenter.handle_event(TextOutput("[/] [bold]literal[/bold]")))
    asyncio.run(presenter.handle_event(ToolFinished("read", "[/]")))
    asyncio.run(presenter.handle_event(ToolFailed("grep", "[bad]")))

    rendered = output.getvalue()
    assert "[/] [bold]literal[/bold]" in rendered
    assert "← read: [/]" in rendered
    assert "← grep failed: [bad]" in rendered
