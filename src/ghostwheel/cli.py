import asyncio
from dataclasses import dataclass
from enum import Enum

from rich.console import Console

from ghostwheel.review import ReviewService
from ghostwheel.rich_ui import RichPresenter
from ghostwheel.session import ChatSession


class CommandKind(str, Enum):
    EMPTY = "empty"
    QUIT = "quit"
    CLEAR = "clear"
    REVIEW = "review"
    CHAT = "chat"


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    kind: CommandKind
    value: str = ""


def parse_command(value: str) -> ParsedCommand:
    normalized = value.strip()
    if not normalized:
        return ParsedCommand(CommandKind.EMPTY)

    command, _, arguments = normalized.partition(" ")
    command_lower = command.lower()
    if command_lower == "/quit":
        return ParsedCommand(CommandKind.QUIT)
    if command_lower == "/clear":
        return ParsedCommand(CommandKind.CLEAR)
    if command_lower == "/review":
        return ParsedCommand(CommandKind.REVIEW, arguments.strip() or ".")
    return ParsedCommand(CommandKind.CHAT, normalized)


async def run_cli(
    console: Console,
    session: ChatSession,
    reviews: ReviewService,
    *,
    presenter: RichPresenter | None = None,
) -> None:
    presenter = presenter or RichPresenter(console)
    presenter.welcome()

    while True:
        try:
            user_input = console.input("\n[bold cyan]> [/bold cyan]")
        except EOFError, KeyboardInterrupt, StopIteration:
            presenter.goodbye()
            break

        command = parse_command(user_input)
        if command.kind is CommandKind.EMPTY:
            continue
        if command.kind is CommandKind.QUIT:
            presenter.goodbye()
            break
        if command.kind is CommandKind.CLEAR:
            session.clear()
            presenter.history_cleared()
            continue
        if command.kind is CommandKind.REVIEW:
            outcome = await reviews.review(
                command.value,
                chat_history=session.history,
            )
            presenter.review_outcome(outcome)
            continue

        outcome = await session.send(command.value)
        presenter.turn_outcome(outcome)
        if session.last_compacted_turns:
            presenter.history_compacted(session.last_compacted_turns)


def main() -> None:
    from ghostwheel.bootstrap import build_application
    from ghostwheel.config import Settings
    from ghostwheel.telemetry import configure_observability

    config = Settings().resolve()
    configure_observability(config.observability)
    console = Console()
    application = build_application(config, console)
    try:
        asyncio.run(
            run_cli(
                console,
                application.session,
                application.reviews,
                presenter=application.presenter,
            )
        )
    finally:
        application.close()
