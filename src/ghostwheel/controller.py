"""UI-neutral command parsing and application orchestration."""

from __future__ import annotations

from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from difflib import get_close_matches
from enum import Enum
from typing import Protocol, TypeVar, cast

from pydantic_ai.messages import ModelMessage

from ghostwheel.cancellation import CANCELLED
from ghostwheel.review import ReviewOutcome
from ghostwheel.runtime_contracts import TurnOutcome


class CommandKind(str, Enum):
    EMPTY = "empty"
    QUIT = "quit"
    CLEAR = "clear"
    REVIEW = "review"
    RETRY = "retry"
    HELP = "help"
    MODEL = "model"
    TOOLS = "tools"
    UNKNOWN = "unknown"
    CHAT = "chat"


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    kind: CommandKind
    value: str = ""


COMMANDS = (
    "/clear",
    "/help",
    "/model",
    "/quit",
    "/retry",
    "/review",
    "/tools",
)


NO_ARGUMENT_COMMANDS = {
    "/quit": CommandKind.QUIT,
    "/clear": CommandKind.CLEAR,
    "/retry": CommandKind.RETRY,
    "/help": CommandKind.HELP,
    "/model": CommandKind.MODEL,
    "/tools": CommandKind.TOOLS,
}


class CompactionNotice(Protocol):
    before_tokens: int
    after_tokens: int


class SessionPort(Protocol):
    @property
    def history(self) -> Sequence[ModelMessage]: ...

    @property
    def last_compaction(self) -> CompactionNotice | None: ...

    async def send(self, prompt: str) -> TurnOutcome: ...

    def clear(self) -> None: ...


class ReviewPort(Protocol):
    async def review(
        self,
        paths: str,
        *,
        chat_history: Sequence[ModelMessage],
    ) -> ReviewOutcome: ...


class PresenterPort(Protocol):
    def welcome(self) -> None: ...

    def goodbye(self) -> None: ...

    def help(self) -> None: ...

    def model_info(self) -> None: ...

    def tools_info(self) -> None: ...

    def unknown_command(
        self,
        command: str,
        suggestion: str | None = None,
    ) -> None: ...

    def retry_unavailable(self) -> None: ...

    def history_cleared(self) -> None: ...

    def history_compacted(self, before_tokens: int, after_tokens: int) -> None: ...

    def turn_started(self, label: str = "Thinking…") -> None: ...

    def turn_cancelled(self) -> None: ...

    def turn_outcome(self, outcome: TurnOutcome) -> None: ...

    def review_outcome(self, outcome: ReviewOutcome) -> None: ...


class InputPort(Protocol):
    async def read(self) -> str: ...


ResultT = TypeVar("ResultT")


class CancellationPort(Protocol):
    def cancel(self) -> bool: ...

    async def run(self, awaitable: Awaitable[ResultT]) -> object: ...


def parse_command(value: str) -> ParsedCommand:
    normalized = value.strip()
    if not normalized:
        return ParsedCommand(CommandKind.EMPTY)

    parts = normalized.split(maxsplit=1)
    command = parts[0].lower()
    arguments = parts[1].strip() if len(parts) == 2 else ""
    if command == "/review":
        return ParsedCommand(CommandKind.REVIEW, arguments or ".")
    if command in NO_ARGUMENT_COMMANDS:
        if arguments:
            return ParsedCommand(CommandKind.UNKNOWN, normalized)
        return ParsedCommand(NO_ARGUMENT_COMMANDS[command])
    if command.startswith("/"):
        return ParsedCommand(CommandKind.UNKNOWN, normalized)
    return ParsedCommand(CommandKind.CHAT, normalized)


async def run_command_loop(
    session: SessionPort,
    reviews: ReviewPort,
    *,
    presenter: PresenterPort,
    input_reader: InputPort,
    cancellation: CancellationPort,
) -> None:
    """Route user commands between application services and a presenter."""

    last_repeatable: ParsedCommand | None = None
    presenter.welcome()

    while True:
        try:
            user_input = await input_reader.read()
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
            last_repeatable = None
            presenter.history_cleared()
            continue
        if command.kind is CommandKind.HELP:
            presenter.help()
            continue
        if command.kind is CommandKind.MODEL:
            presenter.model_info()
            continue
        if command.kind is CommandKind.TOOLS:
            presenter.tools_info()
            continue
        if command.kind is CommandKind.UNKNOWN:
            token = command.value.split(maxsplit=1)[0].lower()
            matches = get_close_matches(token, COMMANDS, n=1, cutoff=0.55)
            presenter.unknown_command(command.value, matches[0] if matches else None)
            continue
        if command.kind is CommandKind.RETRY:
            if last_repeatable is None:
                presenter.retry_unavailable()
                continue
            command = last_repeatable

        if command.kind is CommandKind.REVIEW:
            last_repeatable = command
            presenter.turn_started("Reviewing…")
            outcome = await cancellation.run(
                reviews.review(
                    command.value,
                    chat_history=session.history,
                )
            )
            if outcome is CANCELLED:
                presenter.turn_cancelled()
                continue
            presenter.review_outcome(cast(ReviewOutcome, outcome))
            continue

        last_repeatable = command
        presenter.turn_started()
        outcome = await cancellation.run(session.send(command.value))
        if outcome is CANCELLED:
            presenter.turn_cancelled()
            continue
        presenter.turn_outcome(cast(TurnOutcome, outcome))
        compaction = session.last_compaction
        if compaction is not None:
            presenter.history_compacted(
                compaction.before_tokens,
                compaction.after_tokens,
            )
