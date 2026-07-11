from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import get_close_matches
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console

from ghostwheel.cancellation import CANCELLED, TurnCancellation
from ghostwheel.input_ui import (
    COMMANDS,
    ConsoleInputReader,
    InputReader,
    default_history_path,
)
from ghostwheel.review import ReviewService
from ghostwheel.rich_ui import RichPresenter
from ghostwheel.session import ChatSession


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


NO_ARGUMENT_COMMANDS = {
    "/quit": CommandKind.QUIT,
    "/clear": CommandKind.CLEAR,
    "/retry": CommandKind.RETRY,
    "/help": CommandKind.HELP,
    "/model": CommandKind.MODEL,
    "/tools": CommandKind.TOOLS,
}


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


async def run_cli(
    console: Console,
    session: ChatSession,
    reviews: ReviewService,
    *,
    presenter: RichPresenter | None = None,
    input_reader: InputReader | None = None,
    cancellation: TurnCancellation | None = None,
) -> None:
    presenter = presenter or RichPresenter(console)
    input_reader = input_reader or ConsoleInputReader(console)
    cancellation = cancellation or TurnCancellation(handle_sigint=True)
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
            presenter.review_outcome(outcome)
            continue

        last_repeatable = command
        presenter.turn_started()
        outcome = await cancellation.run(session.send(command.value))
        if outcome is CANCELLED:
            presenter.turn_cancelled()
            continue
        presenter.turn_outcome(outcome)
        if session.last_compacted_turns:
            presenter.history_compacted(session.last_compacted_turns)


def _package_version() -> str:
    try:
        return version("ghostwheel")
    except PackageNotFoundError:
        return "0.1.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ghostwheel",
        description="Local coding assistant and code-review chat",
    )
    parser.add_argument(
        "--ui",
        choices=("auto", "interactive", "plain"),
        default="auto",
        help="terminal interface (default: auto)",
    )
    parser.add_argument(
        "--vim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use Vim-style prompt editing in the interactive interface (default: enabled)",
    )
    history = parser.add_mutually_exclusive_group()
    history.add_argument(
        "--history-file",
        type=Path,
        help="path for interactive input history",
    )
    history.add_argument(
        "--no-history",
        action="store_true",
        help="keep input history in memory only",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
    )
    return parser


def _isatty(stream: object) -> bool:
    try:
        isatty = getattr(stream, "isatty")
        return bool(isatty())
    except AttributeError, OSError, TypeError:
        return False


def _interactive_mode(
    ui: str,
    stdin: object,
    stdout: object,
    *,
    term: str | None = None,
) -> bool:
    terminals_available = _isatty(stdin) and _isatty(stdout)
    if ui == "interactive":
        if not terminals_available:
            raise ValueError("--ui interactive requires terminal stdin and stdout")
        return True
    if ui == "plain":
        return False
    terminal_name = os.environ.get("TERM", "") if term is None else term
    return terminals_available and terminal_name.lower() != "dumb"


def main(argv: Sequence[str] | None = None) -> None:
    from ghostwheel.bootstrap import build_application
    from ghostwheel.config import Settings
    from ghostwheel.telemetry import configure_observability

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        interactive = _interactive_mode(args.ui, sys.stdin, sys.stdout)
        config = Settings().resolve()
    except (ValidationError, ValueError) as error:
        parser.error(str(error))

    configure_observability(config.observability)
    console = Console()
    history_path = (
        None if args.no_history else (args.history_file or default_history_path())
    )
    event_handler = None

    async def route_interactive_event(event) -> None:
        if event_handler is not None:
            await event_handler(event)

    application = build_application(
        config,
        console,
        event_sink=route_interactive_event if interactive else None,
    )
    try:
        if interactive:
            from ghostwheel.textual_ui import GhostwheelApp

            assert application.presenter.app_info is not None
            tui = GhostwheelApp(
                console,
                application.session,
                application.reviews,
                app_info=application.presenter.app_info,
                max_turns=config.history.max_turns,
                history_path=history_path,
                vim_mode=args.vim,
            )
            event_handler = tui.presenter.handle_event
            tui.run()
        else:
            asyncio.run(
                run_cli(
                    console,
                    application.session,
                    application.reviews,
                    presenter=application.presenter,
                    input_reader=ConsoleInputReader(console),
                )
            )
    finally:
        application.close()
