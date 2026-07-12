from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError
from rich.console import Console

from ghostwheel.cancellation import TurnCancellation
from ghostwheel.controller import (
    CancellationPort,
    CommandKind,
    InputPort,
    ParsedCommand,
    PresenterPort,
    ReviewPort,
    SessionPort,
    parse_command,
    run_command_loop,
)
from ghostwheel.event_dispatcher import EventDispatcher
from ghostwheel.input_ui import (
    ConsoleInputReader,
    default_history_path,
)
from ghostwheel.rich_ui import RichPresenter

if TYPE_CHECKING:
    from ghostwheel.config import AppConfig


__all__ = [
    "CommandKind",
    "ParsedCommand",
    "build_parser",
    "main",
    "parse_command",
    "run_cli",
]


async def run_cli(
    console: Console,
    session: SessionPort,
    reviews: ReviewPort,
    *,
    presenter: PresenterPort | None = None,
    input_reader: InputPort | None = None,
    cancellation: CancellationPort | None = None,
) -> None:
    """Run the plain-console adapter around the neutral command controller."""

    presenter = presenter or RichPresenter(console)
    input_reader = input_reader or ConsoleInputReader(console)
    cancellation = cancellation or TurnCancellation(handle_sigint=True)
    await run_command_loop(
        session,
        reviews,
        presenter=presenter,
        input_reader=input_reader,
        cancellation=cancellation,
    )


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
    selected_ui = _run_selected_ui(
        config,
        console,
        interactive=interactive,
        history_path=history_path,
        vim_mode=args.vim,
    )
    try:
        asyncio.run(selected_ui)
    except BaseException:
        # ``asyncio.run`` rejects before taking ownership when called from an
        # active loop (and may fail while creating its runner). Closing is a
        # no-op once the coroutine has completed, and prevents an unawaited
        # coroutine warning when it never started.
        selected_ui.close()
        raise


async def _run_selected_ui(
    config: AppConfig,
    console: Console,
    *,
    interactive: bool,
    history_path: Path | None,
    vim_mode: bool,
) -> None:
    """Construct and run the selected UI inside one event-loop lifetime."""

    from ghostwheel.bootstrap import build_runtime

    events = EventDispatcher()
    runtime = build_runtime(config, event_sink=events)

    async with runtime:
        if interactive:
            from ghostwheel.textual_ui import GhostwheelApp

            tui = GhostwheelApp(
                runtime.session,
                runtime.reviews,
                app_info=runtime.app_info,
                history_path=history_path,
                vim_mode=vim_mode,
            )
            events.bind(tui.presenter.handle_event)
            # Textual routes wheel and scrollbar gestures while keeping content
            # selectable through its screen-level selection support.
            await tui.run_async(mouse=True)
        else:
            presenter = RichPresenter(
                console,
                app_info=runtime.app_info,
            )
            events.bind(presenter.handle_event)
            await run_cli(
                console,
                runtime.session,
                runtime.reviews,
                presenter=presenter,
                input_reader=ConsoleInputReader(console),
            )
