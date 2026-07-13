from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from rich.console import Console

from ghostwheel.event_dispatcher import EventDispatcher
from ghostwheel.terminal_io import supports_prompt_toolkit
from ghostwheel.terminal_ui import TerminalUI, default_history_path

if TYPE_CHECKING:
    from ghostwheel.config import AppConfig


__all__ = [
    "build_parser",
    "main",
]


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
        "--vim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use Vim-style prompt editing (default: enabled)",
    )
    history = parser.add_mutually_exclusive_group()
    history.add_argument(
        "--history-file",
        type=Path,
        help="path for prompt history",
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


def _supports_prompt_toolkit(
    stdin: object,
    stdout: object,
    *,
    term: str | None = None,
) -> bool:
    return _isatty(stdin) and _isatty(stdout) and supports_prompt_toolkit(term)


def main(argv: Sequence[str] | None = None) -> None:
    from ghostwheel.config import Settings
    from ghostwheel.telemetry import configure_observability

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = Settings().resolve()
    except ValueError as error:
        parser.error(str(error))

    configure_observability(config.observability)
    console = Console()
    history_path = (
        None if args.no_history else (args.history_file or default_history_path())
    )
    terminal_ui = _run_terminal_ui(
        config,
        console,
        interactive=_supports_prompt_toolkit(sys.stdin, sys.stdout),
        history_path=history_path,
        vim_mode=args.vim,
        input_stream=sys.stdin,
    )
    try:
        asyncio.run(terminal_ui)
    except BaseException:
        # ``asyncio.run`` rejects before taking ownership when called from an
        # active loop (and may fail while creating its runner). Closing is a
        # no-op once the coroutine has completed, and prevents an unawaited
        # coroutine warning when it never started.
        terminal_ui.close()
        raise


async def _run_terminal_ui(
    config: AppConfig,
    console: Console,
    *,
    interactive: bool,
    history_path: Path | None,
    vim_mode: bool,
    input_stream: TextIO,
) -> None:
    """Construct and run the terminal UI inside one event-loop lifetime."""

    from ghostwheel.bootstrap import build_runtime

    events = EventDispatcher()
    runtime = build_runtime(config, event_sink=events)

    async with runtime:
        ui = TerminalUI(
            console,
            session=runtime.session,
            app_info=runtime.app_info,
            history_path=history_path,
            vim_mode=vim_mode,
            interactive=interactive,
            input_stream=input_stream,
        )
        events.bind(ui.handle_event)
        await ui.run(runtime.reviews)
