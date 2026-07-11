"""Input primitives shared by Ghostwheel's plain and persistent interfaces."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Protocol

from rich.console import Console


COMMANDS = (
    "/clear",
    "/help",
    "/model",
    "/quit",
    "/retry",
    "/review",
    "/tools",
)


class InputReader(Protocol):
    async def read(self) -> str: ...


class ConsoleInputReader:
    """Compatibility reader for pipes, redirected IO, and tests."""

    def __init__(self, console: Console) -> None:
        self.console = console

    async def read(self) -> str:
        prompt = (
            "\n[bold cyan]> [/bold cyan]"
            if getattr(self.console, "is_terminal", True)
            else ""
        )
        return self.console.input(prompt)


def default_history_path() -> Path:
    state_home = os.environ.get("XDG_STATE_HOME")
    base = Path(state_home).expanduser() if state_home else Path.home() / ".local/state"
    return base / "ghostwheel" / "input-history"


class InputHistory:
    """Small private prompt history compatible with prompt_toolkit's file format."""

    def __init__(self, path: Path | None) -> None:
        self.path = path.expanduser() if path is not None else None
        self.entries = self._load()
        if self.path is not None:
            self._ensure_file()

    def append(self, value: str) -> None:
        if not value.strip():
            return
        self.entries.append(value)
        if self.path is None:
            return
        self._ensure_file()
        with self.path.open("a", encoding="utf-8") as history_file:
            history_file.write(f"\n# {dt.datetime.now().isoformat()}\n")
            for line in value.split("\n"):
                history_file.write(f"+{line}\n")

    def _load(self) -> list[str]:
        if self.path is None or not self.path.exists():
            return []
        entries: list[str] = []
        lines: list[str] = []
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("+"):
                lines.append(line[1:])
            elif lines:
                entries.append("\n".join(lines))
                lines = []
        if lines:
            entries.append("\n".join(lines))
        return entries

    def _ensure_file(self) -> None:
        assert self.path is not None
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        os.close(descriptor)
        self.path.chmod(0o600)
