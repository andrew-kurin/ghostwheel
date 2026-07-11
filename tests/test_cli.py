import asyncio
import os
import signal
from io import StringIO
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

from ghostwheel.cli import (
    CommandKind,
    _interactive_mode,
    build_parser,
    main,
    parse_command,
    run_cli,
)
from ghostwheel.events import (
    TextOutput,
    ToolFailed,
    ToolFinished,
)
from ghostwheel.review import ReviewFailed
from ghostwheel.rich_ui import RichPresenter
from ghostwheel.session import CompactionStats, TurnNoResult, TurnSucceeded


class FakeConsole:
    def __init__(self, inputs: list[str]) -> None:
        self.inputs = iter(inputs)

    def input(self, _prompt: str) -> str:
        return next(self.inputs)


class FakeSession:
    def __init__(self) -> None:
        self.history = ("chat-history",)
        self.last_compacted_turns = 0
        self.last_compaction: CompactionStats | None = None
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

    def help(self) -> None:
        self.calls.append(("help", None))

    def model_info(self) -> None:
        self.calls.append(("model", None))

    def tools_info(self) -> None:
        self.calls.append(("tools", None))

    def unknown_command(self, command: str, suggestion: str | None) -> None:
        self.calls.append(("unknown", (command, suggestion)))

    def retry_unavailable(self) -> None:
        self.calls.append(("retry-unavailable", None))

    def history_cleared(self) -> None:
        self.calls.append(("cleared", None))

    def history_compacted(self, before_tokens: int, after_tokens: int) -> None:
        self.calls.append(("compacted", (before_tokens, after_tokens)))

    def turn_started(self, label: str = "Thinking…") -> None:
        self.calls.append(("started", label))

    def turn_cancelled(self) -> None:
        self.calls.append(("cancelled", None))

    def turn_outcome(self, outcome: object) -> None:
        self.calls.append(("turn", outcome))

    def review_outcome(self, outcome: object) -> None:
        self.calls.append(("review", outcome))


def test_parse_command_requires_boundaries_and_keeps_slashes_local() -> None:
    assert parse_command(" /review src ").kind is CommandKind.REVIEW
    assert parse_command("/review\nsrc").value == "src"
    assert parse_command("/review").value == "."
    assert parse_command("/RETRY").kind is CommandKind.RETRY
    assert parse_command("/thinking").kind is CommandKind.UNKNOWN
    assert parse_command("/thinking ON").kind is CommandKind.UNKNOWN
    assert parse_command("/thinking off").kind is CommandKind.UNKNOWN
    assert parse_command("/thinking maybe").kind is CommandKind.UNKNOWN
    assert parse_command("/thinking on extra").kind is CommandKind.UNKNOWN
    assert parse_command("/verbose").kind is CommandKind.UNKNOWN
    assert parse_command("/reviewer src").kind is CommandKind.UNKNOWN
    assert parse_command("/retry extra").kind is CommandKind.UNKNOWN
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
        "started",
        "review",
        "started",
        "turn",
        "cleared",
        "goodbye",
    ]


def test_local_commands_do_not_reach_the_model_and_retry_repeats_chat() -> None:
    session = FakeSession()
    reviews = FakeReviews()
    presenter = FakePresenter()

    asyncio.run(
        run_cli(
            FakeConsole(
                [
                    "/help",
                    "/model",
                    "/tools",
                    "/retrz",
                    "/retry",
                    "hello",
                    "/retry",
                    "/clear",
                    "/retry",
                    "/quit",
                ]
            ),  # type: ignore[arg-type]
            session,  # type: ignore[arg-type]
            reviews,  # type: ignore[arg-type]
            presenter=presenter,  # type: ignore[arg-type]
        )
    )

    assert session.sent == ["hello", "hello"]
    assert reviews.calls == []
    call_names = [name for name, _value in presenter.calls]
    assert call_names[:5] == [
        "welcome",
        "help",
        "model",
        "tools",
        "unknown",
    ]
    assert call_names.count("retry-unavailable") == 2
    assert ("unknown", ("/retrz", "/retry")) in presenter.calls


def test_cli_reports_compaction_token_reduction() -> None:
    class CompactingSession(FakeSession):
        async def send(self, prompt: str) -> TurnNoResult:
            self.sent.append(prompt)
            self.last_compaction = CompactionStats(
                before_tokens=12_000,
                after_tokens=4_200,
                summarized_messages=6,
                summarized_turns=2,
            )
            return TurnNoResult()

    session = CompactingSession()
    presenter = FakePresenter()

    asyncio.run(
        run_cli(
            FakeConsole(["hello", "/quit"]),  # type: ignore[arg-type]
            session,  # type: ignore[arg-type]
            FakeReviews(),  # type: ignore[arg-type]
            presenter=presenter,  # type: ignore[arg-type]
        )
    )

    assert ("compacted", (12_000, 4_200)) in presenter.calls


def test_sigint_cancels_repeated_turns_and_keeps_the_prompt_alive() -> None:
    class BlockingSession(FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.started: asyncio.Queue[str] = asyncio.Queue()
            self.cancelled: list[str] = []

        async def send(self, prompt: str) -> TurnNoResult:
            self.sent.append(prompt)
            await self.started.put(prompt)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.append(prompt)
                raise

    async def scenario() -> tuple[BlockingSession, FakePresenter]:
        session = BlockingSession()
        presenter = FakePresenter()
        cli_task = asyncio.create_task(
            run_cli(
                FakeConsole(["one", "two", "/quit"]),  # type: ignore[arg-type]
                session,  # type: ignore[arg-type]
                FakeReviews(),  # type: ignore[arg-type]
                presenter=presenter,  # type: ignore[arg-type]
            )
        )
        assert await asyncio.wait_for(session.started.get(), 1) == "one"
        os.kill(os.getpid(), signal.SIGINT)
        assert await asyncio.wait_for(session.started.get(), 1) == "two"
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.wait_for(cli_task, 1)
        return session, presenter

    session, presenter = asyncio.run(scenario())

    assert session.cancelled == ["one", "two"]
    assert [name for name, _value in presenter.calls].count("cancelled") == 2
    assert presenter.calls[-1] == ("goodbye", None)


def test_cli_help_and_terminal_mode_detection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as help_exit:
        build_parser().parse_args(["--help"])
    assert help_exit.value.code == 0
    help_output = capsys.readouterr().out
    assert "--ui" in help_output
    assert "--vim" in help_output
    assert build_parser().parse_args([]).vim is True
    assert build_parser().parse_args(["--vim"]).vim is True
    assert build_parser().parse_args(["--no-vim"]).vim is False

    class Stream:
        def __init__(self, terminal: bool) -> None:
            self.terminal = terminal

        def isatty(self) -> bool:
            return self.terminal

    assert _interactive_mode("auto", Stream(True), Stream(True), term="xterm") is True
    assert _interactive_mode("auto", Stream(True), Stream(False), term="xterm") is False
    assert _interactive_mode("auto", Stream(True), Stream(True), term="dumb") is False
    assert _interactive_mode("plain", Stream(True), Stream(True)) is False
    with pytest.raises(ValueError, match="requires terminal"):
        _interactive_mode("interactive", Stream(False), Stream(True))


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [([], True), (["--vim"], True), (["--no-vim"], False)],
)
def test_main_passes_vim_mode_to_the_interactive_app(
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
    expected: bool,
) -> None:
    import ghostwheel.bootstrap as bootstrap_module
    import ghostwheel.cli as cli_module
    import ghostwheel.config as config_module
    import ghostwheel.telemetry as telemetry_module
    import ghostwheel.textual_ui as textual_module

    captured: dict[str, object] = {}
    application = SimpleNamespace(
        session=object(),
        reviews=object(),
        presenter=SimpleNamespace(app_info=object()),
        close=lambda: captured.__setitem__("closed", True),
    )
    config = SimpleNamespace(
        observability=object(),
        history=SimpleNamespace(
            context_window_tokens=16_384,
            compaction=SimpleNamespace(
                enabled=True,
                reserve_tokens=4_096,
                keep_recent_tokens=4_096,
                summary_tokens=2_048,
            ),
        ),
    )

    class FakeSettings:
        def resolve(self) -> object:
            return config

    class FakeTextualApp:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            captured.update(kwargs)
            self.presenter = SimpleNamespace(handle_event=lambda _event: None)

        def run(self, **kwargs: object) -> None:
            captured["ran"] = True
            captured["run_kwargs"] = kwargs

    monkeypatch.setattr(cli_module, "_interactive_mode", lambda *_args: True)
    monkeypatch.setattr(config_module, "Settings", FakeSettings)
    monkeypatch.setattr(
        telemetry_module, "configure_observability", lambda _value: None
    )
    monkeypatch.setattr(
        bootstrap_module, "build_application", lambda *_a, **_k: application
    )
    monkeypatch.setattr(textual_module, "GhostwheelApp", FakeTextualApp)

    main(["--ui", "interactive", "--no-history", *extra_args])

    assert captured["vim_mode"] is expected
    assert captured["ran"] is True
    assert captured["run_kwargs"] == {"mouse": False}
    assert captured["closed"] is True


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


def test_rich_presenter_renders_completed_interactive_output_as_markdown() -> None:
    output = StringIO()
    presenter = RichPresenter(
        Console(file=output, color_system=None, force_terminal=True, width=80),
        live=True,
    )

    presenter.turn_started()
    presenter.turn_outcome(
        TurnSucceeded(
            output="# Heading\n\n**strong** and [bold]literal[/bold]",
            new_messages=(),
        )
    )

    rendered = output.getvalue()
    assert "Heading" in rendered
    assert "strong" in rendered
    assert "[bold]literal[/bold]" in rendered


def test_rich_presenter_shows_compaction_token_reduction() -> None:
    output = StringIO()
    presenter = RichPresenter(
        Console(file=output, color_system=None, force_terminal=False, width=80)
    )

    presenter.history_compacted(12_000, 4_200)

    assert "Context compacted: 12k → ~4.2k." in output.getvalue()


def test_help_is_the_only_place_that_lists_shortcuts() -> None:
    output = StringIO()
    presenter = RichPresenter(
        Console(file=output, color_system=None, force_terminal=False, width=80)
    )

    presenter.help()

    rendered = output.getvalue()
    assert "Shortcuts" in rendered
    assert "Shift+Enter" in rendered
    assert "Alt+Enter" not in rendered
    assert "Ctrl+J" not in rendered
    assert "Ctrl+C" in rendered
    assert "Ctrl+O" in rendered
    assert "/thinking" not in rendered
    assert "/verbose" not in rendered
    assert "Enter sends" not in rendered
