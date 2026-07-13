import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from ghostwheel.cancellation import TurnCancellation
from ghostwheel.cli import _supports_prompt_toolkit, build_parser, main
from ghostwheel.controller import CommandKind, parse_command, run_command_loop
from ghostwheel.event_dispatcher import EventDispatcher
from ghostwheel.review import ReviewFailed
from ghostwheel.session import CompactionStats, TurnNoResult


class FakeInput:
    def __init__(self, inputs: list[str]) -> None:
        self.inputs = iter(inputs)

    async def read(self) -> str:
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


async def run_commands(
    inputs: list[str],
    session: FakeSession,
    reviews: FakeReviews,
    presenter: FakePresenter,
    *,
    cancellation: TurnCancellation | None = None,
) -> None:
    await run_command_loop(
        session,
        reviews,
        presenter=presenter,
        input_reader=FakeInput(inputs),
        cancellation=cancellation or TurnCancellation(),
    )


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


def test_command_loop_routes_review_without_adding_it_to_chat_history() -> None:
    session = FakeSession()
    reviews = FakeReviews()
    presenter = FakePresenter()

    asyncio.run(
        run_commands(
            ["/review src", "hello", "/clear", "/quit"],
            session,
            reviews,
            presenter,
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
        run_commands(
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
            ],
            session,
            reviews,
            presenter,
        )
    )

    assert session.sent == ["hello", "hello"]
    assert reviews.calls == []
    call_names = [name for name, _value in presenter.calls]
    assert call_names[:5] == ["welcome", "help", "model", "tools", "unknown"]
    assert call_names.count("retry-unavailable") == 2
    assert ("unknown", ("/retrz", "/retry")) in presenter.calls


def test_command_loop_reports_compaction_token_reduction() -> None:
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
        run_commands(
            ["hello", "/quit"],
            session,
            FakeReviews(),
            presenter,
        )
    )

    assert ("compacted", (12_000, 4_200)) in presenter.calls


def test_programmatic_cancellation_keeps_the_loop_alive_across_turns() -> None:
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
        cancellation = TurnCancellation()
        loop_task = asyncio.create_task(
            run_commands(
                ["one", "two", "/quit"],
                session,
                FakeReviews(),
                presenter,
                cancellation=cancellation,
            )
        )
        assert await asyncio.wait_for(session.started.get(), 1) == "one"
        assert cancellation.cancel() is True
        assert await asyncio.wait_for(session.started.get(), 1) == "two"
        assert cancellation.cancel() is True
        await asyncio.wait_for(loop_task, 1)
        return session, presenter

    session, presenter = asyncio.run(scenario())

    assert session.cancelled == ["one", "two"]
    assert [name for name, _value in presenter.calls].count("cancelled") == 2
    assert presenter.calls[-1] == ("goodbye", None)


def test_cli_has_one_automatic_terminal_interface(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as help_exit:
        build_parser().parse_args(["--help"])
    assert help_exit.value.code == 0
    help_output = capsys.readouterr().out
    assert "--ui" not in help_output
    assert "--vim" in help_output
    assert build_parser().parse_args([]).vim is True
    assert build_parser().parse_args(["--vim"]).vim is True
    assert build_parser().parse_args(["--no-vim"]).vim is False

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--ui", "plain"])

    class Stream:
        def __init__(self, terminal: bool) -> None:
            self.terminal = terminal

        def isatty(self) -> bool:
            return self.terminal

    assert _supports_prompt_toolkit(Stream(True), Stream(True), term="xterm") is True
    assert _supports_prompt_toolkit(Stream(True), Stream(True), term="XTERM-256COLOR")
    assert _supports_prompt_toolkit(Stream(True), Stream(False), term="xterm") is False

    for unsupported_term in ("", "   ", "dumb", "DUMB", "unknown", "UnKnOwN"):
        assert (
            _supports_prompt_toolkit(
                Stream(True),
                Stream(True),
                term=unsupported_term,
            )
            is False
        )


def test_main_reports_resolution_errors_as_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import ghostwheel.config as config_module

    class InvalidSettings:
        def resolve(self) -> object:
            raise ValueError("Unknown model provider: invalid")

    monkeypatch.setattr(config_module, "Settings", InvalidSettings)

    with pytest.raises(SystemExit) as exit_info:
        main(["--no-history"])

    assert exit_info.value.code == 2
    error_output = capsys.readouterr().err
    assert "ghostwheel: error: Unknown model provider: invalid" in error_output
    assert "Traceback" not in error_output


def test_main_does_not_build_runtime_before_asyncio_run_accepts_coroutine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ghostwheel.bootstrap as bootstrap_module
    import ghostwheel.cli as cli_module
    import ghostwheel.config as config_module
    import ghostwheel.telemetry as telemetry_module

    config = SimpleNamespace(observability=object())
    runtime_built = False
    rejected_coroutine: object | None = None

    class FakeSettings:
        def resolve(self) -> object:
            return config

    def unexpected_build(*_args: object, **_kwargs: object) -> None:
        nonlocal runtime_built
        runtime_built = True

    def reject_coroutine(coroutine: object) -> None:
        nonlocal rejected_coroutine
        rejected_coroutine = coroutine
        raise RuntimeError("event loop unavailable")

    monkeypatch.setattr(cli_module, "_supports_prompt_toolkit", lambda *_args: False)
    monkeypatch.setattr(config_module, "Settings", FakeSettings)
    monkeypatch.setattr(
        telemetry_module,
        "configure_observability",
        lambda _value: None,
    )
    monkeypatch.setattr(bootstrap_module, "build_runtime", unexpected_build)
    monkeypatch.setattr(cli_module.asyncio, "run", reject_coroutine)

    with pytest.raises(RuntimeError, match="event loop unavailable"):
        main(["--no-history"])

    assert runtime_built is False
    assert rejected_coroutine is not None
    assert getattr(rejected_coroutine, "cr_frame") is None


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [([], True), (["--vim"], True), (["--no-vim"], False)],
)
def test_main_constructs_the_single_terminal_ui(
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
    expected: bool,
) -> None:
    import ghostwheel.bootstrap as bootstrap_module
    import ghostwheel.cli as cli_module
    import ghostwheel.config as config_module
    import ghostwheel.telemetry as telemetry_module

    captured: dict[str, object] = {}

    class FakeRuntime:
        session = object()
        reviews = object()
        app_info = object()

        async def __aenter__(self) -> "FakeRuntime":
            captured["started"] = True
            return self

        async def __aexit__(self, *_exc_info: object) -> None:
            captured["closed"] = True

    runtime = FakeRuntime()
    config = SimpleNamespace(observability=object())

    class FakeSettings:
        def resolve(self) -> object:
            return config

    class FakeTerminalUI:
        def __init__(self, console: object, **kwargs: object) -> None:
            captured["console"] = console
            captured.update(kwargs)

        async def handle_event(self, _event: object) -> None:
            return None

        async def run(self, reviews: object) -> None:
            captured["reviews"] = reviews
            captured["ran"] = True

    def build_fake_runtime(*_args: object, **kwargs: object) -> FakeRuntime:
        captured["event_sink"] = kwargs["event_sink"]
        return runtime

    monkeypatch.setattr(cli_module, "_supports_prompt_toolkit", lambda *_args: True)
    monkeypatch.setattr(cli_module, "TerminalUI", FakeTerminalUI)
    monkeypatch.setattr(config_module, "Settings", FakeSettings)
    monkeypatch.setattr(
        telemetry_module,
        "configure_observability",
        lambda _value: None,
    )
    monkeypatch.setattr(bootstrap_module, "build_runtime", build_fake_runtime)

    main(["--no-history", *extra_args])

    assert captured["session"] is runtime.session
    assert captured["app_info"] is runtime.app_info
    assert captured["reviews"] is runtime.reviews
    assert captured["history_path"] is None
    assert captured["vim_mode"] is expected
    assert captured["interactive"] is True
    assert isinstance(captured["event_sink"], EventDispatcher)
    assert captured["ran"] is True
    assert captured["started"] is True
    assert captured["closed"] is True
