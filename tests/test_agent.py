import asyncio
import importlib
from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

import pytest


class FakeConsole:
    def __init__(self, inputs: Iterable[str] = ()) -> None:
        self._inputs = iter(inputs)
        self.printed: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def print(self, *args: Any, **kwargs: Any) -> None:
        self.printed.append((args, kwargs))

    def input(self, _prompt: str) -> str:
        return next(self._inputs)

    def status(self, *_args: Any, **_kwargs: Any):
        return nullcontext()


class FakeRun:
    def __init__(self, result: object | None) -> None:
        self.result = result
        self.ctx = object()

    async def __aenter__(self) -> "FakeRun":
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        return None


class FakeResult:
    def __init__(self, output: str, messages: list[object]) -> None:
        self.output = output
        self._messages = messages

    def all_messages(self) -> list[object]:
        return self._messages


class FakeAgent:
    def __init__(self, *results: object) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, list[object], object]] = []

    def iter(self, prompt: str, message_history: list[object], deps: object) -> FakeRun:
        self.calls.append((prompt, list(message_history), deps))
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return FakeRun(result)


class FakeFormatter:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[str] = []

    async def run(self, prose: str) -> object:
        self.calls.append(prose)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def noop_stream_to_console(_run: object, _console: object) -> None:
    return None


def test_agent_module_import_does_not_create_runtime_singletons() -> None:
    module = importlib.import_module("ghostwheel.agent")

    assert not hasattr(module, "config")
    assert not hasattr(module, "model")
    assert not hasattr(module, "formatter_model")
    assert not hasattr(module, "agent")
    assert not hasattr(module, "formatter")


def test_run_agent_turn_returns_canonical_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_module = importlib.import_module("ghostwheel.agent")
    monkeypatch.setattr(agent_module, "stream_to_console", noop_stream_to_console)
    result = FakeResult("ok", messages=["new-history"])
    chat_agent = FakeAgent(result)
    history = ["old-history"]
    deps = object()

    actual = asyncio.run(
        agent_module.run_agent_turn(chat_agent, "hello", history, deps, FakeConsole())
    )

    assert actual is result
    assert chat_agent.calls == [("hello", ["old-history"], deps)]


def test_run_agent_turn_warns_when_no_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_module = importlib.import_module("ghostwheel.agent")
    monkeypatch.setattr(agent_module, "stream_to_console", noop_stream_to_console)
    console = FakeConsole()

    actual = asyncio.run(
        agent_module.run_agent_turn(FakeAgent(None), "hello", [], object(), console)
    )

    assert actual is None
    assert any(
        "history was not updated" in str(args[0]) for args, _kwargs in console.printed
    )


def test_run_agent_turn_reports_failures_without_updating_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_module = importlib.import_module("ghostwheel.agent")
    monkeypatch.setattr(agent_module, "stream_to_console", noop_stream_to_console)
    console = FakeConsole()

    actual = asyncio.run(
        agent_module.run_agent_turn(
            FakeAgent(RuntimeError("model exploded")), "hello", [], object(), console
        )
    )

    assert actual is None
    assert any(
        getattr(args[0], "renderable", None) == "model exploded"
        for args, _kwargs in console.printed
    )


def test_run_chat_keeps_previous_history_after_missing_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_module = importlib.import_module("ghostwheel.agent")
    monkeypatch.setattr(agent_module, "stream_to_console", noop_stream_to_console)
    successful_result = FakeResult("ok", messages=["successful-history"])
    chat_agent = FakeAgent(None, successful_result)
    console = FakeConsole(["first", "second", "/quit"])
    deps = object()

    asyncio.run(agent_module.run_chat(console, deps, chat_agent, object()))

    assert chat_agent.calls == [
        ("first", [], deps),
        ("second", [], deps),
    ]


def test_review_falls_back_to_raw_prose_when_formatter_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_module = importlib.import_module("ghostwheel.agent")
    monkeypatch.setattr(agent_module, "stream_to_console", noop_stream_to_console)
    review_result = FakeResult("raw review prose", messages=["review-history"])
    chat_agent = FakeAgent(review_result)
    formatter = FakeFormatter(RuntimeError("formatter unavailable"))
    console = FakeConsole(["/review src", "/quit"])
    deps = object()

    asyncio.run(agent_module.run_chat(console, deps, chat_agent, formatter))

    assert formatter.calls == ["raw review prose"]
    assert any(
        "formatter unavailable" in str(getattr(args[0], "renderable", ""))
        and "raw review prose" in str(getattr(args[0], "renderable", ""))
        for args, _kwargs in console.printed
    )
