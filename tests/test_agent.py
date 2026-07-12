import asyncio
import importlib
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.test import TestModel

from ghostwheel.config import Settings
from ghostwheel.events import ToolFailed, ToolFinished, ToolStarted
from ghostwheel.pydantic_runner import (
    PydanticAgentRunner,
    _failure_kind,
    _handle_tool_event,
)
from ghostwheel.schemas import ReviewResult
from ghostwheel.session import FailureKind, TurnFailed, TurnNoResult, TurnSucceeded
from ghostwheel.tools.bash import bash
from ghostwheel.tools.command import CommandResult
from ghostwheel.tools.deps import ToolDeps
from ghostwheel.tools.filesystem import DirectoryListing, ls


class FakeResult:
    def __init__(self, output: object, messages: list[object]) -> None:
        self.output = output
        self._messages = messages

    def new_messages(self) -> list[object]:
        return self._messages


class FakeRun:
    def __init__(self, result: object | None) -> None:
        self.result = result
        self.ctx = object()

    async def __aenter__(self) -> "FakeRun":
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        return None


class FakeAgent:
    def __init__(self, result: object | Exception | None) -> None:
        self.result = result
        self.calls: list[tuple[str, tuple[object, ...], object, type[object]]] = []

    def iter(
        self,
        prompt: str,
        *,
        message_history: tuple[object, ...],
        deps: object,
        output_type: type[object],
    ) -> FakeRun:
        self.calls.append((prompt, tuple(message_history), deps, output_type))
        if isinstance(self.result, Exception):
            raise self.result
        return FakeRun(self.result)


async def noop_stream(_run: object, _sink: object) -> None:
    return None


def test_agent_module_import_does_not_create_runtime_singletons() -> None:
    module = importlib.import_module("ghostwheel.agent")

    assert not hasattr(module, "config")
    assert not hasattr(module, "chat_agent")
    assert not hasattr(module, "review_agent")


def test_pydantic_runner_returns_explicit_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_module = importlib.import_module("ghostwheel.pydantic_runner")
    monkeypatch.setattr(runner_module, "stream_agent_run", noop_stream)
    result = FakeResult("ok", messages=["new-message"])
    deps = object()
    agent = FakeAgent(result)

    outcome = asyncio.run(
        PydanticAgentRunner(agent, deps).run("hello", (), output_type=str)  # type: ignore[arg-type]
    )

    assert isinstance(outcome, TurnSucceeded)
    assert outcome.output == "ok"
    assert outcome.new_messages == ("new-message",)
    assert agent.calls == [("hello", (), deps, str)]


def test_pydantic_runner_distinguishes_no_result_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_module = importlib.import_module("ghostwheel.pydantic_runner")
    monkeypatch.setattr(runner_module, "stream_agent_run", noop_stream)

    no_result = asyncio.run(
        PydanticAgentRunner(FakeAgent(None), object()).run("hello", (), output_type=str)  # type: ignore[arg-type]
    )
    failure = asyncio.run(
        PydanticAgentRunner(FakeAgent(RuntimeError("model exploded")), object()).run(  # type: ignore[arg-type]
            "hello", (), output_type=str
        )
    )

    assert isinstance(no_result, TurnNoResult)
    assert isinstance(failure, TurnFailed)
    assert failure.message == "model exploded"


def test_create_tool_deps_maps_resolved_limits(tmp_path: Path) -> None:
    module = importlib.import_module("ghostwheel.agent")
    config = Settings(
        max_output_bytes=123,
        max_entries=4,
        max_directory_scan_entries=9,
        max_matches=5,
        bash_timeout_seconds=6,
        max_search_file_bytes=7,
        max_search_files=8,
        regex_timeout_seconds=0.2,
        _env_file=None,
    ).resolve()

    deps = module.create_tool_deps(config, tmp_path)

    assert deps.cwd == tmp_path.resolve()
    assert deps.filesystem_roots == (tmp_path.resolve(),)
    assert deps.limits.max_output_bytes == 123
    assert deps.limits.max_entries == 4
    assert deps.limits.max_directory_scan_entries == 9
    assert deps.limits.max_matches == 5
    assert deps.limits.bash_timeout_seconds == 6
    assert deps.limits.max_search_file_bytes == 7
    assert deps.limits.max_search_files == 8
    assert deps.limits.regex_timeout_seconds == 0.2


def test_compaction_agent_is_tool_free_and_caps_summary_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_module = importlib.import_module("ghostwheel.agent")
    monkeypatch.setattr(agent_module, "build_model", lambda _spec: TestModel())
    config = Settings(_env_file=None).resolve()

    agent = agent_module.create_compaction_agent(config)

    assert agent.deps_type is type(None)
    assert agent._function_toolset.tools == {}
    assert agent.model_settings == {"max_tokens": 2_048, "temperature": 0.1}
    assert agent_module.COMPACTION_INSTRUCTIONS in agent._instructions


def test_pydantic_runner_supports_per_run_structured_output() -> None:
    outcome = asyncio.run(
        PydanticAgentRunner(Agent(TestModel()), None).run(
            "review",
            (),
            output_type=ReviewResult,
        )
    )

    assert isinstance(outcome, TurnSucceeded)
    assert isinstance(outcome.output, ReviewResult)


def test_runner_correlates_tool_events() -> None:
    events: list[object] = []
    call_id = "call-123"

    asyncio.run(
        _handle_tool_event(
            FunctionToolCallEvent(
                ToolCallPart("read", {"path": "README.md"}, tool_call_id=call_id)
            ),
            events.append,
        )
    )
    asyncio.run(
        _handle_tool_event(
            FunctionToolResultEvent(
                ToolReturnPart("read", "contents", tool_call_id=call_id)
            ),
            events.append,
        )
    )
    asyncio.run(
        _handle_tool_event(
            FunctionToolResultEvent(
                RetryPromptPart(
                    "outside workspace",
                    tool_name="read",
                    tool_call_id=call_id,
                )
            ),
            events.append,
        )
    )

    assert events == [
        ToolStarted("read", "{'path': 'README.md'}", call_id=call_id),
        ToolFinished("read", "contents", call_id=call_id),
        ToolFailed("read", "outside workspace", call_id=call_id),
    ]


def test_pydantic_runner_awaits_async_tools(tmp_path: Path) -> None:
    class FakeCommandRunner:
        def __init__(self) -> None:
            self.called = False

        async def run(self, *args: object, **kwargs: object) -> CommandResult:
            self.called = True
            return CommandResult(0, "ok", "", False, False)

    command_runner = FakeCommandRunner()
    agent = Agent(
        TestModel(call_tools=["bash"]),
        deps_type=ToolDeps,
        tools=(bash,),
    )
    deps = ToolDeps(cwd=tmp_path, command_runner=command_runner)

    outcome = asyncio.run(
        PydanticAgentRunner(agent, deps).run("inspect", (), output_type=str)
    )

    assert isinstance(outcome, TurnSucceeded)
    assert command_runner.called is True


def test_ls_sends_compact_text_and_retains_structured_metadata(tmp_path: Path) -> None:
    (tmp_path / "value.txt").write_text("value", encoding="utf-8")
    agent = Agent(
        TestModel(call_tools=["ls"]),
        deps_type=ToolDeps,
        tools=(ls,),
    )
    deps = ToolDeps(cwd=tmp_path)

    try:
        outcome = asyncio.run(
            PydanticAgentRunner(agent, deps).run("inspect", (), output_type=str)
        )
    finally:
        deps.close()

    assert isinstance(outcome, TurnSucceeded)
    tool_returns = [
        part
        for message in outcome.new_messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert len(tool_returns) == 1
    assert tool_returns[0].content.startswith(
        'ls "." depth=1 returned=1 scanned=1 complete=true reasons=-'
    )
    assert isinstance(tool_returns[0].metadata, DirectoryListing)
    assert [entry.name for entry in tool_returns[0].metadata.entries] == ["value.txt"]


def test_runner_only_classifies_structured_output_failures_for_fallback() -> None:
    assert (
        _failure_kind(UnexpectedModelBehavior("output validation failed"))
        is FailureKind.MODEL_OUTPUT
    )
    assert (
        _failure_kind(UnexpectedModelBehavior("tool exceeded maximum retries"))
        is FailureKind.TOOL
    )
    assert (
        _failure_kind(
            ModelHTTPError(
                400,
                "local",
                {"error": "response_format json_schema is unsupported"},
            )
        )
        is FailureKind.MODEL_OUTPUT
    )
    assert (
        _failure_kind(ModelHTTPError(503, "local", {"error": "unavailable"}))
        is FailureKind.PROVIDER
    )
