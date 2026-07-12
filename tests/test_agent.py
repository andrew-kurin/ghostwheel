import asyncio
import importlib
from pathlib import Path

import pytest
from pydantic_ai import Agent, Tool
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
from ghostwheel.event_dispatcher import EventDeliveryError, EventDispatcher
from ghostwheel.events import TextOutput, ToolFailed, ToolFinished, ToolStarted
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
from ghostwheel.tools.filesystem import DirectoryListing, ReadResult, ls, read
from ghostwheel.tools.search import GrepResult, grep


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
        self.entered = 0
        self.exited = 0

    async def __aenter__(self) -> "FakeAgent":
        self.entered += 1
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        self.exited += 1

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
    assert agent.entered == 1
    assert agent.exited == 1


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


def test_pydantic_runner_owns_clients_on_each_calling_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_module = importlib.import_module("ghostwheel.pydantic_runner")
    monkeypatch.setattr(runner_module, "stream_agent_run", noop_stream)

    class LoopBoundAgent(FakeAgent):
        def __init__(self) -> None:
            super().__init__(FakeResult("ok", messages=[]))
            self.active_loop: asyncio.AbstractEventLoop | None = None
            self.used_loops: list[asyncio.AbstractEventLoop] = []

        async def __aenter__(self) -> "LoopBoundAgent":
            loop = asyncio.get_running_loop()
            assert self.active_loop is None
            self.active_loop = loop
            self.used_loops.append(loop)
            return self

        async def __aexit__(self, *_exc_info: object) -> None:
            assert asyncio.get_running_loop() is self.active_loop
            self.active_loop = None

    agent = LoopBoundAgent()
    runner = PydanticAgentRunner(agent, object())  # type: ignore[arg-type]

    first = asyncio.run(runner.run("first", (), output_type=str))
    second = asyncio.run(runner.run("second", (), output_type=str))

    assert isinstance(first, TurnSucceeded)
    assert isinstance(second, TurnSucceeded)
    assert len(agent.used_loops) == 2
    assert agent.used_loops[0] is not agent.used_loops[1]
    assert agent.active_loop is None


def test_pydantic_runner_does_not_classify_event_sink_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_module = importlib.import_module("ghostwheel.pydantic_runner")

    async def emit_text(_run: object, sink: object) -> None:
        await runner_module._emit(sink, TextOutput("partial response"))

    async def broken_sink(_event: object) -> None:
        raise RuntimeError("presenter exploded")

    monkeypatch.setattr(runner_module, "stream_agent_run", emit_text)
    result = FakeResult("completed", messages=["new-message"])
    dispatcher = EventDispatcher()
    dispatcher.bind(broken_sink)  # type: ignore[arg-type]
    runner = PydanticAgentRunner(
        FakeAgent(result),  # type: ignore[arg-type]
        object(),
        event_sink=dispatcher,
    )

    with pytest.raises(EventDeliveryError) as raised:
        asyncio.run(runner.run("hello", (), output_type=str))

    assert isinstance(raised.value.cause, RuntimeError)
    assert str(raised.value.cause) == "presenter exploded"
    assert raised.value.event == TextOutput("partial response")


def test_create_tool_deps_maps_resolved_limits(tmp_path: Path) -> None:
    module = importlib.import_module("ghostwheel.agent")
    config = Settings(
        max_output_bytes=123,
        max_read_lines=3,
        max_read_scan_bytes=333,
        max_entries=4,
        max_directory_scan_entries=9,
        max_matches=5,
        bash_timeout_seconds=6,
        max_search_file_bytes=7,
        max_search_total_bytes=70,
        max_search_files=8,
        search_timeout_seconds=1.5,
        regex_timeout_seconds=0.2,
        _env_file=None,
    ).resolve()

    deps = module.create_tool_deps(config, tmp_path)

    assert deps.cwd == tmp_path.resolve()
    assert deps.filesystem_roots == (tmp_path.resolve(),)
    assert deps.limits is config.tools.limits
    assert deps.limits.max_output_bytes == 123
    assert deps.limits.max_read_lines == 3
    assert deps.limits.max_read_scan_bytes == 333
    assert deps.limits.max_entries == 4
    assert deps.limits.max_directory_scan_entries == 9
    assert deps.limits.max_matches == 5
    assert deps.limits.bash_timeout_seconds == 6
    assert deps.limits.max_search_file_bytes == 7
    assert deps.limits.max_search_total_bytes == 70
    assert deps.limits.max_search_files == 8
    assert deps.limits.search_timeout_seconds == 1.5
    assert deps.limits.regex_timeout_seconds == 0.2


def test_compaction_agent_is_tool_free_and_caps_summary_output() -> None:
    agent_module = importlib.import_module("ghostwheel.agent")
    config = Settings(_env_file=None).resolve()

    blueprint = agent_module.compaction_agent_blueprint(config)

    assert blueprint.deps_type is type(None)
    assert blueprint.tools == ()
    assert blueprint.model_settings == {"max_tokens": 2_048, "temperature": 0.1}
    assert blueprint.instructions == agent_module.COMPACTION_INSTRUCTIONS


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


def test_read_sends_compact_text_and_retains_structured_metadata(
    tmp_path: Path,
) -> None:
    (tmp_path / "a").write_text("first\nsecond\n", encoding="utf-8")
    read_tool = Tool(read)
    agent = Agent(
        TestModel(call_tools=["read"]),
        deps_type=ToolDeps,
        tools=(read_tool,),
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
    assert isinstance(tool_returns[0].content, str)
    assert tool_returns[0].content.startswith(
        'read "a" lines=1-2 returned=2 bytes=13 eof=true complete=true reasons=-'
    )
    assert "\n1:first\n2:second" in tool_returns[0].content
    assert '"path":' not in tool_returns[0].content
    assert isinstance(tool_returns[0].metadata, ReadResult)
    assert tool_returns[0].metadata.lines_returned == 2

    function_schema = read_tool.function_schema
    assert function_schema is not None
    properties = function_schema.json_schema["properties"]
    assert {"path", "start_line", "limit", "cursor"} <= properties.keys()
    assert properties["start_line"]["minimum"] == 1
    assert properties["cursor"]["anyOf"][0]["maxLength"] == 4_096


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


def test_grep_sends_compact_text_and_retains_structured_metadata(
    tmp_path: Path,
) -> None:
    (tmp_path / "value.txt").write_text("a value\n", encoding="utf-8")
    grep_tool = Tool(grep)
    agent = Agent(
        TestModel(call_tools=["grep"]),
        deps_type=ToolDeps,
        tools=(grep_tool,),
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
    assert isinstance(tool_returns[0].content, str)
    assert tool_returns[0].content.startswith(
        'grep "." returned=1 scanned=1 searched=1 file_skipped=0 bytes=8 '
        "complete=true reasons=-"
    )
    assert '\nf "value.txt"\n1:1 "a value"' in tool_returns[0].content
    assert '"matches":' not in tool_returns[0].content
    assert isinstance(tool_returns[0].metadata, GrepResult)
    assert [match.file for match in tool_returns[0].metadata.matches] == ["value.txt"]

    function_schema = grep_tool.function_schema
    assert function_schema is not None
    properties = function_schema.json_schema["properties"]
    assert {
        "literal",
        "limit",
        "cursor",
        "show_hidden",
        "include_noise",
    } <= properties.keys()
    assert properties["pattern"]["maxLength"] == 10_000
    assert properties["file_glob"]["maxLength"] == 4_096
    assert function_schema.return_schema == {"type": "string"}


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
