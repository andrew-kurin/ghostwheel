import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

from pydantic_ai import Agent
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    ToolRetryError,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)

from ghostwheel.events import (
    AgentEvent,
    TextOutput,
    ThinkingOutput,
    ToolFailed,
    ToolFinished,
    ToolStarted,
)
from ghostwheel.session import (
    FailureKind,
    RunOutcome,
    TurnFailed,
    TurnNoResult,
    TurnSucceeded,
)

OutputT = TypeVar("OutputT")
EventSink = Callable[[AgentEvent], Awaitable[None] | None]


class PydanticAgentRunner:
    """Adapt Pydantic AI runs to application outcomes and neutral events."""

    def __init__(
        self,
        agent: Agent[Any, Any],
        deps: Any,
        *,
        event_sink: EventSink | None = None,
    ) -> None:
        self._agent = agent
        self._deps = deps
        self._event_sink = event_sink

    async def run(
        self,
        prompt: str,
        history: Sequence[ModelMessage],
        *,
        output_type: type[OutputT],
    ) -> RunOutcome[OutputT]:
        try:
            async with self._agent.iter(
                prompt,
                message_history=history,
                deps=self._deps,
                output_type=output_type,
            ) as run:
                await stream_agent_run(run, self._event_sink)

            if run.result is None:
                return TurnNoResult()

            return TurnSucceeded(
                output=run.result.output,
                new_messages=tuple(run.result.new_messages()),
            )
        except Exception as error:
            return TurnFailed(error, _failure_kind(error))


async def stream_agent_run(run: Any, sink: EventSink | None = None) -> None:
    """Consume a Pydantic AI run and expose only framework-neutral events."""

    try:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    try:
                        async for event in stream:
                            await _handle_model_event(event, sink)
                    except StopAsyncIteration:
                        pass
            elif Agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        await _handle_tool_event(event, sink)
    except StopAsyncIteration:
        pass


async def _handle_model_event(event: Any, sink: EventSink | None) -> None:
    if isinstance(event, PartStartEvent):
        part = event.part
        if isinstance(part, ThinkingPart):
            await _emit(sink, ThinkingOutput(part.content, starts_part=True))
        elif isinstance(part, TextPart):
            await _emit(sink, TextOutput(part.content, starts_part=True))
    elif isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, ThinkingPartDelta):
            content = event.delta.content_delta
            if content:
                await _emit(sink, ThinkingOutput(content))
        elif isinstance(event.delta, TextPartDelta):
            await _emit(sink, TextOutput(event.delta.content_delta))


async def _handle_tool_event(event: Any, sink: EventSink | None) -> None:
    if isinstance(event, FunctionToolCallEvent):
        part = event.part
        if isinstance(part, ToolCallPart):
            await _emit(
                sink,
                ToolStarted(
                    part.tool_name,
                    str(part.args),
                    call_id=part.tool_call_id,
                ),
            )
    elif isinstance(event, FunctionToolResultEvent):
        # Pydantic AI 1.x called this field ``result``; 2.x renamed it ``part``.
        result = getattr(event, "result", None)
        if result is None:
            result = event.part
        if isinstance(result, ToolReturnPart):
            await _emit(
                sink,
                ToolFinished(
                    result.tool_name,
                    str(result.content),
                    call_id=result.tool_call_id,
                ),
            )
        elif isinstance(result, RetryPromptPart):
            await _emit(
                sink,
                ToolFailed(
                    result.tool_name or "tool",
                    str(result.content),
                    call_id=result.tool_call_id,
                ),
            )


async def _emit(sink: EventSink | None, event: AgentEvent) -> None:
    if sink is None:
        return
    emitted = sink(event)
    if inspect.isawaitable(emitted):
        await emitted


def _failure_kind(error: Exception) -> FailureKind:
    if isinstance(error, ToolRetryError):
        return FailureKind.TOOL
    if isinstance(error, ModelHTTPError) and _is_structured_output_error(error):
        return FailureKind.MODEL_OUTPUT
    if isinstance(error, ModelAPIError):
        return FailureKind.PROVIDER
    if isinstance(error, UserError):
        if _is_structured_output_error(error):
            return FailureKind.MODEL_OUTPUT
        return FailureKind.CONFIGURATION
    if isinstance(error, UnexpectedModelBehavior):
        message = str(error).lower()
        if "tool" in message:
            return FailureKind.TOOL
        if _is_structured_output_error(error):
            return FailureKind.MODEL_OUTPUT
    return FailureKind.UNKNOWN


def _is_structured_output_error(error: Exception) -> bool:
    detail = f"{error} {getattr(error, 'body', '')}".lower()
    markers = (
        "json schema",
        "json_schema",
        "output validation",
        "output mode",
        "output not supported",
        "response_format",
        "structured output",
    )
    return any(marker in detail for marker in markers)
