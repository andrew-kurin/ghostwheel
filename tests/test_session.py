import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict, fields, replace
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from ghostwheel.history_config import CompactionConfig, HistoryConfig
from ghostwheel.session import (
    ChatSession,
    ContextCompactor,
    FailureKind,
    HistoryPolicy,
    RunOutcome,
    TurnFailed,
    TurnNoResult,
    TurnSucceeded,
)
from ghostwheel.token_counting import TokenCountingError


def request(content: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content)])


def response(content: str, *, usage: RequestUsage | None = None) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content)],
        **({"usage": usage} if usage is not None else {}),
    )


class FakeRunner:
    def __init__(self, *outcomes: RunOutcome[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, tuple[ModelMessage, ...], type[Any]]] = []

    async def run(
        self,
        prompt: str,
        history: Sequence[ModelMessage],
        *,
        output_type: type[Any],
    ) -> RunOutcome[Any]:
        self.calls.append((prompt, tuple(history), output_type))
        return self.outcomes.pop(0)


class FakeCompactor(ContextCompactor):
    def __init__(self, *outcomes: RunOutcome[str]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str | None, tuple[ModelMessage, ...]]] = []
        self.targets: list[int | None] = []

    async def summarize(
        self,
        previous_summary: str | None,
        messages: Sequence[ModelMessage],
        *,
        target_tokens: int | None = None,
    ) -> RunOutcome[str]:
        self.calls.append((previous_summary, tuple(messages)))
        self.targets.append(target_tokens)
        return self.outcomes.pop(0)


class ContentTokenCounter:
    """Treat semantic text and tool arguments as deterministic token counts."""

    def count_messages(self, messages: Sequence[ModelMessage]) -> int:
        tokens = 0
        for model_message in messages:
            for part in model_message.parts:
                content = getattr(part, "content", None)
                if content is not None:
                    tokens += len(str(content))
                elif isinstance(part, ToolCallPart):
                    arguments = (
                        part.args
                        if isinstance(part.args, str)
                        else json.dumps(part.args, sort_keys=True)
                    )
                    tokens += len(arguments)
        return tokens


class UnitTokenCounter:
    def count_messages(self, messages: Sequence[ModelMessage]) -> int:
        return len(messages)


class FailingNonemptyTokenCounter:
    def count_messages(self, messages: Sequence[ModelMessage]) -> int:
        if messages:
            raise TokenCountingError("tokenizer unavailable")
        return 0


def history_policy(
    context_window_tokens: int = 100,
    *,
    reserve_tokens: int = 0,
    keep_recent_tokens: int = 10,
    summary_tokens: int = 1,
    compaction_enabled: bool = True,
    token_counter: object | None = None,
) -> HistoryPolicy:
    return HistoryPolicy(
        context_window_tokens=context_window_tokens,
        compaction_enabled=compaction_enabled,
        reserve_tokens=reserve_tokens,
        keep_recent_tokens=keep_recent_tokens,
        summary_tokens=summary_tokens,
        token_counter=token_counter or ContentTokenCounter(),  # type: ignore[arg-type]
    )


def test_chat_session_owns_canonical_history() -> None:
    first_messages = (request("first request"), response("first response"))
    second_messages = (request("second request"), response("second response"))
    runner = FakeRunner(
        TurnSucceeded("first", first_messages),
        TurnSucceeded("second", second_messages),
    )
    session = ChatSession(runner, history_policy=history_policy(1_000))

    asyncio.run(session.send("one"))
    asyncio.run(session.send("two"))

    assert runner.calls[0] == ("one", (), str)
    assert runner.calls[1] == ("two", first_messages, str)
    assert session.history == first_messages + second_messages


@pytest.mark.parametrize(
    "outcome",
    [TurnNoResult(), TurnFailed(RuntimeError("unavailable"))],
)
def test_failed_turns_do_not_change_history(outcome: object) -> None:
    initial = (request("existing"),)
    runner = FakeRunner(outcome)  # type: ignore[arg-type]
    session = ChatSession(
        runner,
        history_policy=history_policy(100),
        initial_history=initial,
    )

    asyncio.run(session.send("next"))

    assert runner.calls[0][1] == initial
    assert session.history == initial
    assert session.last_compaction is None


def test_history_has_no_turn_or_message_count_cap() -> None:
    turns = tuple((request("x"),) for _index in range(250))
    runner = FakeRunner(*(TurnSucceeded("ok", turn) for turn in turns))
    session = ChatSession(
        runner,
        history_policy=history_policy(
            1_000,
            keep_recent_tokens=100,
            token_counter=UnitTokenCounter(),
        ),
    )

    for _index in range(250):
        asyncio.run(session.send("p"))

    assert session.turn_count == 250
    assert len(session.history) == 250
    assert session.last_compaction is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"context_window_tokens": 0}, "must be positive"),
        (
            {
                "context_window_tokens": 10,
                "reserve_tokens": 10,
                "keep_recent_tokens": 1,
            },
            "must be smaller",
        ),
        (
            {
                "context_window_tokens": 10,
                "reserve_tokens": 0,
                "keep_recent_tokens": 0,
            },
            "must be positive",
        ),
        (
            {
                "context_window_tokens": 10,
                "reserve_tokens": 0,
                "keep_recent_tokens": 10,
            },
            "must leave room for a summary",
        ),
        (
            {
                "context_window_tokens": 10,
                "reserve_tokens": 0,
                "keep_recent_tokens": 3,
                "summary_tokens": 0,
            },
            "must be positive",
        ),
        (
            {
                "context_window_tokens": 10,
                "reserve_tokens": 0,
                "keep_recent_tokens": 5,
                "summary_tokens": 5,
            },
            "must leave working room",
        ),
    ],
)
def test_history_policy_validates_token_settings(
    kwargs: dict[str, int],
    message: str,
) -> None:
    settings = {"summary_tokens": 1, **kwargs}
    with pytest.raises(ValueError, match=message):
        HistoryPolicy(
            token_counter=UnitTokenCounter(),
            **settings,
        )


def test_history_policy_accepts_canonical_config_and_preserves_flat_accessors() -> None:
    config = HistoryConfig(
        context_window_tokens=10,
        compaction=CompactionConfig(
            enabled=False,
            reserve_tokens=2,
            keep_recent_tokens=3,
            summary_tokens=1,
        ),
    )
    counter = UnitTokenCounter()

    policy = HistoryPolicy.from_config(config, token_counter=counter)

    assert policy.config is config
    assert policy.token_counter is counter
    assert policy.context_window_tokens == 10
    assert policy.compaction_enabled is False
    assert policy.reserve_tokens == 2
    assert policy.keep_recent_tokens == 3
    assert policy.summary_tokens == 1
    assert policy.compaction_trigger_tokens == 8


def test_history_policy_flat_values_override_canonical_configuration() -> None:
    config = HistoryConfig()

    policy = HistoryPolicy(
        config=config,
        keep_recent_tokens=3_000,
        summary_tokens=1_500,
    )

    assert policy.config is not config
    assert policy.context_window_tokens == config.context_window_tokens
    assert policy.keep_recent_tokens == 3_000
    assert policy.summary_tokens == 1_500


def test_history_policy_preserves_legacy_positional_construction() -> None:
    counter = UnitTokenCounter()

    policy = HistoryPolicy(10, False, 2, 3, 1, counter)

    assert policy.context_window_tokens == 10
    assert policy.compaction_enabled is False
    assert policy.reserve_tokens == 2
    assert policy.keep_recent_tokens == 3
    assert policy.summary_tokens == 1
    assert policy.token_counter is counter


def test_history_policy_dataclass_replace_accepts_flat_fields() -> None:
    counter = UnitTokenCounter()
    policy = HistoryPolicy(10, True, 2, 3, 1, counter)

    updated = replace(policy, keep_recent_tokens=2, summary_tokens=2)

    assert updated.config is not policy.config
    assert updated.context_window_tokens == 10
    assert updated.compaction_enabled is True
    assert updated.reserve_tokens == 2
    assert updated.keep_recent_tokens == 2
    assert updated.summary_tokens == 2
    assert updated.token_counter is counter


def test_history_policy_preserves_legacy_dataclass_reflection_contract() -> None:
    counter = UnitTokenCounter()
    policy = HistoryPolicy(10, False, 2, 3, 1, counter)

    assert [policy_field.name for policy_field in fields(HistoryPolicy)] == [
        "context_window_tokens",
        "compaction_enabled",
        "reserve_tokens",
        "keep_recent_tokens",
        "summary_tokens",
        "token_counter",
    ]
    assert HistoryPolicy.__match_args__ == (
        "context_window_tokens",
        "compaction_enabled",
        "reserve_tokens",
        "keep_recent_tokens",
        "summary_tokens",
        "token_counter",
    )
    serialized = asdict(policy)
    assert list(serialized) == list(HistoryPolicy.__match_args__)
    assert serialized["context_window_tokens"] == 10
    assert serialized["compaction_enabled"] is False
    assert "config" not in serialized
    assert repr(policy) == (
        "HistoryPolicy(context_window_tokens=10, compaction_enabled=False, "
        "reserve_tokens=2, keep_recent_tokens=3, summary_tokens=1)"
    )

    match policy:
        case HistoryPolicy(10, False, 2, 3, 1, matched_counter):
            assert matched_counter is counter
        case _:
            pytest.fail("legacy positional pattern did not match")


def test_history_policy_equality_and_hash_ignore_token_counter_and_config() -> None:
    first = HistoryPolicy(10, False, 2, 3, 1, UnitTokenCounter())
    canonical = HistoryConfig(
        context_window_tokens=10,
        compaction=CompactionConfig(
            enabled=False,
            reserve_tokens=2,
            keep_recent_tokens=3,
            summary_tokens=1,
        ),
    )
    second = HistoryPolicy.from_config(canonical, token_counter=ContentTokenCounter())

    assert first == second
    assert hash(first) == hash(second)
    assert first.config is not second.config
    assert second.config is canonical


def test_compaction_threshold_is_strictly_greater_than_available_context() -> None:
    policy = history_policy(10, reserve_tokens=2, keep_recent_tokens=3)

    assert policy.compaction_trigger_tokens == 8
    assert policy.should_compact(8) is False
    assert policy.should_compact(9) is True
    policy.validate_prompt(request("x" * 8))
    with pytest.raises(ValueError, match="Prompt exceeds"):
        policy.validate_prompt(request("x" * 9))


def test_exact_threshold_does_not_invoke_compactor() -> None:
    initial = (request("old!"),)
    runner = FakeRunner(TurnNoResult())
    compactor = FakeCompactor()
    session = ChatSession(
        runner,
        compactor=compactor,
        history_policy=history_policy(10, reserve_tokens=2, keep_recent_tokens=3),
        initial_history=initial,
    )

    asyncio.run(session.send("next"))

    assert runner.calls[0][1] == initial
    assert compactor.calls == []


def test_plan_compaction_keeps_a_maximal_whole_turn_suffix() -> None:
    first = (request("aaaa"), response("bbbb"))
    second = (request("cccc"), response("dddd"))
    policy = history_policy(100, keep_recent_tokens=8)

    plan = policy.plan_compaction((first, second))

    assert plan is not None
    assert plan.messages_to_summarize == first
    assert plan.kept_turns == (second,)
    assert plan.summarized_turns == 1


def _tool_turn(result_message: ModelRequest) -> tuple[ModelMessage, ...]:
    call_id = "call-1"
    return (
        request("start"),
        ModelResponse(
            parts=[
                TextPart("abc"),
                ToolCallPart("read", {}, tool_call_id=call_id),
            ]
        ),
        result_message,
        response("zz"),
    )


def test_plan_compaction_can_split_at_assistant_and_keeps_tool_pair() -> None:
    call_id = "call-1"
    result = ModelRequest(parts=[ToolReturnPart("read", "rrrr", tool_call_id=call_id)])
    turn = _tool_turn(result)
    policy = history_policy(100, keep_recent_tokens=9)

    plan = policy.plan_compaction((turn,))

    assert plan is not None
    assert plan.messages_to_summarize == turn[:1]
    assert plan.kept_turns == (turn[1:],)
    assert plan.kept_turns[0][0] is turn[1]
    assert plan.kept_turns[0][1] is result


@pytest.mark.parametrize(
    "result_parts",
    [
        [ToolReturnPart("read", "rrrr", tool_call_id="call-1")],
        [
            RetryPromptPart(
                "rrrr",
                tool_name="read",
                tool_call_id="call-1",
            )
        ],
        [
            ToolReturnPart("read", "rrrr", tool_call_id="call-1"),
            UserPromptPart("extra"),
        ],
    ],
)
def test_plan_compaction_never_starts_at_tool_or_retry_result(
    result_parts: list[object],
) -> None:
    result = ModelRequest(parts=result_parts)  # type: ignore[arg-type]
    turn = _tool_turn(result)
    policy = history_policy(100, keep_recent_tokens=6)

    plan = policy.plan_compaction((turn,))

    assert plan is not None
    assert plan.messages_to_summarize == turn[:3]
    assert plan.kept_turns == ((turn[3],),)


def test_plan_compaction_summarizes_oversized_newest_message_at_end_sentinel() -> None:
    turn = (request("start"), response("x" * 120))
    policy = history_policy(100, keep_recent_tokens=5)

    plan = policy.plan_compaction((turn,))

    assert plan is not None
    assert plan.messages_to_summarize == turn
    assert plan.kept_turns == ()


def test_rolling_summary_is_replaced_and_receives_only_newly_evicted_messages() -> None:
    initial = tuple(request(f"old-{index}") for index in range(4))
    runner = FakeRunner(
        TurnSucceeded("one", (request("new-1"),)),
        TurnSucceeded("two", (request("new-2"),)),
    )
    compactor = FakeCompactor(
        TurnSucceeded("summary-1", ()),
        TurnSucceeded("summary-2", ()),
    )
    session = ChatSession(
        runner,
        compactor=compactor,
        history_policy=history_policy(
            5,
            reserve_tokens=1,
            keep_recent_tokens=2,
            token_counter=UnitTokenCounter(),
        ),
        initial_history=initial,
    )

    asyncio.run(session.send("first"))
    asyncio.run(session.send("second"))

    assert compactor.calls == [
        (None, initial[:2]),
        ("summary-1", initial[2:3]),
    ]
    assert session.summary == "summary-2"
    summary_content = str(session.history[0].parts[0].content)
    assert "summary-2" in summary_content
    assert "summary-1" not in summary_content
    assert len(session.history) == 4


@pytest.mark.parametrize(
    "outcome",
    [TurnNoResult(), TurnFailed(RuntimeError("provider failed"))],
)
def test_pre_compaction_is_transactional_when_main_turn_does_not_succeed(
    outcome: RunOutcome[str],
) -> None:
    initial = tuple(request(f"old-{index}") for index in range(4))
    runner = FakeRunner(outcome)
    compactor = FakeCompactor(TurnSucceeded("candidate summary", ()))
    session = ChatSession(
        runner,
        compactor=compactor,
        history_policy=history_policy(
            5,
            reserve_tokens=1,
            keep_recent_tokens=2,
            token_counter=UnitTokenCounter(),
        ),
        initial_history=initial,
    )

    result = asyncio.run(session.send("next"))

    assert result is outcome
    assert compactor.calls == [(None, initial[:2])]
    assert len(runner.calls[0][1]) == 3
    assert session.history == initial
    assert session.summary is None
    assert session.last_compaction is None


def test_pre_compaction_failure_does_not_call_main_runner_or_mutate_history() -> None:
    initial = tuple(request(f"old-{index}") for index in range(4))
    failure = TurnFailed(RuntimeError("summarizer unavailable"), FailureKind.PROVIDER)
    runner = FakeRunner()
    compactor = FakeCompactor(failure)
    session = ChatSession(
        runner,
        compactor=compactor,
        history_policy=history_policy(
            5,
            reserve_tokens=1,
            keep_recent_tokens=2,
            token_counter=UnitTokenCounter(),
        ),
        initial_history=initial,
    )

    result = asyncio.run(session.send("next"))

    assert result is failure
    assert runner.calls == []
    assert session.history == initial
    assert session.last_compaction_error is failure


def test_over_threshold_completed_turn_defers_compaction_until_next_prompt() -> None:
    usage = RequestUsage(input_tokens=3, output_tokens=1)
    completed = (request("user"), request("middle"), response("done", usage=usage))
    runner = FakeRunner(TurnSucceeded("done", completed))
    compactor = FakeCompactor(TurnSucceeded("post summary", ()))
    session = ChatSession(
        runner,
        compactor=compactor,
        history_policy=history_policy(
            4,
            reserve_tokens=1,
            keep_recent_tokens=1,
            token_counter=UnitTokenCounter(),
        ),
    )

    outcome = asyncio.run(session.send("go"))

    assert isinstance(outcome, TurnSucceeded)
    assert compactor.calls == []
    assert session.history == completed
    assert session.summary is None
    assert session.last_compaction is None
    assert session.context_tokens_estimated is False
    assert session.estimated_context_tokens == 4


def test_deferred_compaction_runs_before_the_next_provider_request() -> None:
    usage = RequestUsage(input_tokens=3, output_tokens=1)
    completed = (request("user"), request("middle"), response("done", usage=usage))
    next_turn = (request("next"), response("ok", usage=RequestUsage(input_tokens=2)))
    runner = FakeRunner(
        TurnSucceeded("done", completed),
        TurnSucceeded("ok", next_turn),
    )
    compactor = FakeCompactor(
        TurnSucceeded("rolling summary", ()),
        TurnSucceeded("rolling summary 2", ()),
    )
    session = ChatSession(
        runner,
        compactor=compactor,
        history_policy=history_policy(
            4,
            reserve_tokens=1,
            keep_recent_tokens=1,
            token_counter=UnitTokenCounter(),
        ),
    )

    asyncio.run(session.send("go"))
    outcome = asyncio.run(session.send("n"))

    assert isinstance(outcome, TurnSucceeded)
    assert compactor.calls == [
        (None, completed[:2]),
        ("rolling summary", completed[2:]),
    ]
    assert compactor.targets == [1, 1]
    assert len(runner.calls) == 2
    assert len(runner.calls[1][1]) == 1
    assert session.summary == "rolling summary 2"
    assert session.last_compaction is not None
    assert session.last_compaction.summarized_messages == 3


def test_deferred_compaction_failure_keeps_history_and_skips_next_provider_call() -> (
    None
):
    usage = RequestUsage(input_tokens=3, output_tokens=1)
    completed = (request("user"), request("middle"), response("done", usage=usage))
    runner = FakeRunner(TurnSucceeded("done", completed))
    failure = TurnFailed(RuntimeError("summarizer failed"), FailureKind.PROVIDER)
    session = ChatSession(
        runner,
        compactor=FakeCompactor(failure),
        history_policy=history_policy(
            4,
            reserve_tokens=1,
            keep_recent_tokens=1,
            token_counter=UnitTokenCounter(),
        ),
    )

    asyncio.run(session.send("go"))
    outcome = asyncio.run(session.send("n"))

    assert outcome is failure
    assert len(runner.calls) == 1
    assert session.history == completed
    assert session.summary is None


def test_provider_usage_is_preferred_over_local_estimate() -> None:
    usage = RequestUsage(
        input_tokens=10,
        cache_read_tokens=2,
        cache_write_tokens=1,
        output_tokens=3,
    )
    completed = (request("user"), response("assistant", usage=usage))
    runner = FakeRunner(TurnSucceeded("done", completed))
    session = ChatSession(
        runner,
        history_policy=history_policy(
            100,
            keep_recent_tokens=10,
            token_counter=UnitTokenCounter(),
        ),
    )

    asyncio.run(session.send("go"))

    assert session.estimated_context_tokens == 13
    assert session.context_tokens_estimated is False


def test_provider_measurement_can_replace_a_larger_seeded_estimate() -> None:
    completed = (
        request("user"),
        response("assistant", usage=RequestUsage(input_tokens=2, output_tokens=1)),
    )
    session = ChatSession(
        FakeRunner(TurnSucceeded("done", completed)),
        history_policy=history_policy(
            100,
            token_counter=UnitTokenCounter(),
        ),
        initial_overhead_tokens=7,
    )

    assert session.estimated_context_tokens == 7
    assert session.context_tokens_estimated is True

    asyncio.run(session.send("go"))

    assert session.estimated_context_tokens == 3
    assert session.context_tokens_estimated is False

    session.clear()

    assert session.estimated_context_tokens == 7
    assert session.context_tokens_estimated is True


def test_zero_provider_usage_falls_back_to_local_estimate() -> None:
    completed = (request("user"), response("assistant", usage=RequestUsage()))
    runner = FakeRunner(TurnSucceeded("done", completed))
    session = ChatSession(
        runner,
        history_policy=history_policy(
            100,
            keep_recent_tokens=10,
            token_counter=UnitTokenCounter(),
        ),
    )

    asyncio.run(session.send("go"))

    assert session.estimated_context_tokens == 2
    assert session.context_tokens_estimated is True


def test_prompt_larger_than_available_context_fails_before_provider_call() -> None:
    runner = FakeRunner()
    session = ChatSession(
        runner,
        history_policy=history_policy(10, reserve_tokens=2, keep_recent_tokens=3),
    )

    outcome = asyncio.run(session.send("x" * 9))

    assert isinstance(outcome, TurnFailed)
    assert outcome.kind is FailureKind.CONFIGURATION
    assert runner.calls == []


def test_tokenizer_loading_failure_is_a_configuration_failure() -> None:
    runner = FakeRunner()
    session = ChatSession(
        runner,
        history_policy=history_policy(
            100,
            keep_recent_tokens=10,
            token_counter=FailingNonemptyTokenCounter(),
        ),
    )

    outcome = asyncio.run(session.send("hello"))

    assert isinstance(outcome, TurnFailed)
    assert outcome.kind is FailureKind.CONFIGURATION
    assert runner.calls == []


def test_clear_removes_raw_history_summary_and_compaction_metadata() -> None:
    initial = tuple(request(f"old-{index}") for index in range(4))
    runner = FakeRunner(TurnSucceeded("one", (request("new"),)))
    compactor = FakeCompactor(TurnSucceeded("summary", ()))
    session = ChatSession(
        runner,
        compactor=compactor,
        history_policy=history_policy(
            5,
            reserve_tokens=1,
            keep_recent_tokens=2,
            token_counter=UnitTokenCounter(),
        ),
        initial_history=initial,
    )
    asyncio.run(session.send("next"))

    session.clear()

    assert session.history == ()
    assert session.summary is None
    assert session.turn_count == 0
    assert session.estimated_context_tokens == 0
    assert session.last_compaction is None
    assert session.last_compaction_error is None


def test_clear_retains_static_request_overhead_estimate() -> None:
    session = ChatSession(
        FakeRunner(),
        history_policy=history_policy(100),
        initial_overhead_tokens=7,
    )

    session.clear()

    assert session.estimated_context_tokens == 7
    assert session.context_tokens_estimated is True


def test_clear_retains_provider_calibrated_request_overhead() -> None:
    completed = (
        request("user"),
        response("assistant", usage=RequestUsage(input_tokens=8, output_tokens=2)),
    )
    session = ChatSession(
        FakeRunner(TurnSucceeded("done", completed)),
        history_policy=history_policy(
            100,
            token_counter=UnitTokenCounter(),
        ),
    )
    asyncio.run(session.send("go"))

    session.clear()

    assert session.estimated_context_tokens == 8
    assert session.context_tokens_estimated is True


def test_provider_calibration_never_lowers_seeded_static_overhead() -> None:
    completed = (
        request("user"),
        response("assistant", usage=RequestUsage(input_tokens=2, output_tokens=1)),
    )
    session = ChatSession(
        FakeRunner(TurnSucceeded("done", completed)),
        history_policy=history_policy(
            100,
            token_counter=UnitTokenCounter(),
        ),
        initial_overhead_tokens=7,
    )
    asyncio.run(session.send("go"))

    session.clear()

    assert session.estimated_context_tokens == 7
