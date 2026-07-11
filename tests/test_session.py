import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    UserPromptPart,
)

from ghostwheel.session import (
    ChatSession,
    HistoryPolicy,
    RunOutcome,
    TurnFailed,
    TurnNoResult,
    TurnSucceeded,
)


def message(content: str) -> ModelMessage:
    return ModelRequest(parts=[UserPromptPart(content)])


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


def test_chat_session_owns_canonical_history() -> None:
    first_messages = (message("first request"), message("first response"))
    second_messages = (message("second request"), message("second response"))
    runner = FakeRunner(
        TurnSucceeded("first", first_messages),
        TurnSucceeded("second", second_messages),
    )
    session = ChatSession(runner)

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
    initial = (message("existing"),)
    session = ChatSession(FakeRunner(outcome), initial_history=initial)  # type: ignore[arg-type]

    asyncio.run(session.send("next"))

    assert session.history == initial


def test_history_policy_compacts_whole_turns_and_reports_it() -> None:
    first = (message("first"),)
    second = (message("second"),)
    session = ChatSession(
        FakeRunner(TurnSucceeded("one", first), TurnSucceeded("two", second)),
        history_policy=HistoryPolicy(
            max_turns=1,
            max_messages=10,
            max_bytes=10_000,
            response_reserve_bytes=0,
        ),
    )

    asyncio.run(session.send("one"))
    assert session.last_compacted_turns == 0
    asyncio.run(session.send("two"))

    assert session.history == second
    assert session.last_compacted_turns == 1


def test_history_policy_enforces_serialized_byte_budget() -> None:
    oversized = (message("a message larger than one byte"),)
    prompt = "one"
    prompt_bytes = len(
        ModelMessagesTypeAdapter.dump_json(
            [ModelRequest(parts=[UserPromptPart(prompt)])]
        )
    )
    session = ChatSession(
        FakeRunner(TurnSucceeded("ok", oversized)),
        history_policy=HistoryPolicy(
            max_turns=10,
            max_messages=10,
            max_bytes=prompt_bytes,
            response_reserve_bytes=0,
        ),
    )

    asyncio.run(session.send(prompt))

    assert session.history == ()
    assert session.last_compacted_turns == 1


@pytest.mark.parametrize("field", ["max_turns", "max_messages", "max_bytes"])
def test_history_policy_limits_must_be_positive(field: str) -> None:
    values = {
        "max_turns": 1,
        "max_messages": 1,
        "max_bytes": 1,
        "response_reserve_bytes": 0,
        field: 0,
    }
    with pytest.raises(ValueError, match="must be positive"):
        HistoryPolicy(**values)


def test_oversized_new_turn_does_not_delete_valid_older_history() -> None:
    older = (message("older"),)
    oversized = (message("x" * 1_000),)
    policy = HistoryPolicy(
        max_turns=10,
        max_messages=10,
        max_bytes=500,
        response_reserve_bytes=0,
    )
    session = ChatSession(
        FakeRunner(TurnSucceeded("large", oversized)),
        history_policy=policy,
        initial_history=older,
    )

    asyncio.run(session.send("next"))

    assert session.history == older
    assert session.last_compacted_turns == 1


def test_history_is_compacted_before_provider_call_with_response_reserve() -> None:
    older = (message("x" * 150),)
    prompt = "y" * 250
    older_bytes = len(ModelMessagesTypeAdapter.dump_json(list(older)))
    prompt_bytes = len(
        ModelMessagesTypeAdapter.dump_json(
            [ModelRequest(parts=[UserPromptPart(prompt)])]
        )
    )
    reserve = 100
    runner = FakeRunner(TurnNoResult())
    session = ChatSession(
        runner,
        history_policy=HistoryPolicy(
            max_turns=10,
            max_messages=10,
            max_bytes=older_bytes + prompt_bytes + reserve - 1,
            response_reserve_bytes=reserve,
        ),
        initial_history=older,
    )

    asyncio.run(session.send(prompt))

    assert runner.calls[0][1] == ()
    assert session.last_compacted_turns == 1


def test_prompt_larger_than_context_budget_fails_before_provider_call() -> None:
    runner = FakeRunner()
    session = ChatSession(
        runner,
        history_policy=HistoryPolicy(
            max_turns=10,
            max_messages=10,
            max_bytes=100,
            response_reserve_bytes=20,
        ),
    )

    outcome = asyncio.run(session.send("x" * 81))

    assert isinstance(outcome, TurnFailed)
    assert runner.calls == []
