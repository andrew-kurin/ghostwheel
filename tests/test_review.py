import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

from ghostwheel.review import (
    RawReview,
    ReviewContextPolicy,
    ReviewFailed,
    ReviewService,
    StructuredReview,
)
from ghostwheel.schemas import Finding, ReviewResult, Severity
from ghostwheel.session import (
    FailureKind,
    RunOutcome,
    TurnFailed,
    TurnNoResult,
    TurnSucceeded,
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


def approved_review() -> ReviewResult:
    return ReviewResult(
        summary="Looks good.",
        findings=[],
        approve=True,
    )


def test_review_uses_fresh_context_and_structured_output() -> None:
    runner = FakeRunner(TurnSucceeded(approved_review(), ()))
    service = ReviewService(runner)
    chat_history = (ModelRequest(parts=[UserPromptPart("unrelated chat")]),)

    outcome = asyncio.run(service.review("src", chat_history=chat_history))

    assert isinstance(outcome, StructuredReview)
    assert runner.calls[0][1] == ()
    assert runner.calls[0][2] is ReviewResult


def test_review_can_explicitly_receive_chat_context() -> None:
    runner = FakeRunner(TurnSucceeded(approved_review(), ()))
    service = ReviewService(
        runner,
        context_policy=ReviewContextPolicy.CHAT_HISTORY,
    )
    history = (ModelRequest(parts=[UserPromptPart("relevant context")]),)

    asyncio.run(service.review("src", chat_history=history))

    assert runner.calls[0][1] == history


def test_review_falls_back_to_raw_prose_without_polluting_chat() -> None:
    runner = FakeRunner(
        TurnFailed(
            RuntimeError("schema unsupported"),
            FailureKind.MODEL_OUTPUT,
        ),
        TurnSucceeded("raw review prose", ()),
    )

    outcome = asyncio.run(ReviewService(runner).review("src"))

    assert isinstance(outcome, RawReview)
    assert outcome.prose == "raw review prose"
    assert [call[2] for call in runner.calls] == [ReviewResult, str]


def test_review_transcribes_raw_fallback_back_to_structured_result() -> None:
    runner = FakeRunner(
        TurnFailed(
            RuntimeError("schema unsupported"),
            FailureKind.MODEL_OUTPUT,
        ),
        TurnSucceeded("raw review prose", ()),
    )
    fallback_runner = FakeRunner(TurnSucceeded(approved_review(), ()))

    outcome = asyncio.run(
        ReviewService(runner, fallback_runner=fallback_runner).review("src")
    )

    assert isinstance(outcome, StructuredReview)
    assert outcome.used_fallback is True
    assert fallback_runner.calls == [("raw review prose", (), ReviewResult)]


def test_review_does_not_repeat_provider_failures() -> None:
    runner = FakeRunner(
        TurnFailed(ConnectionError("server unavailable"), FailureKind.PROVIDER)
    )

    outcome = asyncio.run(ReviewService(runner).review("src"))

    assert isinstance(outcome, ReviewFailed)
    assert len(runner.calls) == 1


def test_review_can_disable_raw_fallback() -> None:
    runner = FakeRunner(TurnNoResult())

    outcome = asyncio.run(ReviewService(runner, raw_fallback=False).review("src"))

    assert isinstance(outcome, ReviewFailed)
    assert len(runner.calls) == 1


def test_review_result_derives_approval_from_findings() -> None:
    finding = Finding(
        file="app.py",
        line=4,
        severity=Severity.WARNING,
        category="bug",
        message="Broken behavior.",
    )

    review = ReviewResult(summary="Not safe.", findings=[finding], approve=True)

    assert review.approve is False


@pytest.mark.parametrize(
    ("line", "line_end"),
    [(None, 5), (10, 9)],
)
def test_finding_rejects_invalid_line_ranges(
    line: int | None,
    line_end: int,
) -> None:
    with pytest.raises(ValidationError):
        Finding(
            file="app.py",
            line=line,
            line_end=line_end,
            severity=Severity.SUGGESTION,
            category="design",
            message="Consider extracting this.",
        )
