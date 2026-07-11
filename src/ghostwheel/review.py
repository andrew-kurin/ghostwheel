from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from pydantic_ai.messages import ModelMessage

from ghostwheel.schemas import ReviewResult
from ghostwheel.session import (
    AgentRunner,
    FailureKind,
    TurnFailed,
    TurnNoResult,
    TurnSucceeded,
)

REVIEW_PROMPT = (
    "Perform a careful code review of the files at {paths}. "
    "Use your tools to read each file. Identify real issues: bugs, security "
    "concerns, design problems, and dead code. Do not flag stylistic nits. "
    "For every finding, preserve the exact file and the full line range. "
    "Warnings and blockers require changes; suggestions do not prevent approval."
)


class ReviewContextPolicy(str, Enum):
    FRESH = "fresh"
    CHAT_HISTORY = "chat-history"


@dataclass(frozen=True, slots=True)
class StructuredReview:
    review: ReviewResult
    used_fallback: bool = False


@dataclass(frozen=True, slots=True)
class RawReview:
    prose: str
    structured_failure: str


@dataclass(frozen=True, slots=True)
class ReviewFailed:
    message: str


ReviewOutcome: TypeAlias = StructuredReview | RawReview | ReviewFailed


class ReviewService:
    """Run focused reviews independently from conversational session state."""

    def __init__(
        self,
        runner: AgentRunner,
        *,
        context_policy: ReviewContextPolicy = ReviewContextPolicy.FRESH,
        raw_fallback: bool = True,
        fallback_runner: AgentRunner | None = None,
    ) -> None:
        self._runner = runner
        self.context_policy = context_policy
        self.raw_fallback = raw_fallback
        self._fallback_runner = fallback_runner

    async def review(
        self,
        paths: str = ".",
        *,
        chat_history: Sequence[ModelMessage] = (),
    ) -> ReviewOutcome:
        target_paths = paths.strip() or "."
        prompt = REVIEW_PROMPT.format(paths=target_paths)
        history = (
            tuple(chat_history)
            if self.context_policy is ReviewContextPolicy.CHAT_HISTORY
            else ()
        )

        structured = await self._runner.run(
            prompt,
            history,
            output_type=ReviewResult,
        )
        if isinstance(structured, TurnSucceeded):
            return StructuredReview(structured.output)

        failure = _failure_message(structured)
        if not self.raw_fallback or not _supports_raw_fallback(structured):
            return ReviewFailed(failure)

        raw = await self._runner.run(prompt, history, output_type=str)
        if isinstance(raw, TurnSucceeded):
            if self._fallback_runner is not None:
                transcribed = await self._fallback_runner.run(
                    raw.output,
                    (),
                    output_type=ReviewResult,
                )
                if isinstance(transcribed, TurnSucceeded):
                    return StructuredReview(transcribed.output, used_fallback=True)
                failure = (
                    f"{failure}. Fallback transcription failed: "
                    f"{_failure_message(transcribed)}"
                )
            return RawReview(raw.output, failure)

        return ReviewFailed(
            f"Structured review failed: {failure}. "
            f"Raw review also failed: {_failure_message(raw)}"
        )


def _failure_message(outcome: TurnNoResult | TurnFailed) -> str:
    return outcome.message


def _supports_raw_fallback(outcome: TurnNoResult | TurnFailed) -> bool:
    return isinstance(outcome, TurnNoResult) or outcome.kind is FailureKind.MODEL_OUTPUT
