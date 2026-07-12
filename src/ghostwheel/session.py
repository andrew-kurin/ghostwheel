"""Conversational session orchestration.

Runtime outcomes and history mechanics live in their own modules. Imports below
remain public for compatibility with the original session API.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

from ghostwheel.history import (
    CompactionPlan,
    CompactionStats,
    ContextUsage,
    HistoryPolicy,
    HistoryState,
    latest_provider_usage,
    summary_message,
)
from ghostwheel.runtime_contracts import (
    AgentRunner,
    ContextCompactor,
    FailureKind,
    OutputT,
    RunOutcome,
    TurnFailed,
    TurnMessages,
    TurnNoResult,
    TurnOutcome,
    TurnSucceeded,
)
from ghostwheel.token_counting import TokenCountingError

__all__ = [
    "AgentRunner",
    "ChatSession",
    "CompactionPlan",
    "CompactionStats",
    "ContextCompactor",
    "ContextUsage",
    "FailureKind",
    "HistoryPolicy",
    "OutputT",
    "RunOutcome",
    "TurnFailed",
    "TurnMessages",
    "TurnNoResult",
    "TurnOutcome",
    "TurnSucceeded",
    "summary_message",
]


@dataclass(frozen=True, slots=True)
class _CompactionAttempt:
    state: HistoryState
    stats: CompactionStats | None
    failure: TurnFailed | None


class ChatSession:
    """Own chat history and replace old context with rolling LLM summaries."""

    def __init__(
        self,
        runner: AgentRunner,
        *,
        history_policy: HistoryPolicy | None = None,
        compactor: ContextCompactor | None = None,
        initial_overhead_tokens: int = 0,
        initial_history: Sequence[ModelMessage] = (),
    ) -> None:
        if initial_overhead_tokens < 0:
            raise ValueError("initial_overhead_tokens must be non-negative")
        self._runner = runner
        self._compactor = compactor
        self.history_policy = history_policy or HistoryPolicy()
        self._state = HistoryState.initial(initial_history)
        self._minimum_overhead_tokens = initial_overhead_tokens
        self._calibrated_overhead_tokens = initial_overhead_tokens
        self._usage = self.history_policy.estimate_usage(
            self._state.messages,
            overhead_tokens=self._calibrated_overhead_tokens,
        )
        self.last_compaction: CompactionStats | None = None
        self.last_compaction_error: TurnFailed | None = None
        # Compatibility for callers that only need to know whether compaction ran.
        self.last_compacted_turns = 0

    @property
    def history(self) -> tuple[ModelMessage, ...]:
        return self._state.messages

    @property
    def summary(self) -> str | None:
        return self._state.summary

    @property
    def turn_count(self) -> int:
        return len(self._state.turns)

    @property
    def estimated_context_tokens(self) -> int:
        return self._usage.tokens

    @property
    def context_tokens_estimated(self) -> bool:
        return self._usage.estimated

    @property
    def context_window_tokens(self) -> int:
        return self.history_policy.context_window_tokens

    @property
    def compaction_enabled(self) -> bool:
        return self.history_policy.compaction_enabled

    async def send(self, prompt: str) -> TurnOutcome:
        self.last_compaction = None
        self.last_compaction_error = None
        self.last_compacted_turns = 0
        prompt_message = ModelRequest(parts=[UserPromptPart(prompt)])

        try:
            self.history_policy.validate_prompt(prompt_message)
            projected_tokens = self._projected_tokens(prompt_message)
        except (TokenCountingError, ValueError) as error:
            return TurnFailed(error, FailureKind.CONFIGURATION)

        working_state = self._state
        pre_compaction: CompactionStats | None = None
        if self.history_policy.should_compact(projected_tokens):
            attempt = await self._compact_until_ready(
                working_state,
                additional_messages=(prompt_message,),
                before_tokens=self._usage.tokens,
            )
            if attempt.failure is not None:
                self.last_compaction_error = attempt.failure
                return attempt.failure
            working_state = attempt.state
            pre_compaction = attempt.stats

        outcome = await self._runner.run(
            prompt,
            working_state.messages,
            output_type=str,
        )
        if not isinstance(outcome, TurnSucceeded) or not outcome.new_messages:
            return outcome

        candidate_state = working_state.append(outcome.new_messages)
        provider_usage = latest_provider_usage(outcome.new_messages)
        candidate_usage = provider_usage
        if provider_usage is None:
            try:
                candidate_usage = self.history_policy.estimate_usage(
                    candidate_state.messages,
                    overhead_tokens=self._calibrated_overhead_tokens,
                )
            except TokenCountingError:
                # The completed user turn is still valid even if fallback token
                # estimation becomes unavailable after the provider call.
                candidate_usage = self._usage
        else:
            try:
                local_message_tokens = self.history_policy.count_messages(
                    candidate_state.messages
                )
            except TokenCountingError:
                pass
            else:
                # Calibrate the otherwise invisible system prompt, tool schemas,
                # chat template, and tokenizer difference from real usage. The
                # delta is carried into compacted-context estimates until the
                # next normal provider response supplies a fresh measurement.
                self._calibrated_overhead_tokens = max(
                    self._minimum_overhead_tokens,
                    provider_usage.tokens - local_message_tokens,
                )

        self._state = candidate_state
        self._usage = candidate_usage
        # Defer maintenance until the next prompt. This keeps the completed
        # result path cancellation-safe while still guaranteeing compaction
        # before any subsequent provider request crosses the threshold.
        self.last_compaction = pre_compaction
        if self.last_compaction is not None:
            self.last_compacted_turns = self.last_compaction.summarized_turns
        return outcome

    def clear(self) -> None:
        self._state = HistoryState.initial(())
        self._usage = ContextUsage(
            self._calibrated_overhead_tokens,
            estimated=True,
        )
        self.last_compaction = None
        self.last_compaction_error = None
        self.last_compacted_turns = 0

    def _projected_tokens(self, prompt_message: ModelMessage) -> int:
        if self._usage.estimated:
            return (
                self._calibrated_overhead_tokens
                + self.history_policy.count_messages(
                    (*self._state.messages, prompt_message)
                )
            )
        return self._usage.tokens + self.history_policy.count_messages(
            (prompt_message,)
        )

    async def _compact_until_ready(
        self,
        state: HistoryState,
        *,
        additional_messages: Sequence[ModelMessage] = (),
        before_tokens: int,
    ) -> _CompactionAttempt:
        if self._compactor is None:
            return _CompactionAttempt(
                state,
                None,
                TurnFailed(
                    RuntimeError("Context compaction is not configured."),
                    FailureKind.CONFIGURATION,
                ),
            )

        current = state
        first_before = before_tokens
        summarized_messages = 0
        summarized_turns = 0
        recompressed_summary = False
        while True:
            try:
                plan = self.history_policy.plan_compaction(current.turns)
            except TokenCountingError as error:
                return _CompactionAttempt(
                    state,
                    None,
                    TurnFailed(error, FailureKind.CONFIGURATION),
                )
            if (
                plan is None
                and current.summary_message is not None
                and not (recompressed_summary)
            ):
                # A very large pending prompt can leave less room than the
                # configured summary budget. Give the model one chance to
                # recompress the checkpoint itself rather than dead-ending the
                # session after all raw turns have already been folded into it.
                kept_turns = current.turns
                newly_summarized_messages = 1
                newly_summarized_turns = 0
                recompressed_summary = True
                try:
                    target_tokens = self._summary_target_tokens(
                        kept_turns,
                        additional_messages,
                    )
                except TokenCountingError as error:
                    return _CompactionAttempt(
                        state,
                        None,
                        TurnFailed(error, FailureKind.CONFIGURATION),
                    )
                if target_tokens <= 0:
                    return _CompactionAttempt(
                        current,
                        None,
                        TurnFailed(
                            RuntimeError(
                                "The pending prompt leaves no room for the "
                                "conversation summary."
                            ),
                            FailureKind.CONFIGURATION,
                        ),
                    )
                outcome = await self._compactor.summarize(
                    None,
                    (current.summary_message,),
                    target_tokens=target_tokens,
                )
            elif plan is not None:
                kept_turns = plan.kept_turns
                newly_summarized_messages = len(plan.messages_to_summarize)
                newly_summarized_turns = plan.summarized_turns
                try:
                    target_tokens = max(
                        1,
                        self._summary_target_tokens(
                            kept_turns,
                            additional_messages,
                        ),
                    )
                except TokenCountingError as error:
                    return _CompactionAttempt(
                        state,
                        None,
                        TurnFailed(error, FailureKind.CONFIGURATION),
                    )
                outcome = await self._compactor.summarize(
                    current.summary,
                    plan.messages_to_summarize,
                    target_tokens=target_tokens,
                )
            else:
                return _CompactionAttempt(
                    current,
                    None,
                    TurnFailed(
                        RuntimeError(
                            "Context exceeds the compaction threshold, but no "
                            "older messages can be summarized safely."
                        ),
                        FailureKind.CONFIGURATION,
                    ),
                )
            if isinstance(outcome, TurnFailed):
                return _CompactionAttempt(current, None, outcome)
            if isinstance(outcome, TurnNoResult):
                return _CompactionAttempt(
                    current,
                    None,
                    TurnFailed(RuntimeError(outcome.message), FailureKind.PROVIDER),
                )

            current = HistoryState.compacted(outcome.output, kept_turns)
            summarized_messages += newly_summarized_messages
            summarized_turns += newly_summarized_turns
            try:
                projected_after = (
                    self.history_policy.count_messages(
                        (*current.messages, *additional_messages)
                    )
                    + self._calibrated_overhead_tokens
                )
                context_after = (
                    self.history_policy.count_messages(current.messages)
                    + self._calibrated_overhead_tokens
                )
            except TokenCountingError as error:
                return _CompactionAttempt(
                    state,
                    None,
                    TurnFailed(error, FailureKind.CONFIGURATION),
                )

            if not self.history_policy.should_compact(projected_after):
                return _CompactionAttempt(
                    current,
                    CompactionStats(
                        before_tokens=first_before,
                        after_tokens=context_after,
                        summarized_messages=summarized_messages,
                        summarized_turns=summarized_turns,
                    ),
                    None,
                )

    def _summary_target_tokens(
        self,
        kept_turns: Sequence[TurnMessages],
        additional_messages: Sequence[ModelMessage],
    ) -> int:
        empty_summary = HistoryState.compacted("", ()).summary_message
        assert empty_summary is not None
        kept_messages = tuple(message for turn in kept_turns for message in turn)
        fixed_tokens = (
            self._calibrated_overhead_tokens
            + self.history_policy.count_messages(
                (empty_summary, *kept_messages, *additional_messages)
            )
        )
        return min(
            self.history_policy.summary_tokens,
            self.history_policy.compaction_trigger_tokens - fixed_tokens,
        )
