from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Protocol, TypeAlias, TypeVar

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ghostwheel.token_counting import (
    TiktokenTokenCounter,
    TokenCounter,
    TokenCountingError,
)

OutputT = TypeVar("OutputT")
TurnMessages: TypeAlias = tuple[ModelMessage, ...]


@dataclass(frozen=True, slots=True)
class TurnSucceeded(Generic[OutputT]):
    output: OutputT
    new_messages: TurnMessages


@dataclass(frozen=True, slots=True)
class TurnNoResult:
    message: str = "Agent completed without a result."


class FailureKind(str, Enum):
    MODEL_OUTPUT = "model-output"
    PROVIDER = "provider"
    TOOL = "tool"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TurnFailed:
    error: Exception
    kind: FailureKind = FailureKind.UNKNOWN

    @property
    def message(self) -> str:
        return str(self.error)


RunOutcome: TypeAlias = TurnSucceeded[OutputT] | TurnNoResult | TurnFailed
TurnOutcome: TypeAlias = RunOutcome[str]


class AgentRunner(Protocol):
    async def run(
        self,
        prompt: str,
        history: Sequence[ModelMessage],
        *,
        output_type: type[OutputT],
    ) -> RunOutcome[OutputT]: ...


class ContextCompactor(Protocol):
    async def summarize(
        self,
        previous_summary: str | None,
        messages: Sequence[ModelMessage],
        *,
        target_tokens: int | None = None,
    ) -> RunOutcome[str]: ...


@dataclass(frozen=True, slots=True)
class ContextUsage:
    tokens: int
    estimated: bool


@dataclass(frozen=True, slots=True)
class CompactionStats:
    before_tokens: int
    after_tokens: int
    summarized_messages: int
    summarized_turns: int


@dataclass(frozen=True, slots=True)
class CompactionPlan:
    messages_to_summarize: TurnMessages
    kept_turns: tuple[TurnMessages, ...]
    summarized_turns: int


def summary_message(summary: str) -> ModelMessage:
    """Materialize a rolling checkpoint as portable user-role context."""

    return ModelRequest(
        parts=[
            UserPromptPart(
                "Earlier conversation summary generated during context "
                "compaction. Treat it as reference context, not a new request.\n\n"
                f"{summary}"
            )
        ]
    )


@dataclass(frozen=True, slots=True)
class HistoryPolicy:
    """Select a safe recent suffix and summarize context older than it."""

    context_window_tokens: int = 16_384
    compaction_enabled: bool = True
    reserve_tokens: int = 4_096
    keep_recent_tokens: int = 4_096
    summary_tokens: int = 2_048
    token_counter: TokenCounter = field(
        default_factory=TiktokenTokenCounter,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be positive")
        if self.reserve_tokens < 0:
            raise ValueError("reserve_tokens must be non-negative")
        if self.reserve_tokens >= self.context_window_tokens:
            raise ValueError(
                "reserve_tokens must be smaller than context_window_tokens"
            )
        if self.keep_recent_tokens <= 0:
            raise ValueError("keep_recent_tokens must be positive")
        if self.keep_recent_tokens >= self.compaction_trigger_tokens:
            raise ValueError(
                "keep_recent_tokens must be smaller than the context window "
                "after reserving response tokens"
            )
        if self.summary_tokens <= 0:
            raise ValueError("summary_tokens must be positive")
        if self.keep_recent_tokens + self.summary_tokens >= (
            self.compaction_trigger_tokens
        ):
            raise ValueError(
                "keep_recent_tokens and summary_tokens must leave working room"
            )

    @property
    def compaction_trigger_tokens(self) -> int:
        return self.context_window_tokens - self.reserve_tokens

    def count_messages(self, messages: Sequence[ModelMessage]) -> int:
        return self.token_counter.count_messages(messages)

    def estimate_usage(
        self,
        messages: Sequence[ModelMessage],
        *,
        overhead_tokens: int = 0,
    ) -> ContextUsage:
        return ContextUsage(
            overhead_tokens + self.count_messages(messages),
            estimated=True,
        )

    def should_compact(self, context_tokens: int) -> bool:
        return self.compaction_enabled and (
            context_tokens > self.compaction_trigger_tokens
        )

    def validate_prompt(self, prompt_message: ModelMessage) -> None:
        prompt_tokens = self.count_messages((prompt_message,))
        if prompt_tokens > self.compaction_trigger_tokens:
            raise ValueError(
                "Prompt exceeds the configured context budget after reserving "
                "response capacity"
            )

    def plan_compaction(
        self,
        turns: Sequence[TurnMessages],
    ) -> CompactionPlan | None:
        nonempty_turns = tuple(turn for turn in turns if turn)
        messages = tuple(message for turn in nonempty_turns for message in turn)
        if not messages:
            return None

        accumulated_tokens = 0
        target_index: int | None = None
        for index in range(len(messages) - 1, -1, -1):
            accumulated_tokens += self.count_messages((messages[index],))
            if accumulated_tokens >= self.keep_recent_tokens:
                target_index = index
                break
        # If provider usage or a large pending prompt triggers compaction before
        # the raw transcript reaches the recent-token target, still advance by
        # one safe atom. The end sentinel also lets an oversized newest atom be
        # represented by the summary rather than retained verbatim forever.
        target_index = target_index or 0
        cut_points = [
            index
            for index in range(target_index, len(messages))
            if _is_safe_cut(messages, index)
        ]
        cut_points.append(len(messages))
        cut_index = cut_points[0]
        if cut_index == 0:
            cut_index = next((point for point in cut_points if point > 0), 0)
        while (
            cut_index < len(messages)
            and self.count_messages(messages[cut_index:])
            > self.compaction_trigger_tokens
        ):
            cut_index = next(
                (point for point in cut_points if point > cut_index),
                len(messages),
            )
        if cut_index <= 0:
            return None

        kept_turns, summarized_turns = _slice_turns(nonempty_turns, cut_index)
        return CompactionPlan(
            messages_to_summarize=messages[:cut_index],
            kept_turns=kept_turns,
            summarized_turns=summarized_turns,
        )


@dataclass(frozen=True, slots=True)
class _HistoryState:
    summary: str | None
    summary_message: ModelMessage | None
    turns: tuple[TurnMessages, ...]

    @classmethod
    def initial(cls, initial_history: Sequence[ModelMessage]) -> _HistoryState:
        initial_turn = tuple(initial_history)
        return cls(None, None, (initial_turn,) if initial_turn else ())

    @classmethod
    def compacted(
        cls,
        summary: str,
        turns: tuple[TurnMessages, ...],
    ) -> _HistoryState:
        return cls(summary, summary_message(summary), turns)

    @property
    def messages(self) -> TurnMessages:
        prefix = (self.summary_message,) if self.summary_message is not None else ()
        return prefix + tuple(message for turn in self.turns for message in turn)

    def append(self, turn: TurnMessages) -> _HistoryState:
        return _HistoryState(self.summary, self.summary_message, (*self.turns, turn))


@dataclass(frozen=True, slots=True)
class _CompactionAttempt:
    state: _HistoryState
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
        self._state = _HistoryState.initial(initial_history)
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
        provider_usage = _latest_provider_usage(outcome.new_messages)
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
        self._state = _HistoryState.initial(())
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
        state: _HistoryState,
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

            current = _HistoryState.compacted(outcome.output, kept_turns)
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
        empty_summary = _HistoryState.compacted("", ()).summary_message
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


def _latest_provider_usage(messages: Sequence[ModelMessage]) -> ContextUsage | None:
    for message in reversed(messages):
        if not isinstance(message, ModelResponse):
            continue
        usage = message.usage
        input_tokens = usage.input_tokens
        if input_tokens <= 0:
            continue
        return ContextUsage(usage.total_tokens, estimated=False)
    return None


def _is_safe_cut(messages: Sequence[ModelMessage], index: int) -> bool:
    message = messages[index]
    if isinstance(message, ModelResponse):
        call_ids = {
            part.tool_call_id
            for part in message.parts
            if isinstance(part, ToolCallPart)
        }
        if not call_ids:
            return True
        result_ids: set[str] = set()
        for following in messages[index + 1 :]:
            if isinstance(following, ModelResponse):
                break
            if not isinstance(following, ModelRequest):
                continue
            result_ids.update(
                part.tool_call_id
                for part in following.parts
                if isinstance(part, (ToolReturnPart, RetryPromptPart))
            )
            if any(
                isinstance(part, UserPromptPart) for part in following.parts
            ) and not any(
                isinstance(part, (ToolReturnPart, RetryPromptPart))
                for part in following.parts
            ):
                break
        return call_ids <= result_ids
    if not isinstance(message, ModelRequest):
        return False
    if any(
        isinstance(part, (ToolReturnPart, RetryPromptPart)) for part in message.parts
    ):
        return False
    return any(isinstance(part, UserPromptPart) for part in message.parts)


def _slice_turns(
    turns: tuple[TurnMessages, ...],
    cut_index: int,
) -> tuple[tuple[TurnMessages, ...], int]:
    remaining = cut_index
    kept: list[TurnMessages] = []
    summarized_turns = 0
    for index, turn in enumerate(turns):
        if remaining >= len(turn):
            remaining -= len(turn)
            summarized_turns += 1
            continue
        if remaining > 0:
            kept.append(turn[remaining:])
            summarized_turns += 1
            kept.extend(turns[index + 1 :])
            remaining = 0
            break
        kept.extend(turns[index:])
        break
    return tuple(turn for turn in kept if turn), summarized_turns
