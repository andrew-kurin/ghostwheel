"""Conversation history state, compaction policy, and message boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ghostwheel.history_config import (
    DEFAULT_HISTORY_CONFIG,
    CompactionConfig,
    HistoryConfig,
)
from ghostwheel.runtime_contracts import TurnMessages
from ghostwheel.token_counting import TiktokenTokenCounter, TokenCounter

ConversationAtom: TypeAlias = tuple[ModelMessage, ...]


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


def conversation_atoms(
    messages: Sequence[ModelMessage],
) -> tuple[ConversationAtom, ...]:
    """Group messages that must remain together across history boundaries.

    A model response containing tool calls and its immediately following tool
    result or retry requests form one atom. Both context slicing and summary
    chunking use this representation, so neither can split a call from the
    provider messages that resolve it.
    """

    atoms: list[ConversationAtom] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        atom = [message]
        index += 1
        if isinstance(message, ModelResponse) and any(
            isinstance(part, ToolCallPart) for part in message.parts
        ):
            while index < len(messages):
                following = messages[index]
                if not _is_tool_result_request(following):
                    break
                atom.append(following)
                index += 1
        atoms.append(tuple(atom))
    return tuple(atoms)


def safe_cut_indices(messages: Sequence[ModelMessage]) -> tuple[int, ...]:
    """Return message offsets where a self-contained suffix may begin."""

    indices: list[int] = []
    offset = 0
    for atom in conversation_atoms(messages):
        if _can_start_context(atom):
            indices.append(offset)
        offset += len(atom)
    return tuple(indices)


def _is_tool_result_request(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and any(
        isinstance(part, (ToolReturnPart, RetryPromptPart)) for part in message.parts
    )


def _can_start_context(atom: ConversationAtom) -> bool:
    message = atom[0]
    if isinstance(message, ModelResponse):
        call_ids = {
            part.tool_call_id
            for part in message.parts
            if isinstance(part, ToolCallPart)
        }
        if not call_ids:
            return True
        result_ids = {
            part.tool_call_id
            for following in atom[1:]
            if isinstance(following, ModelRequest)
            for part in following.parts
            if isinstance(part, (ToolReturnPart, RetryPromptPart))
        }
        return call_ids <= result_ids
    if not isinstance(message, ModelRequest):
        return False
    if _is_tool_result_request(message):
        return False
    return any(isinstance(part, UserPromptPart) for part in message.parts)


def slice_turns(
    turns: tuple[TurnMessages, ...],
    cut_index: int,
) -> tuple[tuple[TurnMessages, ...], int]:
    """Slice flattened turns at a safe message offset."""

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
            break
        kept.extend(turns[index:])
        break
    return tuple(turn for turn in kept if turn), summarized_turns


@dataclass(frozen=True, init=False)
class HistoryPolicy:
    """Select a safe recent suffix and summarize context older than it.

    The six dataclass fields preserve the original public reflection contract.
    ``config`` exposes the canonical configuration stored outside that dataclass
    surface, so ``fields()``, ``asdict()``, representation, matching, equality,
    and hashing continue to behave as they did before configuration extraction.
    """

    context_window_tokens: int = DEFAULT_HISTORY_CONFIG.context_window_tokens
    compaction_enabled: bool = DEFAULT_HISTORY_CONFIG.compaction.enabled
    reserve_tokens: int = DEFAULT_HISTORY_CONFIG.compaction.reserve_tokens
    keep_recent_tokens: int = DEFAULT_HISTORY_CONFIG.compaction.keep_recent_tokens
    summary_tokens: int = DEFAULT_HISTORY_CONFIG.compaction.summary_tokens
    token_counter: TokenCounter = field(
        default_factory=TiktokenTokenCounter,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        context_window_tokens: int | None = None,
        compaction_enabled: bool | None = None,
        reserve_tokens: int | None = None,
        keep_recent_tokens: int | None = None,
        summary_tokens: int | None = None,
        token_counter: TokenCounter | None = None,
        *,
        config: HistoryConfig | None = None,
    ) -> None:
        """Create a policy from canonical config or legacy flat values.

        The first six parameters retain their original positional order. Flat
        values supplied with ``config=`` override that base, which also lets
        :func:`dataclasses.replace` continue to replace legacy policy fields.
        """

        flat_values = (
            context_window_tokens,
            compaction_enabled,
            reserve_tokens,
            keep_recent_tokens,
            summary_tokens,
        )
        has_flat_overrides = any(value is not None for value in flat_values)
        base = DEFAULT_HISTORY_CONFIG if config is None else config
        if has_flat_overrides:
            config = HistoryConfig.for_policy(
                context_window_tokens=(
                    base.context_window_tokens
                    if context_window_tokens is None
                    else context_window_tokens
                ),
                compaction=CompactionConfig(
                    enabled=(
                        base.compaction.enabled
                        if compaction_enabled is None
                        else compaction_enabled
                    ),
                    reserve_tokens=(
                        base.compaction.reserve_tokens
                        if reserve_tokens is None
                        else reserve_tokens
                    ),
                    keep_recent_tokens=(
                        base.compaction.keep_recent_tokens
                        if keep_recent_tokens is None
                        else keep_recent_tokens
                    ),
                    summary_tokens=(
                        base.compaction.summary_tokens
                        if summary_tokens is None
                        else summary_tokens
                    ),
                ),
            )
        elif config is None:
            config = DEFAULT_HISTORY_CONFIG
        object.__setattr__(self, "context_window_tokens", config.context_window_tokens)
        object.__setattr__(self, "compaction_enabled", config.compaction.enabled)
        object.__setattr__(self, "reserve_tokens", config.compaction.reserve_tokens)
        object.__setattr__(
            self,
            "keep_recent_tokens",
            config.compaction.keep_recent_tokens,
        )
        object.__setattr__(self, "summary_tokens", config.compaction.summary_tokens)
        object.__setattr__(
            self,
            "token_counter",
            TiktokenTokenCounter() if token_counter is None else token_counter,
        )
        object.__setattr__(self, "_config", config)

    @classmethod
    def from_config(
        cls,
        config: HistoryConfig,
        *,
        token_counter: TokenCounter | None = None,
    ) -> HistoryPolicy:
        return cls(config=config, token_counter=token_counter)

    @property
    def config(self) -> HistoryConfig:
        """Return the canonical config hidden from dataclass reflection."""

        return self.__dict__["_config"]

    @property
    def compaction_trigger_tokens(self) -> int:
        return self.config.compaction_trigger_tokens

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
            index for index in safe_cut_indices(messages) if index >= target_index
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

        kept_turns, summarized_turns = slice_turns(nonempty_turns, cut_index)
        return CompactionPlan(
            messages_to_summarize=messages[:cut_index],
            kept_turns=kept_turns,
            summarized_turns=summarized_turns,
        )


@dataclass(frozen=True, slots=True)
class HistoryState:
    summary: str | None
    summary_message: ModelMessage | None
    turns: tuple[TurnMessages, ...]

    @classmethod
    def initial(cls, initial_history: Sequence[ModelMessage]) -> HistoryState:
        initial_turn = tuple(initial_history)
        return cls(None, None, (initial_turn,) if initial_turn else ())

    @classmethod
    def compacted(
        cls,
        summary: str,
        turns: tuple[TurnMessages, ...],
    ) -> HistoryState:
        return cls(summary, summary_message(summary), turns)

    @property
    def messages(self) -> TurnMessages:
        prefix = (self.summary_message,) if self.summary_message is not None else ()
        return prefix + tuple(message for turn in self.turns for message in turn)

    def append(self, turn: TurnMessages) -> HistoryState:
        return HistoryState(self.summary, self.summary_message, (*self.turns, turn))


def latest_provider_usage(
    messages: Sequence[ModelMessage],
) -> ContextUsage | None:
    """Return the newest usable provider-reported context measurement."""

    for message in reversed(messages):
        if not isinstance(message, ModelResponse):
            continue
        usage = message.usage
        if usage.input_tokens <= 0:
            continue
        return ContextUsage(usage.total_tokens, estimated=False)
    return None
