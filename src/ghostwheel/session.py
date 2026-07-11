from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Protocol, TypeAlias, TypeVar

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    UserPromptPart,
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


@dataclass(frozen=True, slots=True)
class HistoryPolicy:
    """Bound chat history without splitting an agent turn's message sequence.

    ``max_bytes`` is measured from Pydantic AI's serialized message form. It is a
    stable context-size guard, not an exact provider-specific token count.
    """

    max_turns: int = 20
    max_messages: int = 200
    max_bytes: int = 400_000
    response_reserve_bytes: int = 50_000

    def __post_init__(self) -> None:
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if self.max_messages <= 0:
            raise ValueError("max_messages must be positive")
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if self.response_reserve_bytes < 0:
            raise ValueError("response_reserve_bytes must be non-negative")
        if self.response_reserve_bytes >= self.max_bytes:
            raise ValueError("response_reserve_bytes must be smaller than max_bytes")

    def retain(
        self,
        turns: Sequence[TurnMessages],
        *,
        max_bytes: int | None = None,
    ) -> tuple[TurnMessages, ...]:
        retained: list[TurnMessages] = []
        message_count = 0
        byte_count = 0
        byte_limit = (
            self.max_bytes - self.response_reserve_bytes
            if max_bytes is None
            else max_bytes
        )

        for turn in reversed(turns):
            if not turn:
                continue
            turn_bytes = len(ModelMessagesTypeAdapter.dump_json(list(turn)))
            exceeds_limit = (
                len(retained) >= self.max_turns
                or message_count + len(turn) > self.max_messages
                or byte_count + turn_bytes > byte_limit
            )
            if exceeds_limit:
                # If the newest turn cannot fit, drop it but preserve the newest
                # contiguous suffix of older turns that does fit.
                if not retained:
                    continue
                break
            retained.append(turn)
            message_count += len(turn)
            byte_count += turn_bytes

        retained.reverse()
        return tuple(retained)

    def prepare_for_prompt(
        self,
        turns: Sequence[TurnMessages],
        prompt: str,
    ) -> tuple[TurnMessages, ...]:
        prompt_message = ModelRequest(parts=[UserPromptPart(prompt)])
        prompt_bytes = len(ModelMessagesTypeAdapter.dump_json([prompt_message]))
        available = self.max_bytes - self.response_reserve_bytes - prompt_bytes
        if available < 0:
            raise ValueError(
                "Prompt exceeds the configured context budget after reserving "
                "response capacity"
            )
        return self.retain(turns, max_bytes=available)


class ChatSession:
    """Own the canonical, bounded message history for interactive chat."""

    def __init__(
        self,
        runner: AgentRunner,
        *,
        history_policy: HistoryPolicy | None = None,
        initial_history: Sequence[ModelMessage] = (),
    ) -> None:
        self._runner = runner
        self.history_policy = history_policy or HistoryPolicy()
        initial_turn = tuple(initial_history)
        self._turns = self.history_policy.retain(
            (initial_turn,) if initial_turn else ()
        )
        self.last_compacted_turns = int(bool(initial_turn)) - len(self._turns)

    @property
    def history(self) -> tuple[ModelMessage, ...]:
        return tuple(message for turn in self._turns for message in turn)

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    async def send(self, prompt: str) -> TurnOutcome:
        previous_turn_count = len(self._turns)
        try:
            self._turns = self.history_policy.prepare_for_prompt(self._turns, prompt)
        except ValueError as error:
            self.last_compacted_turns = 0
            return TurnFailed(error)
        compacted_before_run = previous_turn_count - len(self._turns)
        self.last_compacted_turns = compacted_before_run
        outcome = await self._runner.run(prompt, self.history, output_type=str)
        if isinstance(outcome, TurnSucceeded) and outcome.new_messages:
            candidate_turns = (*self._turns, outcome.new_messages)
            self._turns = self.history_policy.retain(candidate_turns)
            self.last_compacted_turns += len(candidate_turns) - len(self._turns)
        return outcome

    def clear(self) -> None:
        self._turns = ()
        self.last_compacted_turns = 0
