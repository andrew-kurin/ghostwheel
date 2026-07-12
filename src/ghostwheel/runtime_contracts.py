"""Framework-neutral contracts for running agents and compactors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Protocol, TypeAlias, TypeVar

from pydantic_ai.messages import ModelMessage

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
