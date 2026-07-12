"""Framework-neutral configuration for conversation history and compaction."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field

COMPACTION_REQUEST_OVERHEAD_TOKENS = 512
MIN_COMPACTION_INPUT_TOKENS = 1_024


@dataclass(frozen=True)
class CompactionConfig:
    enabled: bool = True
    reserve_tokens: int = 4_096
    keep_recent_tokens: int = 4_096
    summary_tokens: int = 2_048


@dataclass(frozen=True)
class HistoryConfig:
    context_window_tokens: int = 16_384
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    _check_compactor_capacity: InitVar[bool] = True

    def __post_init__(self, _check_compactor_capacity: bool) -> None:
        if self.context_window_tokens <= 0:
            raise ValueError("history context window must be positive")
        if self.compaction.reserve_tokens < 0:
            raise ValueError("compaction reserve must be non-negative")
        if self.compaction.keep_recent_tokens <= 0:
            raise ValueError("compaction recent-token target must be positive")
        if self.compaction.summary_tokens <= 0:
            raise ValueError("compaction summary-token budget must be positive")
        if self.compaction.reserve_tokens >= self.context_window_tokens:
            raise ValueError(
                "compaction reserve must be smaller than the context window"
            )
        if self.compaction.enabled and (
            self.compaction.keep_recent_tokens >= self.compaction_trigger_tokens
        ):
            raise ValueError(
                "compaction recent-token target must leave room for a summary"
            )
        if self.compaction.enabled and (
            self.compaction.keep_recent_tokens + self.compaction.summary_tokens
            >= self.compaction_trigger_tokens
        ):
            raise ValueError(
                "compaction recent and summary token budgets must leave working room"
            )
        if self.compaction.enabled and _check_compactor_capacity:
            validate_compactor_capacity(self)

    @classmethod
    def for_policy(
        cls,
        *,
        context_window_tokens: int,
        compaction: CompactionConfig,
    ) -> HistoryConfig:
        """Build abstract policy settings without app-compactor capacity checks.

        ``HistoryPolicy`` historically supports tiny synthetic token domains and
        does not itself require an LLM compactor. Application configuration should
        use the normal constructor, which validates the real compactor budget.
        """

        return cls(
            context_window_tokens=context_window_tokens,
            compaction=compaction,
            _check_compactor_capacity=False,
        )

    @property
    def compaction_trigger_tokens(self) -> int:
        return self.context_window_tokens - self.compaction.reserve_tokens

    @property
    def compactor_input_tokens(self) -> int:
        return (
            self.context_window_tokens
            - self.compaction.summary_tokens
            - COMPACTION_REQUEST_OVERHEAD_TOKENS
        )


def validate_compactor_capacity(config: HistoryConfig) -> None:
    """Validate the application-specific rolling summarizer input budget."""

    if not config.compaction.enabled:
        return
    if (
        config.compactor_input_tokens
        < config.compaction.summary_tokens + MIN_COMPACTION_INPUT_TOKENS
    ):
        raise ValueError(
            "compactor input budget must fit the prior summary and at least "
            f"{MIN_COMPACTION_INPUT_TOKENS} prompt tokens"
        )


DEFAULT_HISTORY_CONFIG = HistoryConfig()
