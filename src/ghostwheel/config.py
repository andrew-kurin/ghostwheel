from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ghostwheel.model_config import (
    ModelProvider,
    ModelSpec,
    default_model_base_url,
    validate_model_provider,
)

LogfireSendTo: TypeAlias = bool | Literal["if-token-present"]
ToolProfileName: TypeAlias = Literal["read-only", "shell-only", "full"]
COMPACTION_REQUEST_OVERHEAD_TOKENS = 512
MIN_COMPACTION_INPUT_TOKENS = 1_024


@dataclass(frozen=True)
class ToolConfig:
    max_output_bytes: int
    max_entries: int
    max_directory_scan_entries: int
    max_matches: int
    bash_timeout_seconds: int
    max_search_file_bytes: int
    max_search_total_bytes: int
    max_search_files: int
    search_timeout_seconds: float
    regex_timeout_seconds: float
    profile: ToolProfileName
    review_profile: ToolProfileName


@dataclass(frozen=True)
class ReviewConfig:
    model: ModelSpec
    retries: int
    raw_fallback: bool = True


# Kept as an import-compatible name for callers of the original configuration API.
FormatterConfig = ReviewConfig


@dataclass(frozen=True)
class CompactionConfig:
    enabled: bool
    reserve_tokens: int
    keep_recent_tokens: int
    summary_tokens: int


@dataclass(frozen=True)
class HistoryConfig:
    context_window_tokens: int
    compaction: CompactionConfig

    def __post_init__(self) -> None:
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
        if self.compaction.keep_recent_tokens >= (
            self.context_window_tokens - self.compaction.reserve_tokens
        ):
            raise ValueError(
                "compaction recent-token target must leave room for a summary"
            )
        if (
            self.compaction.keep_recent_tokens + self.compaction.summary_tokens
            >= self.context_window_tokens - self.compaction.reserve_tokens
        ):
            raise ValueError(
                "compaction recent and summary token budgets must leave working room"
            )
        if (
            self.context_window_tokens
            - self.compaction.summary_tokens
            - COMPACTION_REQUEST_OVERHEAD_TOKENS
            < self.compaction.summary_tokens + MIN_COMPACTION_INPUT_TOKENS
        ):
            raise ValueError(
                "compactor input budget must fit the prior summary and at least "
                f"{MIN_COMPACTION_INPUT_TOKENS} prompt tokens"
            )


@dataclass(frozen=True)
class ObservabilityConfig:
    enabled: bool
    include_content: bool
    send_to_logfire: LogfireSendTo


@dataclass(frozen=True)
class AppConfig:
    chat_model: ModelSpec
    review: ReviewConfig
    tools: ToolConfig
    history: HistoryConfig
    observability: ObservabilityConfig

    @property
    def formatter(self) -> ReviewConfig:
        """Compatibility alias for the former prose formatter configuration."""
        return self.review


class Settings(BaseSettings):
    """Environment-backed raw settings.

    Keep this class close to the GHOSTWHEEL_* environment variable names. Use
    resolve() to turn optional overrides into the concrete configuration used by
    the app.
    """

    model_config = SettingsConfigDict(
        env_prefix="GHOSTWHEEL_",
        env_file=".env",
        extra="ignore",
    )

    model_provider: str = "ollama"
    model: str = "gemma4:26b"
    model_base_url: str | None = None

    formatter_provider: str | None = None
    formatter_model: str | None = None
    formatter_base_url: str | None = None
    formatter_retries: int = Field(default=5, ge=0)

    review_provider: str | None = None
    review_model: str | None = None
    review_base_url: str | None = None
    review_retries: int | None = Field(default=None, ge=0)
    review_raw_fallback: bool = True

    max_output_bytes: int = Field(default=100_000, gt=0)
    max_entries: int = Field(default=200, gt=0)
    max_directory_scan_entries: int = Field(default=10_000, gt=0)
    max_matches: int = Field(default=200, gt=0)
    bash_timeout_seconds: int = Field(default=30, gt=0)
    max_search_file_bytes: int = Field(default=5_000_000, gt=0)
    max_search_total_bytes: int = Field(default=50_000_000, gt=0)
    max_search_files: int = Field(default=10_000, gt=0)
    search_timeout_seconds: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    regex_timeout_seconds: float = Field(default=0.05, gt=0, allow_inf_nan=False)
    tool_profile: ToolProfileName = "full"
    review_tool_profile: ToolProfileName = "full"

    history_context_window_tokens: int = Field(default=16_384, gt=0)
    compaction_enabled: bool = True
    compaction_reserve_tokens: int = Field(default=4_096, ge=0)
    compaction_keep_recent_tokens: int = Field(default=4_096, gt=0)
    compaction_summary_tokens: int = Field(default=2_048, gt=0)

    observability_enabled: bool = False
    observability_include_content: bool = False
    observability_send_to_logfire: LogfireSendTo = "if-token-present"

    def resolve(self) -> AppConfig:
        chat_model = self._chat_model_spec()
        review_model = self._review_model_spec(chat_model)

        return AppConfig(
            chat_model=chat_model,
            review=ReviewConfig(
                model=review_model,
                retries=(
                    self.formatter_retries
                    if self.review_retries is None
                    else self.review_retries
                ),
                raw_fallback=self.review_raw_fallback,
            ),
            tools=ToolConfig(
                max_output_bytes=self.max_output_bytes,
                max_entries=self.max_entries,
                max_directory_scan_entries=self.max_directory_scan_entries,
                max_matches=self.max_matches,
                bash_timeout_seconds=self.bash_timeout_seconds,
                max_search_file_bytes=self.max_search_file_bytes,
                max_search_total_bytes=self.max_search_total_bytes,
                max_search_files=self.max_search_files,
                search_timeout_seconds=self.search_timeout_seconds,
                regex_timeout_seconds=self.regex_timeout_seconds,
                profile=self.tool_profile,
                review_profile=self.review_tool_profile,
            ),
            history=HistoryConfig(
                context_window_tokens=self.history_context_window_tokens,
                compaction=CompactionConfig(
                    enabled=self.compaction_enabled,
                    reserve_tokens=self.compaction_reserve_tokens,
                    keep_recent_tokens=self.compaction_keep_recent_tokens,
                    summary_tokens=self.compaction_summary_tokens,
                ),
            ),
            observability=ObservabilityConfig(
                enabled=self.observability_enabled,
                include_content=self.observability_include_content,
                send_to_logfire=self.observability_send_to_logfire,
            ),
        )

    def _chat_model_spec(self) -> ModelSpec:
        provider = validate_model_provider(self.model_provider)
        return ModelSpec(
            provider=provider,
            model=self.model,
            base_url=self._base_url_for(provider, self.model_base_url),
        )

    def _review_model_spec(self, chat_model: ModelSpec) -> ModelSpec:
        has_review_override = any(
            value is not None
            for value in (
                self.review_provider,
                self.review_model,
                self.review_base_url,
            )
        )
        if has_review_override:
            provider = validate_model_provider(
                self.review_provider or chat_model.provider
            )
            model = self.review_model or chat_model.model
            explicit_base_url = self.review_base_url
        else:
            provider = validate_model_provider(
                self.formatter_provider or chat_model.provider
            )
            model = self.formatter_model or chat_model.model
            explicit_base_url = self.formatter_base_url
        if explicit_base_url:
            base_url = explicit_base_url.rstrip("/")
        elif provider == chat_model.provider:
            base_url = chat_model.base_url
        else:
            base_url = self._base_url_for(provider, None)

        return ModelSpec(provider=provider, model=model, base_url=base_url)

    def _base_url_for(
        self,
        provider: str | ModelProvider,
        explicit_base_url: str | None,
    ) -> str:
        provider = validate_model_provider(provider)
        if explicit_base_url:
            return explicit_base_url.rstrip("/")

        return default_model_base_url(provider)
