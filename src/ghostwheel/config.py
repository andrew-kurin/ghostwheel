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


@dataclass(frozen=True)
class ToolConfig:
    max_output_bytes: int
    max_entries: int
    max_directory_scan_entries: int
    max_matches: int
    bash_timeout_seconds: int
    max_search_file_bytes: int
    max_search_files: int
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
class HistoryConfig:
    max_turns: int
    max_messages: int
    max_bytes: int
    response_reserve_bytes: int

    def __post_init__(self) -> None:
        if self.response_reserve_bytes >= self.max_bytes:
            raise ValueError("history response reserve must be smaller than max bytes")


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
    max_search_files: int = Field(default=10_000, gt=0)
    regex_timeout_seconds: float = Field(default=0.05, gt=0, allow_inf_nan=False)
    tool_profile: ToolProfileName = "full"
    review_tool_profile: ToolProfileName = "full"

    history_max_turns: int = Field(default=20, gt=0)
    history_max_messages: int = Field(default=200, gt=0)
    history_max_bytes: int = Field(default=400_000, gt=0)
    history_response_reserve_bytes: int = Field(default=50_000, ge=0)

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
                max_search_files=self.max_search_files,
                regex_timeout_seconds=self.regex_timeout_seconds,
                profile=self.tool_profile,
                review_profile=self.review_tool_profile,
            ),
            history=HistoryConfig(
                max_turns=self.history_max_turns,
                max_messages=self.history_max_messages,
                max_bytes=self.history_max_bytes,
                response_reserve_bytes=self.history_response_reserve_bytes,
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
