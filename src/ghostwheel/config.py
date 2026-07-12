from dataclasses import dataclass, fields, replace
from typing import Literal, TypeAlias

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ghostwheel.history_config import (
    COMPACTION_REQUEST_OVERHEAD_TOKENS as COMPACTION_REQUEST_OVERHEAD_TOKENS,
    DEFAULT_HISTORY_CONFIG,
    MIN_COMPACTION_INPUT_TOKENS as MIN_COMPACTION_INPUT_TOKENS,
    CompactionConfig,
    HistoryConfig,
    validate_compactor_capacity,
)
from ghostwheel.model_config import (
    ModelProvider,
    ModelSpec,
    default_model_base_url,
    validate_model_provider,
)
from ghostwheel.tool_config import DEFAULT_TOOL_LIMITS, ToolLimits, ToolProfile

LogfireSendTo: TypeAlias = bool | Literal["if-token-present"]
# Import-compatible alias retained for callers of the former literal type.
ToolProfileName: TypeAlias = ToolProfile


@dataclass(frozen=True, init=False)
class ToolConfig:
    """Resolved tool configuration with one canonical limits value.

    The scalar dataclass fields retain the original construction, reflection,
    ``asdict()``, and ``replace()`` API. They are immutable mirrors populated
    from the canonical :class:`ToolLimits` object exposed by ``limits``; runtime
    composition passes that object onward unchanged.
    """

    max_output_bytes: int
    max_read_lines: int
    max_read_scan_bytes: int
    max_entries: int
    max_directory_scan_entries: int
    max_matches: int
    bash_timeout_seconds: float
    max_search_file_bytes: int
    max_search_total_bytes: int
    max_search_files: int
    search_timeout_seconds: float
    regex_timeout_seconds: float
    profile: ToolProfile
    review_profile: ToolProfile

    def __init__(
        self,
        max_output_bytes: int | ToolLimits | None = None,
        max_read_lines: int | ToolProfile | str | None = None,
        max_read_scan_bytes: int | ToolProfile | str | None = None,
        max_entries: int | None = None,
        max_directory_scan_entries: int | None = None,
        max_matches: int | None = None,
        bash_timeout_seconds: float | None = None,
        max_search_file_bytes: int | None = None,
        max_search_total_bytes: int | None = None,
        max_search_files: int | None = None,
        search_timeout_seconds: float | None = None,
        regex_timeout_seconds: float | None = None,
        profile: ToolProfile | str | None = None,
        review_profile: ToolProfile | str | None = None,
        *,
        limits: ToolLimits | None = None,
    ) -> None:
        """Accept both the original flat API and canonical ``limits=`` API."""

        # The refactor briefly exposed ``(limits, profile, review_profile)`` as the
        # positional dataclass order. Preserve that form as well as the original
        # fourteen-value scalar order.
        if isinstance(max_output_bytes, ToolLimits):
            if limits is not None:
                raise TypeError("limits was provided more than once")
            if any(
                value is not None
                for value in (
                    max_entries,
                    max_directory_scan_entries,
                    max_matches,
                    bash_timeout_seconds,
                    max_search_file_bytes,
                    max_search_total_bytes,
                    max_search_files,
                    search_timeout_seconds,
                    regex_timeout_seconds,
                )
            ):
                raise TypeError(
                    "The positional limits form accepts only limits and profiles"
                )
            limits = max_output_bytes
            if max_read_lines is not None:
                if profile is not None:
                    raise TypeError("profile was provided more than once")
                profile = max_read_lines
            if max_read_scan_bytes is not None:
                if review_profile is not None:
                    raise TypeError("review_profile was provided more than once")
                review_profile = max_read_scan_bytes
            max_output_bytes = None
            max_read_lines = None
            max_read_scan_bytes = None

        if profile is None:
            raise TypeError("profile is required")
        if review_profile is None:
            raise TypeError("review_profile is required")
        if limits is not None and not isinstance(limits, ToolLimits):
            raise TypeError("limits must be a ToolLimits instance")

        overrides = {
            name: value
            for name, value in {
                "max_output_bytes": max_output_bytes,
                "max_read_lines": max_read_lines,
                "max_read_scan_bytes": max_read_scan_bytes,
                "max_entries": max_entries,
                "max_directory_scan_entries": max_directory_scan_entries,
                "max_matches": max_matches,
                "bash_timeout_seconds": bash_timeout_seconds,
                "max_search_file_bytes": max_search_file_bytes,
                "max_search_total_bytes": max_search_total_bytes,
                "max_search_files": max_search_files,
                "search_timeout_seconds": search_timeout_seconds,
                "regex_timeout_seconds": regex_timeout_seconds,
            }.items()
            if value is not None
        }
        if limits is not None and overrides:
            raise TypeError(
                "limits cannot be combined with scalar limit values; use "
                "ToolConfig.from_limits() or config.with_limits()"
            )
        base_limits = DEFAULT_TOOL_LIMITS if limits is None else limits
        resolved_limits = (
            base_limits if not overrides else replace(base_limits, **overrides)
        )
        for limit_field in fields(ToolLimits):
            object.__setattr__(
                self,
                limit_field.name,
                getattr(resolved_limits, limit_field.name),
            )
        object.__setattr__(self, "profile", ToolProfile(profile))
        object.__setattr__(self, "review_profile", ToolProfile(review_profile))
        object.__setattr__(self, "_limits", resolved_limits)

    @classmethod
    def from_limits(
        cls,
        limits: ToolLimits,
        *,
        profile: ToolProfile | str,
        review_profile: ToolProfile | str,
    ) -> ToolConfig:
        """Construct directly from one canonical limits object."""

        return cls(
            limits=limits,
            profile=profile,
            review_profile=review_profile,
        )

    def with_limits(self, limits: ToolLimits) -> ToolConfig:
        """Return an equivalent profile configuration using ``limits`` by identity."""

        return type(self).from_limits(
            limits,
            profile=self.profile,
            review_profile=self.review_profile,
        )

    @property
    def limits(self) -> ToolLimits:
        """Return the canonical limits object mirrored by the scalar fields."""

        return self.__dict__["_limits"]


@dataclass(frozen=True)
class ReviewConfig:
    model: ModelSpec
    retries: int
    raw_fallback: bool = True


# Kept as an import-compatible name for callers of the original configuration API.
FormatterConfig = ReviewConfig


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

    def __post_init__(self) -> None:
        validate_compactor_capacity(self.history)

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

    max_output_bytes: int = Field(
        default=DEFAULT_TOOL_LIMITS.max_output_bytes,
        gt=0,
    )
    max_read_lines: int = Field(
        default=DEFAULT_TOOL_LIMITS.max_read_lines,
        gt=0,
    )
    max_read_scan_bytes: int = Field(
        default=DEFAULT_TOOL_LIMITS.max_read_scan_bytes,
        gt=0,
    )
    max_entries: int = Field(default=DEFAULT_TOOL_LIMITS.max_entries, gt=0)
    max_directory_scan_entries: int = Field(
        default=DEFAULT_TOOL_LIMITS.max_directory_scan_entries,
        gt=0,
    )
    max_matches: int = Field(default=DEFAULT_TOOL_LIMITS.max_matches, gt=0)
    bash_timeout_seconds: float = Field(
        default=DEFAULT_TOOL_LIMITS.bash_timeout_seconds,
        gt=0,
        allow_inf_nan=False,
    )
    max_search_file_bytes: int = Field(
        default=DEFAULT_TOOL_LIMITS.max_search_file_bytes,
        gt=0,
    )
    max_search_total_bytes: int = Field(
        default=DEFAULT_TOOL_LIMITS.max_search_total_bytes,
        gt=0,
    )
    max_search_files: int = Field(
        default=DEFAULT_TOOL_LIMITS.max_search_files,
        gt=0,
    )
    search_timeout_seconds: float = Field(
        default=DEFAULT_TOOL_LIMITS.search_timeout_seconds,
        gt=0,
        allow_inf_nan=False,
    )
    regex_timeout_seconds: float = Field(
        default=DEFAULT_TOOL_LIMITS.regex_timeout_seconds,
        gt=0,
        allow_inf_nan=False,
    )
    tool_profile: ToolProfile = ToolProfile.FULL
    review_tool_profile: ToolProfile = ToolProfile.FULL

    history_context_window_tokens: int = Field(
        default=DEFAULT_HISTORY_CONFIG.context_window_tokens,
        gt=0,
    )
    compaction_enabled: bool = DEFAULT_HISTORY_CONFIG.compaction.enabled
    compaction_reserve_tokens: int = Field(
        default=DEFAULT_HISTORY_CONFIG.compaction.reserve_tokens,
        ge=0,
    )
    compaction_keep_recent_tokens: int = Field(
        default=DEFAULT_HISTORY_CONFIG.compaction.keep_recent_tokens,
        gt=0,
    )
    compaction_summary_tokens: int = Field(
        default=DEFAULT_HISTORY_CONFIG.compaction.summary_tokens,
        gt=0,
    )

    observability_enabled: bool = False
    observability_include_content: bool = False
    observability_send_to_logfire: LogfireSendTo = "if-token-present"

    @field_validator(
        "max_output_bytes",
        "max_read_lines",
        "max_read_scan_bytes",
        "max_entries",
        "max_directory_scan_entries",
        "max_matches",
        "max_search_file_bytes",
        "max_search_total_bytes",
        "max_search_files",
        mode="before",
    )
    @classmethod
    def _validate_integer_tool_limit_input(cls, value: object) -> object:
        """Reject coercions that would weaken ``ToolLimits`` integer types."""

        if isinstance(value, str):
            candidate = value.strip()
            if candidate[:1] in {"+", "-"}:
                candidate = candidate[1:]
            if candidate.isdecimal():
                return value
        elif isinstance(value, int) and not isinstance(value, bool):
            return value
        raise ValueError("tool integer limits must be integers")

    @field_validator(
        "bash_timeout_seconds",
        "search_timeout_seconds",
        "regex_timeout_seconds",
        mode="before",
    )
    @classmethod
    def _reject_boolean_timeout_input(cls, value: object) -> object:
        """Prevent booleans from being coerced into numeric timeouts."""

        if isinstance(value, bool):
            raise ValueError("tool timeouts must be real numbers, not booleans")
        return value

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
                limits=ToolLimits(
                    max_output_bytes=self.max_output_bytes,
                    max_read_lines=self.max_read_lines,
                    max_read_scan_bytes=self.max_read_scan_bytes,
                    max_entries=self.max_entries,
                    max_directory_scan_entries=self.max_directory_scan_entries,
                    max_matches=self.max_matches,
                    bash_timeout_seconds=self.bash_timeout_seconds,
                    max_search_file_bytes=self.max_search_file_bytes,
                    max_search_total_bytes=self.max_search_total_bytes,
                    max_search_files=self.max_search_files,
                    search_timeout_seconds=self.search_timeout_seconds,
                    regex_timeout_seconds=self.regex_timeout_seconds,
                ),
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
