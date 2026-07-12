import os
from dataclasses import asdict, fields, replace

import pytest
from pydantic import ValidationError

from ghostwheel.config import (
    COMPACTION_REQUEST_OVERHEAD_TOKENS,
    MIN_COMPACTION_INPUT_TOKENS,
    CompactionConfig as ConfigCompactionConfig,
    HistoryConfig as ConfigHistoryConfig,
    Settings,
    ToolConfig,
    ToolProfileName,
)
from ghostwheel.history_config import (
    DEFAULT_HISTORY_CONFIG,
    COMPACTION_REQUEST_OVERHEAD_TOKENS as CANONICAL_COMPACTION_OVERHEAD,
    MIN_COMPACTION_INPUT_TOKENS as CANONICAL_MIN_COMPACTION_INPUT,
    CompactionConfig,
    HistoryConfig,
)
from ghostwheel.models import ModelProvider, ModelSpec
from ghostwheel.tool_config import DEFAULT_TOOL_LIMITS, ToolLimits, ToolProfile
from ghostwheel.tools.catalog import ToolProfile as CatalogToolProfile
from ghostwheel.tools.deps import ToolLimits as DepsToolLimits


@pytest.fixture(autouse=True)
def clear_ghostwheel_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("GHOSTWHEEL_"):
            monkeypatch.delenv(key, raising=False)


def test_default_config_uses_ollama_and_formatter_inherits_chat_model() -> None:
    config = Settings(_env_file=None).resolve()

    assert config.chat_model == ModelSpec(
        provider=ModelProvider.OLLAMA,
        model="gemma4:26b",
        base_url="http://localhost:11434/v1",
    )
    assert config.formatter.model == config.chat_model
    assert config.review.model == config.chat_model
    assert config.formatter.retries == 5
    assert config.review.raw_fallback is True
    assert config.tools.max_output_bytes == 100_000
    assert config.tools.max_read_lines == 200
    assert config.tools.max_read_scan_bytes == 5_000_000
    assert config.tools.max_entries == 200
    assert config.tools.max_directory_scan_entries == 10_000
    assert config.tools.max_matches == 200
    assert config.tools.bash_timeout_seconds == 30
    assert config.tools.max_search_file_bytes == 5_000_000
    assert config.tools.max_search_total_bytes == 50_000_000
    assert config.tools.max_search_files == 10_000
    assert config.tools.search_timeout_seconds == 5.0
    assert config.tools.regex_timeout_seconds == 0.05
    assert config.tools.limits == ToolLimits()
    assert config.tools.profile is ToolProfile.FULL
    assert config.tools.review_profile is ToolProfile.FULL
    assert config.history.context_window_tokens == 16_384
    assert config.history.compaction.enabled is True
    assert config.history.compaction.reserve_tokens == 4_096
    assert config.history.compaction.keep_recent_tokens == 4_096
    assert config.history.compaction.summary_tokens == 2_048
    assert config.history == DEFAULT_HISTORY_CONFIG
    assert config.observability.enabled is False
    assert config.observability.include_content is False
    assert config.observability.send_to_logfire == "if-token-present"


def test_llama_cpp_config_normalizes_provider_and_formatter_inherits() -> None:
    config = Settings(
        model_provider="llama_cpp",
        model="ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M",
        model_base_url="http://localhost:8080/v1/",
        _env_file=None,
    ).resolve()

    assert config.chat_model == ModelSpec(
        provider=ModelProvider.LLAMA_CPP,
        model="ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M",
        base_url="http://localhost:8080/v1",
    )
    assert config.formatter.model == config.chat_model


def test_read_limits_load_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GHOSTWHEEL_MAX_READ_LINES", "321")
    monkeypatch.setenv("GHOSTWHEEL_MAX_READ_SCAN_BYTES", "7654321")

    config = Settings(_env_file=None).resolve()

    assert config.tools.max_read_lines == 321
    assert config.tools.max_read_scan_bytes == 7_654_321


def test_fractional_bash_timeout_loads_from_settings() -> None:
    config = Settings(bash_timeout_seconds=0.25, _env_file=None).resolve()

    assert config.tools.bash_timeout_seconds == 0.25
    assert config.tools.limits.bash_timeout_seconds == 0.25


def test_compaction_settings_load_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GHOSTWHEEL_HISTORY_CONTEXT_WINDOW_TOKENS", "32768")
    monkeypatch.setenv("GHOSTWHEEL_COMPACTION_ENABLED", "false")
    monkeypatch.setenv("GHOSTWHEEL_COMPACTION_RESERVE_TOKENS", "8192")
    monkeypatch.setenv("GHOSTWHEEL_COMPACTION_KEEP_RECENT_TOKENS", "6144")
    monkeypatch.setenv("GHOSTWHEEL_COMPACTION_SUMMARY_TOKENS", "3072")

    config = Settings(_env_file=None).resolve()

    assert config.history.context_window_tokens == 32_768
    assert config.history.compaction.enabled is False
    assert config.history.compaction.reserve_tokens == 8_192
    assert config.history.compaction.keep_recent_tokens == 6_144
    assert config.history.compaction.summary_tokens == 3_072


def test_formatter_can_use_a_different_provider_default_base_url() -> None:
    config = Settings(
        model_provider="llama-cpp",
        model="local-gemma",
        formatter_provider="ollama",
        formatter_model="gemma4:26b",
        _env_file=None,
    ).resolve()

    assert config.chat_model == ModelSpec(
        provider=ModelProvider.LLAMA_CPP,
        model="local-gemma",
        base_url="http://localhost:8080/v1",
    )
    assert config.formatter.model == ModelSpec(
        provider=ModelProvider.OLLAMA,
        model="gemma4:26b",
        base_url="http://localhost:11434/v1",
    )


def test_review_settings_take_precedence_over_legacy_formatter_settings() -> None:
    config = Settings(
        formatter_provider="ollama",
        formatter_model="legacy-model",
        formatter_retries=5,
        review_provider="llama-cpp",
        review_model="review-model",
        review_retries=2,
        review_raw_fallback=False,
        _env_file=None,
    ).resolve()

    assert config.review.model == ModelSpec(
        provider=ModelProvider.LLAMA_CPP,
        model="review-model",
        base_url="http://localhost:8080/v1",
    )
    assert config.review.retries == 2
    assert config.review.raw_fallback is False


def test_partial_review_override_does_not_mix_with_legacy_formatter_tier() -> None:
    config = Settings(
        model_provider="ollama",
        model="chat-model",
        formatter_provider="ollama",
        formatter_model="legacy-model",
        formatter_base_url="http://legacy.invalid/v1",
        review_provider="llama-cpp",
        _env_file=None,
    ).resolve()

    assert config.review.model == ModelSpec(
        provider=ModelProvider.LLAMA_CPP,
        model="chat-model",
        base_url="http://localhost:8080/v1",
    )


def test_unknown_provider_fails_during_resolution() -> None:
    settings = Settings(model_provider="not-a-provider", _env_file=None)

    with pytest.raises(ValueError, match="Unknown model provider"):
        settings.resolve()


def test_unknown_provider_fails_even_with_explicit_base_url() -> None:
    settings = Settings(
        model_provider="not-a-provider",
        model_base_url="http://localhost:9999/v1",
        _env_file=None,
    )

    with pytest.raises(ValueError, match="Unknown model provider"):
        settings.resolve()


def test_max_output_bytes_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        Settings(max_output_bytes=0, _env_file=None)


def test_formatter_retries_cannot_be_negative() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        Settings(formatter_retries=-1, _env_file=None)


@pytest.mark.parametrize(
    "field",
    [
        "max_read_lines",
        "max_read_scan_bytes",
        "max_entries",
        "max_directory_scan_entries",
        "max_matches",
        "bash_timeout_seconds",
        "max_search_file_bytes",
        "max_search_total_bytes",
        "max_search_files",
        "search_timeout_seconds",
        "regex_timeout_seconds",
    ],
)
def test_tool_limits_must_be_positive(field: str) -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        Settings(**{field: 0}, _env_file=None)


@pytest.mark.parametrize(
    "field",
    ["bash_timeout_seconds", "regex_timeout_seconds", "search_timeout_seconds"],
)
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_float_timeouts_must_be_finite(field: str, value: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        Settings(**{field: value}, _env_file=None)


def test_history_context_window_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        Settings(history_context_window_tokens=0, _env_file=None)


def test_compaction_reserve_must_fit_context_budget() -> None:
    settings = Settings(
        history_context_window_tokens=100,
        compaction_reserve_tokens=100,
        compaction_keep_recent_tokens=1,
        _env_file=None,
    )

    with pytest.raises(ValueError, match="reserve must be smaller"):
        settings.resolve()


def test_compaction_recent_tokens_must_leave_room_for_summary() -> None:
    settings = Settings(
        history_context_window_tokens=100,
        compaction_reserve_tokens=25,
        compaction_keep_recent_tokens=75,
        compaction_summary_tokens=1,
        _env_file=None,
    )

    with pytest.raises(ValueError, match="must leave room for a summary"):
        settings.resolve()


def test_compaction_recent_and_summary_budgets_leave_working_room() -> None:
    settings = Settings(
        history_context_window_tokens=10_000,
        compaction_reserve_tokens=1_000,
        compaction_keep_recent_tokens=7_000,
        compaction_summary_tokens=2_000,
        _env_file=None,
    )

    with pytest.raises(ValueError, match="must leave working room"):
        settings.resolve()


def test_compaction_context_leaves_a_usable_summarizer_input_budget() -> None:
    settings = Settings(
        history_context_window_tokens=1_500,
        compaction_reserve_tokens=100,
        compaction_keep_recent_tokens=100,
        compaction_summary_tokens=100,
        _env_file=None,
    )

    with pytest.raises(ValueError, match="at least 1024 prompt tokens"):
        settings.resolve()


def test_canonical_history_config_validates_enabled_compactor_capacity() -> None:
    with pytest.raises(ValueError, match="at least 1024 prompt tokens"):
        HistoryConfig(
            context_window_tokens=1_500,
            compaction=CompactionConfig(
                enabled=True,
                reserve_tokens=100,
                keep_recent_tokens=100,
                summary_tokens=100,
            ),
        )


def test_disabled_compaction_does_not_require_compactor_capacity() -> None:
    config = HistoryConfig(
        context_window_tokens=1_500,
        compaction=CompactionConfig(
            enabled=False,
            reserve_tokens=100,
            keep_recent_tokens=100,
            summary_tokens=100,
        ),
    )

    assert config.compactor_input_tokens == 888


def test_disabled_compaction_accepts_default_keep_and_summary_for_4k_context() -> None:
    config = HistoryConfig(
        context_window_tokens=4_096,
        compaction=CompactionConfig(
            enabled=False,
            reserve_tokens=1_024,
            keep_recent_tokens=DEFAULT_HISTORY_CONFIG.compaction.keep_recent_tokens,
            summary_tokens=DEFAULT_HISTORY_CONFIG.compaction.summary_tokens,
        ),
    )

    assert config.compaction_trigger_tokens == 3_072


def test_settings_resolve_disabled_compaction_with_4k_context_defaults() -> None:
    config = Settings(
        history_context_window_tokens=4_096,
        compaction_enabled=False,
        compaction_reserve_tokens=1_024,
        _env_file=None,
    ).resolve()

    assert config.history.compaction.enabled is False
    assert config.history.compaction.keep_recent_tokens == 4_096
    assert config.history.compaction.summary_tokens == 2_048


@pytest.mark.parametrize(
    ("context_window_tokens", "reserve_tokens", "message"),
    [
        (0, 0, "context window must be positive"),
        (4_096, -1, "reserve must be non-negative"),
        (4_096, 4_096, "reserve must be smaller"),
    ],
)
def test_disabled_compaction_still_validates_context_and_reserve(
    context_window_tokens: int,
    reserve_tokens: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        HistoryConfig(
            context_window_tokens=context_window_tokens,
            compaction=CompactionConfig(
                enabled=False,
                reserve_tokens=reserve_tokens,
                keep_recent_tokens=4_096,
                summary_tokens=2_048,
            ),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("compaction_reserve_tokens", -1, "greater than or equal to 0"),
        ("compaction_keep_recent_tokens", 0, "greater than 0"),
        ("compaction_summary_tokens", 0, "greater than 0"),
    ],
)
def test_compaction_token_settings_validate_individual_bounds(
    field: str,
    value: int,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        Settings(**{field: value}, _env_file=None)


def test_tool_profiles_are_validated() -> None:
    with pytest.raises(ValidationError, match="full"):
        Settings(tool_profile="unrestricted", _env_file=None)


def test_tool_profile_and_limits_compatibility_exports_are_canonical() -> None:
    assert ToolProfileName is ToolProfile
    assert CatalogToolProfile is ToolProfile
    assert DepsToolLimits is ToolLimits


def test_settings_tool_defaults_come_from_canonical_limits() -> None:
    for limit_field in fields(ToolLimits):
        assert Settings.model_fields[limit_field.name].default == getattr(
            DEFAULT_TOOL_LIMITS, limit_field.name
        )


@pytest.mark.parametrize(
    "field",
    [
        "max_output_bytes",
        "max_read_lines",
        "max_read_scan_bytes",
        "max_entries",
        "max_directory_scan_entries",
        "max_matches",
        "max_search_file_bytes",
        "max_search_total_bytes",
        "max_search_files",
    ],
)
@pytest.mark.parametrize("value", [True, 1.0, "1", None])
def test_integer_tool_limits_require_non_boolean_ints(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ToolLimits(**{field: value})  # type: ignore[arg-type]


def test_integer_tool_limits_do_not_float_convert_huge_ints() -> None:
    huge_integer = 10**400

    limits = ToolLimits(max_output_bytes=huge_integer)

    assert limits.max_output_bytes == huge_integer


@pytest.mark.parametrize(
    "field",
    ["bash_timeout_seconds", "search_timeout_seconds", "regex_timeout_seconds"],
)
@pytest.mark.parametrize(
    "value",
    [True, "1", object(), 0, -1, float("inf"), float("nan"), 10**400],
)
def test_timeout_tool_limits_require_finite_positive_non_boolean_reals(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="positive and finite real number"):
        ToolLimits(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, 1.0, "1.0"])
def test_settings_rejects_integer_limit_coercion(value: object) -> None:
    with pytest.raises(ValidationError, match="tool integer limits must be integers"):
        Settings(max_entries=value, _env_file=None)  # type: ignore[arg-type]


def test_settings_resolution_applies_canonical_tool_limit_validation() -> None:
    settings = Settings.model_construct(bash_timeout_seconds=10**400)

    with pytest.raises(ValueError, match="positive and finite real number"):
        settings.resolve()


def test_settings_resolution_preserves_huge_integer_limits() -> None:
    huge_integer = 10**400

    config = Settings(max_output_bytes=huge_integer, _env_file=None).resolve()

    assert config.tools.max_output_bytes == huge_integer


def test_history_configuration_compatibility_exports_are_canonical() -> None:
    assert ConfigHistoryConfig is HistoryConfig
    assert ConfigCompactionConfig is CompactionConfig
    assert COMPACTION_REQUEST_OVERHEAD_TOKENS == CANONICAL_COMPACTION_OVERHEAD
    assert MIN_COMPACTION_INPUT_TOKENS == CANONICAL_MIN_COMPACTION_INPUT


def test_tool_profiles_resolve_from_environment_to_the_canonical_enum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GHOSTWHEEL_TOOL_PROFILE", "read-only")
    monkeypatch.setenv("GHOSTWHEEL_REVIEW_TOOL_PROFILE", "shell-only")

    config = Settings(_env_file=None).resolve()

    assert config.tools.profile is ToolProfile.READ_ONLY
    assert config.tools.review_profile is ToolProfile.SHELL_ONLY


def test_resolved_tool_config_coerces_legacy_profile_strings() -> None:
    config = ToolConfig(
        limits=ToolLimits(),
        profile="read-only",  # type: ignore[arg-type]
        review_profile="full",  # type: ignore[arg-type]
    )

    assert config.profile is ToolProfile.READ_ONLY
    assert config.review_profile is ToolProfile.FULL


def test_tool_config_preserves_legacy_positional_dataclass_surface() -> None:
    config = ToolConfig(
        101,
        2,
        303,
        4,
        505,
        6,
        0.75,
        707,
        808,
        9,
        1.25,
        0.15,
        "read-only",
        "shell-only",
    )

    assert config.limits == ToolLimits(
        max_output_bytes=101,
        max_read_lines=2,
        max_read_scan_bytes=303,
        max_entries=4,
        max_directory_scan_entries=505,
        max_matches=6,
        bash_timeout_seconds=0.75,
        max_search_file_bytes=707,
        max_search_total_bytes=808,
        max_search_files=9,
        search_timeout_seconds=1.25,
        regex_timeout_seconds=0.15,
    )
    assert [field.name for field in fields(ToolConfig)] == [
        "max_output_bytes",
        "max_read_lines",
        "max_read_scan_bytes",
        "max_entries",
        "max_directory_scan_entries",
        "max_matches",
        "bash_timeout_seconds",
        "max_search_file_bytes",
        "max_search_total_bytes",
        "max_search_files",
        "search_timeout_seconds",
        "regex_timeout_seconds",
        "profile",
        "review_profile",
    ]
    serialized = asdict(config)
    assert "limits" not in serialized
    assert serialized["max_output_bytes"] == 101
    assert serialized["profile"] is ToolProfile.READ_ONLY


def test_tool_config_replace_updates_canonical_limits() -> None:
    original_limits = ToolLimits(max_output_bytes=101)
    config = ToolConfig(
        limits=original_limits,
        profile="full",
        review_profile="read-only",
    )

    updated = replace(config, max_output_bytes=202, bash_timeout_seconds=0.5)

    assert config.limits is original_limits
    assert updated.max_output_bytes == 202
    assert updated.bash_timeout_seconds == 0.5
    assert updated.limits.max_output_bytes == 202
    assert updated.limits.bash_timeout_seconds == 0.5
    assert updated.profile is ToolProfile.FULL
    assert updated.review_profile is ToolProfile.READ_ONLY


def test_tool_config_replace_never_silently_ignores_limits() -> None:
    config = ToolConfig.from_limits(
        ToolLimits(max_entries=17),
        profile="full",
        review_profile="read-only",
    )
    replacement_limits = ToolLimits(max_entries=23)

    with pytest.raises(TypeError, match="limits cannot be combined with scalar"):
        replace(config, limits=replacement_limits)


def test_tool_config_with_limits_preserves_supplied_identity_and_profiles() -> None:
    config = ToolConfig.from_limits(
        ToolLimits(max_entries=17),
        profile="shell-only",
        review_profile="full",
    )
    replacement_limits = ToolLimits(max_entries=23, bash_timeout_seconds=0.25)

    updated = config.with_limits(replacement_limits)

    assert updated.limits is replacement_limits
    assert updated.max_entries == 23
    assert updated.bash_timeout_seconds == 0.25
    assert updated.profile is ToolProfile.SHELL_ONLY
    assert updated.review_profile is ToolProfile.FULL


def test_tool_config_accepts_brief_canonical_positional_form() -> None:
    limits = ToolLimits(max_entries=17)

    config = ToolConfig(limits, "shell-only", "full")

    assert config.limits is limits
    assert config.max_entries == 17
    assert config.profile is ToolProfile.SHELL_ONLY
    assert config.review_profile is ToolProfile.FULL


def test_observability_is_explicitly_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GHOSTWHEEL_OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("GHOSTWHEEL_OBSERVABILITY_INCLUDE_CONTENT", "true")
    monkeypatch.setenv("GHOSTWHEEL_OBSERVABILITY_SEND_TO_LOGFIRE", "false")

    config = Settings(_env_file=None).resolve()

    assert config.observability.enabled is True
    assert config.observability.include_content is True
    assert config.observability.send_to_logfire is False
