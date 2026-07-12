import os

import pytest
from pydantic import ValidationError

from ghostwheel.config import Settings
from ghostwheel.models import ModelProvider, ModelSpec


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
    assert config.tools.max_entries == 200
    assert config.tools.max_directory_scan_entries == 10_000
    assert config.tools.max_matches == 200
    assert config.tools.bash_timeout_seconds == 30
    assert config.tools.max_search_file_bytes == 5_000_000
    assert config.tools.max_search_total_bytes == 50_000_000
    assert config.tools.max_search_files == 10_000
    assert config.tools.search_timeout_seconds == 5.0
    assert config.tools.regex_timeout_seconds == 0.05
    assert config.tools.profile == "full"
    assert config.tools.review_profile == "full"
    assert config.history.context_window_tokens == 16_384
    assert config.history.compaction.enabled is True
    assert config.history.compaction.reserve_tokens == 4_096
    assert config.history.compaction.keep_recent_tokens == 4_096
    assert config.history.compaction.summary_tokens == 2_048
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


@pytest.mark.parametrize("field", ["regex_timeout_seconds", "search_timeout_seconds"])
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_search_timeouts_must_be_finite(field: str, value: float) -> None:
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


def test_observability_is_explicitly_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GHOSTWHEEL_OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("GHOSTWHEEL_OBSERVABILITY_INCLUDE_CONTENT", "true")
    monkeypatch.setenv("GHOSTWHEEL_OBSERVABILITY_SEND_TO_LOGFIRE", "false")

    config = Settings(_env_file=None).resolve()

    assert config.observability.enabled is True
    assert config.observability.include_content is True
    assert config.observability.send_to_logfire is False
