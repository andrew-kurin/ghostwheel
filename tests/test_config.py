import os

import pytest

from ghostwheel.config import Settings
from ghostwheel.models import ModelSpec


@pytest.fixture(autouse=True)
def clear_ghostwheel_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("GHOSTWHEEL_"):
            monkeypatch.delenv(key, raising=False)


def test_default_config_uses_ollama_and_formatter_inherits_chat_model() -> None:
    config = Settings(_env_file=None).resolve()

    assert config.chat_model == ModelSpec(
        provider="ollama",
        model="gemma4:26b",
        base_url="http://localhost:11434/v1",
    )
    assert config.formatter.model == config.chat_model
    assert config.formatter.retries == 5
    assert config.tools.max_output_bytes == 100_000


def test_llama_cpp_config_normalizes_provider_and_formatter_inherits() -> None:
    config = Settings(
        model_provider="llama_cpp",
        model="ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M",
        model_base_url="http://localhost:8080/v1/",
        _env_file=None,
    ).resolve()

    assert config.chat_model == ModelSpec(
        provider="llama-cpp",
        model="ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M",
        base_url="http://localhost:8080/v1",
    )
    assert config.formatter.model == config.chat_model


def test_formatter_can_use_a_different_provider_default_base_url() -> None:
    config = Settings(
        model_provider="llama-cpp",
        model="local-gemma",
        formatter_provider="ollama",
        formatter_model="gemma4:26b",
        _env_file=None,
    ).resolve()

    assert config.chat_model == ModelSpec(
        provider="llama-cpp",
        model="local-gemma",
        base_url="http://localhost:8080/v1",
    )
    assert config.formatter.model == ModelSpec(
        provider="ollama",
        model="gemma4:26b",
        base_url="http://localhost:11434/v1",
    )


def test_unknown_provider_fails_during_resolution() -> None:
    settings = Settings(model_provider="not-a-provider", _env_file=None)

    with pytest.raises(ValueError, match="Unknown model provider"):
        settings.resolve()
