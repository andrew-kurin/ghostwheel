import pytest
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIChatModel

from ghostwheel.models import (
    ModelSpec,
    build_model,
    default_base_url,
    formatter_model_settings,
    normalize_provider,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ollama", "ollama"),
        ("llama-cpp", "llama-cpp"),
        ("llama_cpp", "llama-cpp"),
        ("llamacpp", "llama-cpp"),
        ("llama.cpp", "llama-cpp"),
    ],
)
def test_normalize_provider_aliases(raw: str, expected: str) -> None:
    assert normalize_provider(raw) == expected


def test_default_base_url_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown model provider"):
        default_base_url("missing")


def test_build_model_creates_ollama_model() -> None:
    model = build_model(
        ModelSpec(
            provider="ollama",
            model="gemma4:26b",
            base_url="http://localhost:11434/v1",
        )
    )

    assert isinstance(model, OllamaModel)


def test_build_model_creates_llama_cpp_openai_compatible_model() -> None:
    model = build_model(
        ModelSpec(
            provider="llama-cpp",
            model="local-gemma",
            base_url="http://localhost:8080/v1",
        )
    )

    assert isinstance(model, OpenAIChatModel)


def test_formatter_settings_disable_reasoning_for_ollama() -> None:
    settings = formatter_model_settings(
        ModelSpec(
            provider="ollama",
            model="gemma4:26b",
            base_url="http://localhost:11434/v1",
        )
    )

    assert settings is not None
    assert settings["openai_reasoning_effort"] == "none"


def test_formatter_settings_disable_template_thinking_for_llama_cpp() -> None:
    settings = formatter_model_settings(
        ModelSpec(
            provider="llama-cpp",
            model="local-gemma",
            base_url="http://localhost:8080/v1",
        )
    )

    assert settings is not None
    assert settings["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
