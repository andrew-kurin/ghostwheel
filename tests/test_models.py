import pytest
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIChatModel

import ghostwheel.providers as provider_adapters
from ghostwheel.model_config import (
    MODEL_PROVIDER_ALIASES,
    MODEL_PROVIDER_DEFAULT_BASE_URLS,
    MODEL_PROVIDER_DESCRIPTORS,
    SUPPORTED_PROVIDERS,
    ModelSpec as ConfigurationModelSpec,
    model_provider_descriptor,
)
from ghostwheel.models import (
    ModelProvider,
    ModelSpec,
    build_model,
    default_base_url,
    formatter_model_settings,
    normalize_provider,
    provider_registration,
    structured_output_model_settings,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ollama", ModelProvider.OLLAMA),
        ("llama-cpp", ModelProvider.LLAMA_CPP),
        ("llama_cpp", ModelProvider.LLAMA_CPP),
        ("llamacpp", ModelProvider.LLAMA_CPP),
        ("llama.cpp", ModelProvider.LLAMA_CPP),
    ],
)
def test_normalize_provider_aliases(raw: str, expected: ModelProvider) -> None:
    assert normalize_provider(raw) is expected


def test_legacy_model_spec_import_is_framework_neutral_value() -> None:
    assert ModelSpec is ConfigurationModelSpec
    assert (
        ModelSpec("ollama", "model", "http://localhost").provider
        is ModelProvider.OLLAMA
    )


@pytest.mark.parametrize("alias", ["llama_cpp", "llamacpp", "llama.cpp"])
def test_model_spec_accepts_legacy_provider_aliases(alias: str) -> None:
    assert (
        ModelSpec(alias, "model", "http://localhost").provider
        is ModelProvider.LLAMA_CPP
    )


def test_default_base_url_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown model provider"):
        default_base_url("missing")


def test_neutral_provider_descriptor_owns_aliases_and_defaults() -> None:
    descriptor = model_provider_descriptor("llama.cpp")
    registration = provider_registration("llama.cpp")

    assert descriptor.provider is ModelProvider.LLAMA_CPP
    assert "llama.cpp" in descriptor.aliases
    assert descriptor.default_base_url == "http://localhost:8080/v1"
    assert registration.provider is descriptor.provider
    assert registration.aliases is descriptor.aliases
    assert registration.default_base_url == descriptor.default_base_url


def test_provider_metadata_views_are_derived_from_the_descriptor_registry() -> None:
    descriptors = tuple(MODEL_PROVIDER_DESCRIPTORS.values())

    assert set(MODEL_PROVIDER_DESCRIPTORS) == set(ModelProvider)
    assert SUPPORTED_PROVIDERS == tuple(
        descriptor.canonical_id for descriptor in descriptors
    )
    assert dict(MODEL_PROVIDER_ALIASES) == {
        alias: descriptor.provider
        for descriptor in descriptors
        for alias in descriptor.aliases
    }
    assert dict(MODEL_PROVIDER_DEFAULT_BASE_URLS) == {
        descriptor.provider: descriptor.default_base_url for descriptor in descriptors
    }


def test_provider_adapter_keys_exactly_match_validated_descriptors() -> None:
    assert set(provider_adapters._PROVIDER_ADAPTERS) == set(MODEL_PROVIDER_DESCRIPTORS)


def test_provider_registry_covers_every_canonical_provider() -> None:
    assert {
        provider_registration(provider).provider for provider in ModelProvider
    } == set(ModelProvider)


def test_build_model_creates_ollama_model() -> None:
    model = build_model(
        ModelSpec(
            provider=ModelProvider.OLLAMA,
            model="gemma4:26b",
            base_url="http://localhost:11434/v1",
        )
    )

    assert isinstance(model, OllamaModel)


def test_build_model_creates_llama_cpp_openai_compatible_model() -> None:
    model = build_model(
        ModelSpec(
            provider=ModelProvider.LLAMA_CPP,
            model="local-gemma",
            base_url="http://localhost:8080/v1",
        )
    )

    assert isinstance(model, OpenAIChatModel)


def test_formatter_settings_disable_reasoning_for_ollama() -> None:
    settings = formatter_model_settings(
        ModelSpec(
            provider=ModelProvider.OLLAMA,
            model="gemma4:26b",
            base_url="http://localhost:11434/v1",
        )
    )

    assert settings is not None
    assert settings["openai_reasoning_effort"] == "none"


def test_structured_output_settings_are_the_canonical_api() -> None:
    spec = ModelSpec(
        provider=ModelProvider.OLLAMA,
        model="gemma4:26b",
        base_url="http://localhost:11434/v1",
    )

    assert structured_output_model_settings(spec) == formatter_model_settings(spec)


def test_formatter_settings_disable_template_thinking_for_llama_cpp() -> None:
    settings = formatter_model_settings(
        ModelSpec(
            provider=ModelProvider.LLAMA_CPP,
            model="local-gemma",
            base_url="http://localhost:8080/v1",
        )
    )

    assert settings is not None
    assert settings["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
