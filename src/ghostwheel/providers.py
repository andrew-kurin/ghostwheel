"""Provider registry and Pydantic AI model adapters."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from pydantic_ai.models import Model
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from ghostwheel.model_config import (
    MODEL_PROVIDER_ALIASES,
    SUPPORTED_PROVIDERS as MODEL_SUPPORTED_PROVIDERS,
    ModelProvider,
    ModelSpec,
    default_model_base_url,
    model_provider_key,
    validate_model_provider,
)

SUPPORTED_PROVIDERS = MODEL_SUPPORTED_PROVIDERS


# llama.cpp exposes an OpenAI-compatible Chat Completions API, but it is not
# OpenAI. Keep the OpenAI wire format while using a local-friendly profile.
LLAMA_CPP_PROFILE: Final = OpenAIModelProfile(
    json_schema_transformer=OpenAIJsonSchemaTransformer,
    openai_chat_thinking_field="reasoning",
    openai_supports_strict_tool_definition=False,
    supports_json_schema_output=True,
    supports_json_object_output=True,
)

StructuredOutputSettingsFactory = Callable[[ModelSpec], OpenAIChatModelSettings | None]
ModelFactory = Callable[[ModelSpec], Model]


@dataclass(frozen=True)
class ProviderRegistration:
    """All provider-specific behavior and metadata in one registry entry."""

    provider: ModelProvider
    aliases: frozenset[str]
    default_base_url: str
    model_factory: ModelFactory
    structured_output_settings_factory: StructuredOutputSettingsFactory


def _build_ollama(spec: ModelSpec) -> Model:
    return OllamaModel(
        spec.model,
        provider=OllamaProvider(base_url=spec.base_url),
    )


def _ollama_structured_output_settings(_spec: ModelSpec) -> OpenAIChatModelSettings:
    return OpenAIChatModelSettings(openai_reasoning_effort="none")


def _build_llama_cpp(spec: ModelSpec) -> Model:
    return OpenAIChatModel(
        spec.model,
        provider=OpenAIProvider(base_url=spec.base_url, api_key="no-key"),
        profile=LLAMA_CPP_PROFILE,
    )


def _llama_cpp_structured_output_settings(
    _spec: ModelSpec,
) -> OpenAIChatModelSettings:
    return OpenAIChatModelSettings(
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )


_PROVIDER_REGISTRY: Final[dict[ModelProvider, ProviderRegistration]] = {
    ModelProvider.OLLAMA: ProviderRegistration(
        provider=ModelProvider.OLLAMA,
        aliases=frozenset(),
        default_base_url=default_model_base_url(ModelProvider.OLLAMA),
        model_factory=_build_ollama,
        structured_output_settings_factory=_ollama_structured_output_settings,
    ),
    ModelProvider.LLAMA_CPP: ProviderRegistration(
        provider=ModelProvider.LLAMA_CPP,
        aliases=frozenset(
            alias
            for alias, provider in MODEL_PROVIDER_ALIASES.items()
            if provider is ModelProvider.LLAMA_CPP
        ),
        default_base_url=default_model_base_url(ModelProvider.LLAMA_CPP),
        model_factory=_build_llama_cpp,
        structured_output_settings_factory=_llama_cpp_structured_output_settings,
    ),
}


def _provider_key(provider: str | ModelProvider) -> str:
    return model_provider_key(provider)


def _provider_lookup() -> dict[str, ModelProvider]:
    lookup: dict[str, ModelProvider] = {}
    for registration in _PROVIDER_REGISTRY.values():
        lookup[_provider_key(registration.provider)] = registration.provider
        lookup.update(
            {
                _provider_key(alias): registration.provider
                for alias in registration.aliases
            }
        )
    return lookup


_PROVIDER_LOOKUP: Final = _provider_lookup()


def normalize_provider(provider: str | ModelProvider) -> ModelProvider:
    """Resolve a provider name or alias to its validated canonical value."""
    normalized = validate_model_provider(provider)
    # Assert registry completeness at the adapter boundary.
    return _PROVIDER_LOOKUP[_provider_key(normalized)]


def validate_provider(provider: str | ModelProvider) -> ModelProvider:
    """Compatibility alias for provider normalization and validation."""

    return normalize_provider(provider)


def provider_registration(
    provider: str | ModelProvider,
) -> ProviderRegistration:
    return _PROVIDER_REGISTRY[normalize_provider(provider)]


def default_base_url(provider: str | ModelProvider) -> str:
    return provider_registration(provider).default_base_url


def build_model(spec: ModelSpec) -> Model:
    return provider_registration(spec.provider).model_factory(spec)


def structured_output_model_settings(
    spec: ModelSpec,
) -> OpenAIChatModelSettings | None:
    registration = provider_registration(spec.provider)
    return registration.structured_output_settings_factory(spec)


def formatter_model_settings(spec: ModelSpec) -> OpenAIChatModelSettings | None:
    """Compatibility alias for structured-output model settings."""
    return structured_output_model_settings(spec)
