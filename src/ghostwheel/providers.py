"""Pydantic-AI adapter factories for framework-neutral model providers."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from pydantic_ai.models import Model
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from ghostwheel.model_config import (
    MODEL_PROVIDER_DESCRIPTORS,
    SUPPORTED_PROVIDERS as SUPPORTED_PROVIDERS,
    ModelProvider,
    ModelSpec,
    default_model_base_url,
    model_provider_descriptor,
    validate_model_provider,
)

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


@dataclass(frozen=True, slots=True)
class _ProviderAdapter:
    model_factory: ModelFactory
    structured_output_settings_factory: StructuredOutputSettingsFactory


@dataclass(frozen=True)
class ProviderRegistration:
    """Compatibility aggregate composed from a descriptor and SDK adapter."""

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


_PROVIDER_ADAPTERS: Final[Mapping[ModelProvider, _ProviderAdapter]] = MappingProxyType(
    {
        ModelProvider.OLLAMA: _ProviderAdapter(
            model_factory=_build_ollama,
            structured_output_settings_factory=_ollama_structured_output_settings,
        ),
        ModelProvider.LLAMA_CPP: _ProviderAdapter(
            model_factory=_build_llama_cpp,
            structured_output_settings_factory=_llama_cpp_structured_output_settings,
        ),
    }
)


def _validate_adapter_registry() -> None:
    descriptor_keys = set(MODEL_PROVIDER_DESCRIPTORS)
    adapter_keys = set(_PROVIDER_ADAPTERS)
    if adapter_keys != descriptor_keys:
        missing = sorted(provider.value for provider in descriptor_keys - adapter_keys)
        extra = sorted(provider.value for provider in adapter_keys - descriptor_keys)
        raise RuntimeError(
            "Provider adapter registry keys must exactly match provider descriptors; "
            f"missing={missing}, extra={extra}"
        )


_validate_adapter_registry()


def normalize_provider(provider: str | ModelProvider) -> ModelProvider:
    """Resolve a provider name or alias to its validated canonical value."""

    return validate_model_provider(provider)


def validate_provider(provider: str | ModelProvider) -> ModelProvider:
    """Compatibility alias for provider normalization and validation."""

    return normalize_provider(provider)


def provider_registration(
    provider: str | ModelProvider,
) -> ProviderRegistration:
    normalized = normalize_provider(provider)
    descriptor = model_provider_descriptor(normalized)
    adapter = _PROVIDER_ADAPTERS[normalized]
    return ProviderRegistration(
        provider=descriptor.provider,
        aliases=descriptor.aliases,
        default_base_url=descriptor.default_base_url,
        model_factory=adapter.model_factory,
        structured_output_settings_factory=adapter.structured_output_settings_factory,
    )


def default_base_url(provider: str | ModelProvider) -> str:
    return default_model_base_url(provider)


def build_model(spec: ModelSpec) -> Model:
    adapter = _PROVIDER_ADAPTERS[normalize_provider(spec.provider)]
    return adapter.model_factory(spec)


def structured_output_model_settings(
    spec: ModelSpec,
) -> OpenAIChatModelSettings | None:
    adapter = _PROVIDER_ADAPTERS[normalize_provider(spec.provider)]
    return adapter.structured_output_settings_factory(spec)


def formatter_model_settings(spec: ModelSpec) -> OpenAIChatModelSettings | None:
    """Compatibility alias for structured-output model settings."""

    return structured_output_model_settings(spec)
