"""Framework-neutral model configuration values and provider descriptors."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class ModelProvider(StrEnum):
    """Canonical identifiers for model providers supported by Ghostwheel."""

    OLLAMA = "ollama"
    LLAMA_CPP = "llama-cpp"


def model_provider_key(provider: str | ModelProvider) -> str:
    return str(provider).strip().lower().replace("_", "-")


@dataclass(frozen=True, slots=True)
class ModelProviderDescriptor:
    """Framework-neutral identity and connection defaults for one provider."""

    provider: ModelProvider
    aliases: frozenset[str]
    default_base_url: str

    @property
    def canonical_id(self) -> str:
        return self.provider.value


MODEL_PROVIDER_DESCRIPTORS: Final[Mapping[ModelProvider, ModelProviderDescriptor]] = (
    MappingProxyType(
        {
            ModelProvider.OLLAMA: ModelProviderDescriptor(
                provider=ModelProvider.OLLAMA,
                aliases=frozenset(),
                default_base_url="http://localhost:11434/v1",
            ),
            ModelProvider.LLAMA_CPP: ModelProviderDescriptor(
                provider=ModelProvider.LLAMA_CPP,
                aliases=frozenset({"llamacpp", "llama.cpp"}),
                default_base_url="http://localhost:8080/v1",
            ),
        }
    )
)


def _validate_descriptor_registry() -> None:
    expected = set(ModelProvider)
    actual = set(MODEL_PROVIDER_DESCRIPTORS)
    if actual != expected:
        missing = sorted(provider.value for provider in expected - actual)
        extra = sorted(str(provider) for provider in actual - expected)
        raise RuntimeError(
            "Provider descriptor registry keys must exactly match ModelProvider; "
            f"missing={missing}, extra={extra}"
        )
    mismatched = [
        provider.value
        for provider, descriptor in MODEL_PROVIDER_DESCRIPTORS.items()
        if descriptor.provider is not provider
    ]
    if mismatched:
        raise RuntimeError(
            f"Provider descriptor registry entries must match their keys: {mismatched}"
        )


def _build_provider_lookup() -> dict[str, ModelProvider]:
    lookup: dict[str, ModelProvider] = {}
    for descriptor in MODEL_PROVIDER_DESCRIPTORS.values():
        for name in (descriptor.canonical_id, *descriptor.aliases):
            key = model_provider_key(name)
            previous = lookup.get(key)
            if previous is not None and previous is not descriptor.provider:
                raise RuntimeError(
                    f"Provider name {name!r} is registered for multiple providers"
                )
            lookup[key] = descriptor.provider
    return lookup


_validate_descriptor_registry()
_MODEL_PROVIDER_LOOKUP: Final[Mapping[str, ModelProvider]] = MappingProxyType(
    _build_provider_lookup()
)

# Compatibility views derived from the descriptor registry. They are immutable
# so metadata still has exactly one source of truth.
MODEL_PROVIDER_ALIASES: Final[Mapping[str, ModelProvider]] = MappingProxyType(
    {
        alias: descriptor.provider
        for descriptor in MODEL_PROVIDER_DESCRIPTORS.values()
        for alias in descriptor.aliases
    }
)
MODEL_PROVIDER_DEFAULT_BASE_URLS: Final[Mapping[ModelProvider, str]] = MappingProxyType(
    {
        provider: descriptor.default_base_url
        for provider, descriptor in MODEL_PROVIDER_DESCRIPTORS.items()
    }
)
SUPPORTED_PROVIDERS: Final[tuple[str, ...]] = tuple(
    descriptor.canonical_id for descriptor in MODEL_PROVIDER_DESCRIPTORS.values()
)


def coerce_model_provider(provider: str | ModelProvider) -> ModelProvider:
    key = model_provider_key(provider)
    try:
        return _MODEL_PROVIDER_LOOKUP[key]
    except KeyError:
        raise ValueError(key) from None


def validate_model_provider(provider: str | ModelProvider) -> ModelProvider:
    try:
        return coerce_model_provider(provider)
    except ValueError:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(
            f"Unknown model provider '{provider}'. Supported providers: {supported}"
        ) from None


def model_provider_descriptor(
    provider: str | ModelProvider,
) -> ModelProviderDescriptor:
    return MODEL_PROVIDER_DESCRIPTORS[validate_model_provider(provider)]


def default_model_base_url(provider: str | ModelProvider) -> str:
    return model_provider_descriptor(provider).default_base_url


@dataclass(frozen=True)
class ModelSpec:
    """Resolved model configuration independent of any model SDK."""

    provider: ModelProvider
    model: str
    base_url: str

    def __post_init__(self) -> None:
        # Preserve compatibility with callers that pass a canonical string while
        # ensuring resolved configuration always carries the validated enum.
        if not isinstance(self.provider, ModelProvider):
            object.__setattr__(self, "provider", coerce_model_provider(self.provider))
