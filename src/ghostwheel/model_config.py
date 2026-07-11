"""Framework-neutral model configuration values."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class ModelProvider(StrEnum):
    """Canonical identifiers for model providers supported by Ghostwheel."""

    OLLAMA = "ollama"
    LLAMA_CPP = "llama-cpp"


MODEL_PROVIDER_ALIASES: Final[dict[str, ModelProvider]] = {
    "llamacpp": ModelProvider.LLAMA_CPP,
    "llama.cpp": ModelProvider.LLAMA_CPP,
}
MODEL_PROVIDER_DEFAULT_BASE_URLS: Final[dict[ModelProvider, str]] = {
    ModelProvider.OLLAMA: "http://localhost:11434/v1",
    ModelProvider.LLAMA_CPP: "http://localhost:8080/v1",
}
SUPPORTED_PROVIDERS: Final[tuple[str, ...]] = tuple(
    provider.value for provider in MODEL_PROVIDER_DEFAULT_BASE_URLS
)


def model_provider_key(provider: str | ModelProvider) -> str:
    return str(provider).strip().lower().replace("_", "-")


def coerce_model_provider(provider: str | ModelProvider) -> ModelProvider:
    key = model_provider_key(provider)
    if key in MODEL_PROVIDER_ALIASES:
        return MODEL_PROVIDER_ALIASES[key]
    return ModelProvider(key)


def validate_model_provider(provider: str | ModelProvider) -> ModelProvider:
    try:
        return coerce_model_provider(provider)
    except ValueError:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(
            f"Unknown model provider '{provider}'. Supported providers: {supported}"
        ) from None


def default_model_base_url(provider: str | ModelProvider) -> str:
    return MODEL_PROVIDER_DEFAULT_BASE_URLS[validate_model_provider(provider)]


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
