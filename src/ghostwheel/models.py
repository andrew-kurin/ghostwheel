from dataclasses import dataclass
from typing import Final

from pydantic_ai.models import Model
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

SUPPORTED_PROVIDERS: Final[tuple[str, ...]] = ("ollama", "llama-cpp")

_DEFAULT_BASE_URLS: Final[dict[str, str]] = {
    "ollama": "http://localhost:11434/v1",
    "llama-cpp": "http://localhost:8080/v1",
}

_PROVIDER_ALIASES: Final[dict[str, str]] = {
    "llamacpp": "llama-cpp",
    "llama.cpp": "llama-cpp",
    "llama_cpp": "llama-cpp",
}

# llama.cpp exposes an OpenAI-compatible Chat Completions API, but it is not
# OpenAI. Keep the OpenAI wire format while using a local-friendly profile.
LLAMA_CPP_PROFILE: Final = OpenAIModelProfile(
    json_schema_transformer=OpenAIJsonSchemaTransformer,
    openai_chat_thinking_field="reasoning",
    openai_supports_strict_tool_definition=False,
    supports_json_schema_output=True,
    supports_json_object_output=True,
)


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    base_url: str


def normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower().replace("_", "-")
    return _PROVIDER_ALIASES.get(normalized, normalized)


def default_base_url(provider: str) -> str:
    normalized = normalize_provider(provider)
    try:
        return _DEFAULT_BASE_URLS[normalized]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(
            f"Unknown model provider '{provider}'. Supported providers: {supported}"
        ) from exc


def build_model(spec: ModelSpec) -> Model:
    provider = normalize_provider(spec.provider)

    if provider == "ollama":
        return OllamaModel(
            spec.model,
            provider=OllamaProvider(base_url=spec.base_url),
        )

    if provider == "llama-cpp":
        return OpenAIChatModel(
            spec.model,
            provider=OpenAIProvider(base_url=spec.base_url, api_key="no-key"),
            profile=LLAMA_CPP_PROFILE,
        )

    supported = ", ".join(SUPPORTED_PROVIDERS)
    raise ValueError(
        f"Unknown model provider '{spec.provider}'. Supported providers: {supported}"
    )


def formatter_model_settings(spec: ModelSpec) -> OpenAIChatModelSettings | None:
    provider = normalize_provider(spec.provider)

    if provider == "ollama":
        return OpenAIChatModelSettings(openai_reasoning_effort="none")

    if provider == "llama-cpp":
        return OpenAIChatModelSettings(
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )

    return None
