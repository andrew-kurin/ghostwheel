from dataclasses import dataclass

from pydantic_settings import BaseSettings, SettingsConfigDict

from ghostwheel.models import ModelSpec, default_base_url, validate_provider


@dataclass(frozen=True)
class ToolConfig:
    max_output_bytes: int


@dataclass(frozen=True)
class FormatterConfig:
    model: ModelSpec
    retries: int


@dataclass(frozen=True)
class AppConfig:
    chat_model: ModelSpec
    formatter: FormatterConfig
    tools: ToolConfig


class Settings(BaseSettings):
    """Environment-backed raw settings.

    Keep this class close to the GHOSTWHEEL_* environment variable names. Use
    resolve() to turn optional overrides into the concrete configuration used by
    the app.
    """

    model_config = SettingsConfigDict(
        env_prefix="GHOSTWHEEL_",
        env_file=".env",
        extra="ignore",
    )

    model_provider: str = "ollama"
    model: str = "gemma4:26b"
    model_base_url: str | None = None

    formatter_provider: str | None = None
    formatter_model: str | None = None
    formatter_base_url: str | None = None
    formatter_retries: int = 5

    max_output_bytes: int = 100_000

    def resolve(self) -> AppConfig:
        chat_model = self._chat_model_spec()
        formatter_model = self._formatter_model_spec(chat_model)

        return AppConfig(
            chat_model=chat_model,
            formatter=FormatterConfig(
                model=formatter_model,
                retries=self.formatter_retries,
            ),
            tools=ToolConfig(max_output_bytes=self.max_output_bytes),
        )

    def _chat_model_spec(self) -> ModelSpec:
        provider = validate_provider(self.model_provider)
        return ModelSpec(
            provider=provider,
            model=self.model,
            base_url=self._base_url_for(provider, self.model_base_url),
        )

    def _formatter_model_spec(self, chat_model: ModelSpec) -> ModelSpec:
        provider = validate_provider(self.formatter_provider or chat_model.provider)
        model = self.formatter_model or chat_model.model

        if self.formatter_base_url:
            base_url = self.formatter_base_url.rstrip("/")
        elif provider == chat_model.provider:
            base_url = chat_model.base_url
        else:
            base_url = self._base_url_for(provider, None)

        return ModelSpec(provider=provider, model=model, base_url=base_url)

    def _base_url_for(self, provider: str, explicit_base_url: str | None) -> str:
        provider = validate_provider(provider)
        if explicit_base_url:
            return explicit_base_url.rstrip("/")

        return default_base_url(provider)
