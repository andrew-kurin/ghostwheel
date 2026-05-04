from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GHOSTWHEEL_",
        env_file=".env",
        extra="ignore",
    )
    model: str = "gemma4:26b"
    formatter_model: str = "gemma4:26b"
    ollama_url: str = "http://localhost:11434/v1"
    max_output_bytes: int = 100_000
    formatter_retries: int = 5
