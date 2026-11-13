import functools

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Env vars only; no env_file: the stack must start without a .env file.
    model_config = SettingsConfigDict(frozen=True)

    # Host-run default; docker-compose.yml sets the container URL.
    # A plain str, not HttpUrl, which would append a trailing "/" to bare hosts.
    ollama_url: str = "http://localhost:11434"
    # Required; the default lives only in docker-compose.yml (single source of truth).
    ollama_model: str = Field(min_length=1)


@functools.cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]  # fields come from env vars
