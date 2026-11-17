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
    # Seconds; the deadline for the whole agent run (model call + the tool's send),
    # also passed to the Ollama HTTP client.
    ollama_timeout: float = Field(default=60.0, gt=0)
    # False turns off qwen3-style "thinking" (5-10x slower, no routing benefit seen).
    ollama_reasoning: bool = False
    # Cap on generated tokens. A tool call needs ~25; without a cap a rambling model
    # holds Ollama's single slot until the deadline (504) instead of ending early and
    # falling back to other@. Thinking does not fit in 128 tokens: with a hybrid model
    # and OLLAMA_REASONING=true, raise it (e.g. OLLAMA_NUM_PREDICT=1024).
    ollama_num_predict: int = Field(default=128, gt=0)
    # One model call at API startup fills Ollama's prompt cache (system prompt + tool
    # schema), so the first request is as fast as later ones. It blocks startup
    # (healthy = warm), bounded by ollama_timeout; failures are logged, startup goes on.
    ollama_warmup: bool = True

    # Host-run defaults reach the compose MailHog published on 127.0.0.1:1025;
    # docker-compose.yml sets only SMTP_HOST: mailhog.
    smtp_host: str = "localhost"
    smtp_port: int = Field(default=1025, gt=0, lt=65536)
    # Seconds; aiosmtplib applies it to connect and to each SMTP command.
    smtp_timeout: float = Field(default=10.0, gt=0)
    smtp_from: str = "message-router@example.com"


@functools.cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]  # fields come from env vars
