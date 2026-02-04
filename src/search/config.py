"""Environment configuration using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Configuration
    llm_base_url: str = "http://0.0.0.0:7000/v1"
    llm_api_key: str = "local-key"
    llm_model: str = "lfm2.5"
    llm_temperature: float = 0.1
    llm_top_p: float = 0.1
    llm_max_tokens: int = 2048

    # SearXNG Configuration
    searxng_url: str = "http://searxng.chat.svc.cluster.local:8080"
    searxng_timeout: float = 30.0

    # Research Mode
    research_mode: Literal["speed", "balanced", "quality"] = "balanced"
    max_iterations: int = 5


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
