"""ChatOpenAI wrapper for local llama-server."""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from ..config import get_settings


@lru_cache
def get_llm_client() -> ChatOpenAI:
    """Get configured ChatOpenAI client for local llama-server."""
    settings = get_settings()
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        top_p=settings.llm_top_p,
    )
