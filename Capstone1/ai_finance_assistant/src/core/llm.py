"""Single place that creates the LLM, so changing provider touches one file."""
from langchain_openai import ChatOpenAI

from src.core.config import get_config, get_openai_key


def get_llm() -> ChatOpenAI | None:
    """Return the chat model, or None when no API key is configured."""
    key = get_openai_key()
    if not key:
        return None
    cfg = get_config()["llm"]
    return ChatOpenAI(
        model=cfg["model"],
        temperature=cfg["temperature"],
        timeout=cfg["timeout_seconds"],
        api_key=key,
    )
