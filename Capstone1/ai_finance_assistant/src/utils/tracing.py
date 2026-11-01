"""Optional Arize AX tracing, off unless ARIZE_TRACING=true.

Call setup_tracing() before importing LangChain or LangGraph, because the
instrumentor can only see what is created after it runs. Tracing is a diagnostic
aid, so a failure here logs a warning and the app carries on.
"""
import logging
import os

from src.core.config import get_config
from src.utils.metrics import use_langchain_spans

logger = logging.getLogger(__name__)
_started = False


def setup_tracing() -> bool:
    """Start tracing once if enabled. Returns whether tracing is running."""
    global _started
    if _started:
        return True
    get_config()  # loads .env
    if os.getenv("ARIZE_TRACING", "").strip().lower() != "true":
        return False
    space_id, api_key = os.getenv("ARIZE_SPACE_ID"), os.getenv("ARIZE_API_KEY")
    if not space_id or not api_key:
        logger.warning("ARIZE_TRACING is on, but ARIZE_SPACE_ID or ARIZE_API_KEY is missing")
        return False
    try:
        from arize.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor

        provider = register(
            space_id=space_id,
            api_key=api_key,
            project_name=get_config()["tracing"]["project_name"],
        )
        LangChainInstrumentor().instrument(tracer_provider=provider)
    except Exception:
        logger.exception("Could not start tracing; continuing without it")
        return False
    use_langchain_spans()
    _started = True
    logger.info("Arize tracing started")
    return True
