"""Operational metrics: counted in-process and attached to the current trace.

Everything here works with tracing off, because OpenTelemetry then uses no-op
spans. Never pass user text or portfolio details as attributes.
"""
import threading
from collections import Counter
from contextlib import contextmanager

from opentelemetry import trace

_counts: Counter = Counter()
_lock = threading.Lock()
_langchain_traced = False  # set by setup_tracing; the lookup logs errors otherwise


def use_langchain_spans(enabled: bool = True) -> None:
    global _langchain_traced
    _langchain_traced = enabled


def _current_span():
    """The active span: ours from traced(), else the one LangChain is running.

    LangChain's instrumentor tracks its spans itself and doesn't set OpenTelemetry's
    current span, so without this our work would become separate, unconnected traces.
    """
    span = trace.get_current_span()
    if span.is_recording() or not _langchain_traced:
        return span
    try:
        from openinference.instrumentation.langchain import get_current_span as langchain_span

        return langchain_span() or span
    except ImportError:
        return span


def annotate(**attributes) -> None:
    """Add searchable fields to the current span, as app.<name>."""
    span = _current_span()
    for name, value in attributes.items():
        span.set_attribute(f"app.{name}", value)


def record(event: str, **attributes) -> None:
    """Count an event such as a fallback, and mark it on the current span."""
    with _lock:
        _counts[event] += 1
    annotate(**{event: True}, **attributes)


def snapshot() -> dict[str, int]:
    with _lock:
        return dict(_counts)


def reset() -> None:
    with _lock:
        _counts.clear()


@contextmanager
def traced(name: str, kind: str = "TOOL", **attributes):
    """A span for work LangChain can't see, such as searches and calculations."""
    tracer = trace.get_tracer("ai-finance-assistant")
    parent = trace.set_span_in_context(_current_span())
    with tracer.start_as_current_span(name, context=parent, attributes={"openinference.span.kind": kind}) as span:
        annotate(**attributes)
        yield span
