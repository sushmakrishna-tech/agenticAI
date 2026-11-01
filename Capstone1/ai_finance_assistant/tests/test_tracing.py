import sys
import types

import pytest

from src.utils import metrics, tracing


@pytest.fixture
def fake_arize(monkeypatch):
    """Fake Arize packages, so tests never contact the service."""
    calls = {}

    def register(**kwargs):
        calls["register"] = kwargs
        if calls.get("fail"):
            raise RuntimeError("no network")
        return "provider"

    class LangChainInstrumentor:
        def instrument(self, tracer_provider):
            calls["provider"] = tracer_provider

    otel = types.ModuleType("arize.otel")
    otel.register = register
    instrumentation = types.ModuleType("openinference.instrumentation.langchain")
    instrumentation.LangChainInstrumentor = LangChainInstrumentor
    monkeypatch.setitem(sys.modules, "arize.otel", otel)
    monkeypatch.setitem(sys.modules, "openinference.instrumentation.langchain", instrumentation)
    monkeypatch.setattr(tracing, "_started", False)
    monkeypatch.setenv("ARIZE_SPACE_ID", "space")
    monkeypatch.setenv("ARIZE_API_KEY", "key")
    return calls


def test_off_by_default_even_with_keys(fake_arize, monkeypatch):
    monkeypatch.delenv("ARIZE_TRACING", raising=False)
    assert tracing.setup_tracing() is False
    assert "register" not in fake_arize


def test_enabled_without_keys_does_not_start(fake_arize, monkeypatch):
    monkeypatch.setenv("ARIZE_TRACING", "true")
    monkeypatch.setenv("ARIZE_API_KEY", "")
    assert tracing.setup_tracing() is False


def test_enabled_registers_and_instruments_once(fake_arize, monkeypatch):
    monkeypatch.setenv("ARIZE_TRACING", "TRUE")
    assert tracing.setup_tracing() is True
    assert fake_arize["provider"] == "provider"
    assert metrics._langchain_traced is True
    metrics.use_langchain_spans(False)
    assert fake_arize["register"]["project_name"] == "ai-finance-assistant"
    fake_arize.pop("register")
    assert tracing.setup_tracing() is True  # Streamlit reruns must not instrument again
    assert "register" not in fake_arize


def test_failure_does_not_stop_the_app(fake_arize, monkeypatch):
    monkeypatch.setenv("ARIZE_TRACING", "true")
    fake_arize["fail"] = True
    assert tracing.setup_tracing() is False
