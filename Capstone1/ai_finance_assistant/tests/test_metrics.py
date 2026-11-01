"""Metrics, spans, feedback, usage, and eval scoring, with no API calls."""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.agents.common import explain
from src.agents.registry import AGENTS
from src.utils import metrics
from src.utils.feedback import save_feedback, summarize
from src.utils.usage import UsageTracker
from src.workflow.evaluate import percentile, route_scores, to_markdown
from src.workflow.graph import build_graph
from src.workflow.router import make_router


@pytest.fixture(autouse=True)
def clean_counts():
    metrics.reset()
    yield
    metrics.reset()


@pytest.fixture
def spans(monkeypatch):
    """Capture spans in memory for the code under test."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(metrics.trace, "get_tracer", lambda name: provider.get_tracer(name))
    return exporter


def test_record_counts_events():
    metrics.record("routing_fallback")
    metrics.record("routing_fallback")
    assert metrics.snapshot() == {"routing_fallback": 2}


def test_traced_span_has_kind_and_attributes(spans):
    with metrics.traced("knowledge_search", "RETRIEVER", agent="tax"):
        metrics.annotate(result_count=3)
    span = spans.get_finished_spans()[0]
    assert span.name == "knowledge_search"
    assert span.attributes["openinference.span.kind"] == "RETRIEVER"
    assert span.attributes["app.agent"] == "tax"
    assert span.attributes["app.result_count"] == 3


def test_nested_traced_spans_stay_in_one_trace(spans):
    with metrics.traced("outer"):
        with metrics.traced("inner"):
            pass
    inner, outer = spans.get_finished_spans()
    assert inner.parent.span_id == outer.context.span_id
    assert inner.context.trace_id == outer.context.trace_id


def test_market_span_keeps_its_tool_kind(spans):
    from datetime import datetime, timezone

    from src.data.market_data import MarketDataService, Quote

    class Provider:
        name = "p"

        def get_quote(self, symbol):
            return Quote(symbol, 1.0, None, None, datetime.now(timezone.utc), "p")

    MarketDataService([Provider()]).get_quote("AAPL")
    attributes = spans.get_finished_spans()[0].attributes
    assert attributes["openinference.span.kind"] == "TOOL"
    assert attributes["app.request"] == "quote" and attributes["app.provider"] == "p"


def test_everything_works_with_tracing_off():
    with metrics.traced("x"):
        metrics.annotate(a=1)
        metrics.record("event")
    assert metrics.snapshot() == {"event": 1}


def test_rejected_answer_is_counted():
    class LLM:
        def invoke(self, messages):
            return AIMessage("You will have $40,000")

    explain(LLM(), "t", "Target is $60,000", {"messages": [HumanMessage("q")]})
    assert metrics.snapshot()["answer_rejected"] == 1


def test_routing_fallback_is_counted():
    class Broken:
        def with_structured_output(self, schema):
            raise RuntimeError("down")

    make_router(Broken())({"messages": [HumanMessage("What is an IRA?")]})
    assert metrics.snapshot()["routing_fallback"] == 1


def test_no_fallback_event_without_a_key():
    make_router(None)({"messages": [HumanMessage("What is an IRA?")]})
    assert "routing_fallback" not in metrics.snapshot()


def test_failed_agent_and_compliance_flag_are_counted():
    def boom(state):
        raise RuntimeError("down")

    def advises(state):
        return {"agent": "finance_qa", "content": "You should buy TSLA", "sources": [], "error": False}

    functions = {name: advises for name in AGENTS} | {"tax": boom}
    graph = build_graph(llm=None, agent_functions=functions)
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"messages": [HumanMessage("What is an ETF?")]}, config)
    assert metrics.snapshot()["compliance_flagged"] == 1
    graph.invoke({"messages": [HumanMessage("What is a Roth IRA?")]}, config)
    assert metrics.snapshot()["agent_exception"] == 1


def test_market_fallback_and_stale_are_counted():
    from src.data.market_data import MarketDataService, ProviderError, Quote
    from datetime import datetime, timezone

    class Provider:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        def get_quote(self, symbol):
            if self.fail:
                raise ProviderError("down")
            return Quote(symbol, 1.0, None, None, datetime.now(timezone.utc), self.name)

    good = Provider("b")
    now = [0.0]
    service = MarketDataService([Provider("a", True), good], clock=lambda: now[0])
    service.get_quote("AAPL")
    assert metrics.snapshot()["market_provider_fallback"] == 1
    good.fail, now[0] = True, 1000.0
    assert service.get_quote("AAPL").stale is True
    assert metrics.snapshot()["market_stale_served"] == 1


def test_feedback_is_saved_without_conversation_text(tmp_path):
    path = tmp_path / "f.jsonl"
    save_feedback(True, "thread", ["tax"], path)
    save_feedback(False, "thread", ["goals"], path)
    entry = json.loads(path.read_text().splitlines()[0])
    assert set(entry) == {"time", "positive", "thread_id", "agents"}
    assert summarize(path) == {"total": 2, "positive_pct": 50}
    assert metrics.snapshot() == {"feedback_positive": 1, "feedback_negative": 1}


def test_feedback_summary_with_no_file(tmp_path):
    assert summarize(tmp_path / "none.jsonl")["total"] == 0


def generation(input_tokens=100, output_tokens=50, finish="stop"):
    message = AIMessage("x", usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens,
                                             "total_tokens": input_tokens + output_tokens})
    return LLMResult(generations=[[ChatGeneration(message=message, generation_info={"finish_reason": finish})]])


def test_usage_tracker_tokens_cost_and_finish_reasons():
    tracker = UsageTracker()
    tracker.on_llm_end(generation())
    tracker.on_llm_end(generation(1_000_000, 1_000_000, "length"))
    tracker.on_llm_error(RuntimeError("x"))
    assert (tracker.calls, tracker.errors) == (2, 1)
    assert tracker.finish_reasons == {"stop": 1, "length": 1}
    assert tracker.input_tokens == 1_000_100
    # 1,000,100 x $0.15 / 1M + 1,000,050 x $0.60 / 1M
    assert tracker.cost() == pytest.approx(0.150015 + 0.60003)


def test_route_scores():
    cases = [(["tax"], ["tax"]), (["tax"], ["tax", "goals"]), (["tax"], ["news"])]
    scores = route_scores(cases)
    assert scores["exact_pct"] == pytest.approx(100 / 3)
    assert scores["covers_pct"] == pytest.approx(200 / 3)


def test_percentile():
    assert percentile([3.0, 1.0, 2.0], 50) == 2.0
    assert percentile(list(map(float, range(1, 101))), 95) == 95.0


def test_results_markdown_lists_failures():
    report = to_markdown({
        "model": "m", "turns": 1, "routing": {"exact_pct": 100, "covers_pct": 100},
        "faithfulness_pct": 50, "faithfulness_checked": 2, "relevance_pct": 100, "advice_safety_pct": 100,
        "out_of_scope_declined_pct": 100, "latency_p50_s": 1, "latency_p95_s": 2, "cost_per_turn_usd": 0.001,
        "llm_calls_per_turn": 3, "llm_errors": 0, "cut_off_by_token_limit": 0, "agent_errors": 0,
        "events": {"answer_rejected": 1},
        "failures": [{"check": "routing", "question": "Q?", "reason": "why"}],
    })
    assert "| Faithfulness, LLM judge (2 answers) | 50% |" in report
    assert "answer_rejected: 1" in report and "**routing:** Q? why" in report
