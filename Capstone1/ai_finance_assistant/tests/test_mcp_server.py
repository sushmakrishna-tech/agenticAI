"""MCP tools, called through FastMCP the way a client would, with no network."""
import asyncio
import json
from datetime import datetime, timezone

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from mcp.server.fastmcp.exceptions import ToolError

from src.data.news import Article
from src.mcp_server.server import create_server
from tests.fakes import FakeMarket


class FakeKB:
    def similarity_search_with_score(self, query, k, filter=None):
        self.filter = filter
        meta = {"title": "Bonds", "category": "basics", "source_name": "Investor.gov",
                "source_url": "https://www.investor.gov/bonds"}
        return [(Document("A bond is a loan.", metadata=meta), 0.2),  # score 0.8
                (Document("Unrelated.", metadata=meta), 0.9)]  # score 0.1, below the cutoff


class FakeNews:
    def search(self, query, symbol=None):
        self.args = (query, symbol)
        return [Article("Apple shares rise", "Reuters", "https://example.com/a",
                        datetime(2026, 11, 1, tzinfo=timezone.utc))]


class FakeGraph:
    def invoke(self, state, config):
        self.state, self.config = state, config
        return {"messages": [AIMessage("A bond is a loan.")], "route": ["finance_qa"],
                "sources": ["Investor.gov"]}


@pytest.fixture
def fakes():
    kb, news, graph = FakeKB(), FakeNews(), FakeGraph()
    server = create_server(get_market=FakeMarket, get_kb=lambda: kb, get_news=lambda: news,
                           get_graph=lambda: graph)
    return server, kb, news, graph


def call(server, tool, args=None):
    content = asyncio.run(server.call_tool(tool, args or {}))
    return json.loads(content[0].text)


def test_lists_all_tools(fakes):
    names = {t.name for t in asyncio.run(fakes[0].list_tools())}
    assert names == {"get_quote", "market_overview", "get_price_history", "analyze_portfolio",
                     "project_goal", "search_knowledge_base", "search_news", "ask_finance_assistant"}


def test_quote_reports_unknown_symbols_separately(fakes):
    result = call(fakes[0], "get_quote", {"symbols": ["aapl", "ZZZZ"]})
    assert result["quotes"][0]["symbol"] == "AAPL" and result["quotes"][0]["price"] == 200
    assert list(result["errors"]) == ["ZZZZ"]


def test_quote_limits_symbol_count(fakes):
    with pytest.raises(ToolError):
        call(fakes[0], "get_quote", {"symbols": ["AAPL"] * 11})


def test_market_overview_names_the_indexes(fakes):
    names = {q["name"] for q in call(fakes[0], "market_overview")["quotes"]}
    assert names == {"S&P 500", "Dow Jones Industrial Average", "Nasdaq Composite"}


def test_price_history_summary(fakes):
    result = call(fakes[0], "get_price_history", {"symbol": "SPY", "period": "3mo"})
    assert result["days"] == 120 and result["low"] <= result["end_close"] <= result["high"]
    expected = (result["end_close"] / result["start_close"] - 1) * 100
    assert result["change_pct"] == pytest.approx(expected, abs=0.01)


def test_price_history_rejects_unknown_period(fakes):
    with pytest.raises(ToolError):
        call(fakes[0], "get_price_history", {"symbol": "SPY", "period": "10y"})


def test_numbers_are_rounded_for_the_client(fakes):
    quote = call(fakes[0], "get_quote", {"symbols": ["AAPL"]})["quotes"][0]
    assert quote["change_pct"] == 1.01 and quote["previous_close"] == 198.0
    result = call(fakes[0], "project_goal", {"target": 12345.678, "years": 2})
    assert all(round(v, 2) == v for v in result.values() if isinstance(v, float))


def test_analyze_portfolio(fakes):
    result = call(fakes[0], "analyze_portfolio", {"holdings": [
        {"symbol": "AAPL", "shares": 10, "cost_per_share": 150}, {"symbol": "VOO", "shares": 6}]})
    assert result["total_value"] == 5000
    assert {p["symbol"] for p in result["positions"]} == {"AAPL", "VOO"}
    assert "growth" not in result and "not personalized" in result["note"].lower()


def test_analyze_portfolio_rejects_negative_shares(fakes):
    with pytest.raises(ToolError):
        call(fakes[0], "analyze_portfolio", {"holdings": [{"symbol": "AAPL", "shares": -1}]})


def test_project_goal(fakes):
    result = call(fakes[0], "project_goal", {"target": 50000, "years": 10,
                                             "monthly_contribution": 300})
    assert result["risk_profile"] == "moderate"
    assert result["final_p10"] <= result["final_p50"] <= result["final_p90"]
    assert "path_p50" not in result


def test_project_goal_explains_invalid_input(fakes):
    with pytest.raises(ToolError, match="greater than 0"):
        call(fakes[0], "project_goal", {"target": 0, "years": 5})


def test_knowledge_search_keeps_only_good_matches_with_sources(fakes):
    server, kb = fakes[0], fakes[1]
    result = call(server, "search_knowledge_base", {"query": "What is a bond?", "category": "basics"})
    assert [r["text"] for r in result["results"]] == ["A bond is a loan."]
    assert result["results"][0]["source_url"].startswith("https://www.investor.gov")
    assert kb.filter == {"category": {"$in": ["basics"]}}


def test_news_search(fakes):
    server, news = fakes[0], fakes[2]
    result = call(server, "search_news", {"query": "Apple", "symbol": "AAPL"})
    assert result["articles"][0]["publisher"] == "Reuters" and news.args == ("Apple", "AAPL")


def test_ask_runs_the_graph_in_a_new_conversation(fakes):
    server, graph = fakes[0], fakes[3]
    result = call(server, "ask_finance_assistant", {"question": "What is a bond?"})
    assert result == {"answer": "A bond is a loan.", "agents": ["finance_qa"],
                      "sources": ["Investor.gov"]}
    first = graph.config["configurable"]["thread_id"]
    call(server, "ask_finance_assistant", {"question": "And a stock?"})
    assert graph.config["configurable"]["thread_id"] != first
    assert graph.state["profile"]["knowledge_level"] == "beginner"


@pytest.mark.parametrize("question", ["   ", "x" * 2001])
def test_ask_rejects_empty_or_long_questions(fakes, question):
    with pytest.raises(ToolError):
        call(fakes[0], "ask_finance_assistant", {"question": question})
