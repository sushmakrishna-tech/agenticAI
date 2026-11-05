"""MCP server: lets AI clients such as Claude Desktop use the assistant's tools.

MCP (Model Context Protocol) is a standard way for an AI app to call outside
tools. Each @tool below becomes one tool the client's model can call; FastMCP
builds the input schema from the type hints and the description from the docstring.

Most tools are plain calculations or data lookups with no LLM call, so the
client's model does the explaining. ask_finance_assistant runs the whole
LangGraph workflow, including its guardrails.

Run with: python -m src.mcp_server.server
Over stdio, stdout carries the protocol, so nothing else may print to it.
"""
import contextlib
import sys
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from src.utils.tracing import setup_tracing

with contextlib.redirect_stdout(sys.stderr):  # the Arize banner would corrupt stdio
    setup_tracing()  # must run before LangChain and LangGraph are imported

from langchain_core.messages import HumanMessage  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

from src.agents.market import OVERVIEW  # noqa: E402
from src.core.config import get_config  # noqa: E402
from src.data.market_data import MarketDataError, get_market_data  # noqa: E402
from src.data.news import get_news_service  # noqa: E402
from src.rag.store import get_store, search  # noqa: E402
from src.utils.goal_projections import GoalInput, project_goal as run_projection  # noqa: E402
from src.utils.portfolio_metrics import Holding, analyze_portfolio as run_analysis  # noqa: E402

Period = Literal["1mo", "3mo", "6mo", "1y"]
RiskProfile = Literal["conservative", "moderate", "aggressive"]
Category = Literal["basics", "markets", "planning", "products", "risk_portfolio", "tax"]
MAX_SYMBOLS = 10
MAX_HOLDINGS = 50
MAX_QUESTION_CHARS = 2000


class HoldingInput(BaseModel):
    symbol: str = Field(description="US ticker symbol, such as AAPL or VOO")
    shares: float = Field(gt=0)
    cost_per_share: float | None = Field(None, ge=0, description="Purchase price, if known")


def _plain(value):
    """Make values JSON-friendly: dates as text, numpy values as Python, floats to 2 decimals."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float):
        return round(value, 2)
    return value


def _note() -> str:
    return get_config()["compliance"]["disclaimer"]


def _quote(market, symbol: str) -> dict:
    quote = market.get_quote(symbol)
    return _plain({"symbol": quote.symbol, "name": OVERVIEW.get(quote.symbol, quote.symbol),
            "price": quote.price, "previous_close": quote.previous_close,
            "change_pct": quote.change_pct, "as_of": quote.as_of.isoformat(),
            "source": quote.source, "stale": quote.stale})


def create_server(get_market=get_market_data, get_kb=get_store, get_news=get_news_service,
                  get_graph=None) -> FastMCP:
    """Build the server. Services are passed as functions so tests can use fakes."""
    mcp = FastMCP(
        "ai-finance-assistant",
        instructions=(
            "Educational finance tools: market data, portfolio and goal calculations, a cited "
            "knowledge base, news, and US tax concepts. Results are educational, not personalized "
            "advice. Don't present them as buy or sell recommendations or as predictions."
        ),
    )
    graph_holder = {}

    def graph():
        if "graph" not in graph_holder:
            if get_graph:
                graph_holder["graph"] = get_graph()
            else:
                from src.workflow.graph import build_graph
                graph_holder["graph"] = build_graph()
        return graph_holder["graph"]

    # An error raised in a tool goes back to the client as a tool error, not a crash.
    @mcp.tool()
    def get_quote(symbols: list[str]) -> dict:
        """Latest price and daily change for up to 10 US tickers or indexes (such as ^GSPC).

        A quote marked stale came from an older cached value because live providers failed.
        """
        if not symbols or len(symbols) > MAX_SYMBOLS:
            raise ValueError(f"Pass between 1 and {MAX_SYMBOLS} symbols")
        market = get_market()
        quotes, errors = [], {}
        for symbol in symbols:
            try:
                quotes.append(_quote(market, symbol))
            except MarketDataError as exc:
                errors[symbol] = str(exc)
        return {"quotes": quotes, "errors": errors}

    @mcp.tool()
    def market_overview() -> dict:
        """Latest level of the S&P 500, Dow Jones Industrial Average, and Nasdaq Composite."""
        return get_quote(list(OVERVIEW))

    @mcp.tool()
    def get_price_history(symbol: str, period: Period = "6mo") -> dict:
        """Summary of a ticker's daily closes over a period: start, end, change, high, and low.

        Past performance doesn't predict future results.
        """
        history = get_market().get_history(symbol, period)
        closes = history.closes.dropna()
        if closes.empty:
            raise ValueError(f"No price history for {symbol}")
        start, end = float(closes.iloc[0]), float(closes.iloc[-1])
        return _plain({
            "symbol": history.symbol, "period": period, "days": len(closes),
            "start_date": closes.index[0].date().isoformat(), "start_close": start,
            "end_date": closes.index[-1].date().isoformat(), "end_close": end,
            "change_pct": (end / start - 1) * 100,
            "high": float(closes.max()), "low": float(closes.min()),
            "source": history.source, "stale": history.stale,
        })

    @mcp.tool()
    def analyze_portfolio(holdings: list[HoldingInput]) -> dict:
        """Value, allocation, gain, diversification, and one-year risk measures for holdings.

        Uses live prices. Risk measures compare the mix with the S&P 500 over the past year.
        """
        if not holdings or len(holdings) > MAX_HOLDINGS:
            raise ValueError(f"Pass between 1 and {MAX_HOLDINGS} holdings")
        analysis = run_analysis(
            [Holding(h.symbol, h.shares, h.cost_per_share) for h in holdings], get_market())
        result = asdict(analysis)
        result.pop("growth")  # a chart table, too large to be useful here
        return {**_plain(result), "note": _note()}

    @mcp.tool()
    def project_goal(target: float, years: float, current_savings: float = 0,
                     monthly_contribution: float = 0, risk_profile: RiskProfile = "moderate") -> dict:
        """Simulate saving toward a goal and estimate the chance of reaching it.

        target is in today's dollars and is adjusted for inflation. Returns the 10th, 50th,
        and 90th percentile outcomes and the monthly amount for an 80% chance of success.
        These use assumed returns, not predictions.
        """
        goal = GoalInput(target, years, current_savings, monthly_contribution, risk_profile)
        projection = asdict(run_projection(goal))
        for key in ("path_years", "path_p10", "path_p50", "path_p90"):
            projection.pop(key)
        return {**_plain(projection), "note": _note()}

    @mcp.tool()
    def search_knowledge_base(query: str, category: Category | None = None) -> dict:
        """Search the verified knowledge base of articles summarizing Investor.gov, FINRA,
        and IRS.gov. Returns matching passages with their sources; cite them when answering.

        An empty result means the knowledge base doesn't cover the topic.
        """
        cfg = get_config()["rag"]
        results = search(get_kb(), query[:MAX_QUESTION_CHARS], [category] if category else None,
                         cfg["top_k"], cfg["min_score"])
        return {"results": [
            {"title": r.title, "category": r.category, "score": round(r.score, 2),
             "source_name": r.source_name, "source_url": r.source_url, "text": r.text}
            for r in results
        ]}

    @mcp.tool()
    def search_news(query: str, symbol: str | None = None) -> dict:
        """Recent financial headlines matching a topic, or for one ticker if symbol is given."""
        articles = get_news().search(query[:MAX_QUESTION_CHARS], symbol)
        return {"articles": [
            {"title": a.title, "publisher": a.publisher, "url": a.url,
             "published": a.published.isoformat() if a.published else None}
            for a in articles
        ]}

    @mcp.tool()
    def ask_finance_assistant(question: str, knowledge_level: Literal[
            "beginner", "intermediate", "advanced"] = "beginner",
            risk_tolerance: RiskProfile = "moderate") -> dict:
        """Ask the full AI Finance Assistant, which routes to its six agents and applies its
        guardrails. Each call is a new conversation. Calls OpenAI, so it is slower and costs more
        than the other tools.
        """
        question = question.strip()
        if not question:
            raise ValueError("The question is empty")
        if len(question) > MAX_QUESTION_CHARS:
            raise ValueError(f"Questions are limited to {MAX_QUESTION_CHARS} characters")
        config = {"configurable": {"thread_id": f"mcp-{uuid.uuid4()}"}}
        result = graph().invoke(
            {"messages": [HumanMessage(content=question)],
             "profile": {"knowledge_level": knowledge_level, "risk_tolerance": risk_tolerance}},
            config,
        )
        return {"answer": result["messages"][-1].content, "agents": result["route"],
                "sources": result.get("sources", [])}

    return mcp


if __name__ == "__main__":
    create_server().run()  # stdio by default
