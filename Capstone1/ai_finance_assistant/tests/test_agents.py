"""Agent behavior with fake services and a fake LLM, so tests make no API calls."""
from datetime import datetime, timezone

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.common import RULES, explain, invented_figures, money
from src.agents.goals import GoalRequest, make_goals_agent
from src.agents.knowledge import NOT_FOUND, make_knowledge_agent, standalone_question
from src.agents.market import MarketRequest, make_market_agent
from src.agents.news import NewsRequest, make_news_agent
from src.agents.portfolio import HoldingDetail, HoldingsRequest, make_portfolio_agent
from src.agents.registry import AGENTS, build_agents
from src.data.market_data import MarketDataError, PriceHistory, Quote
from src.data.news import Article, NewsError
from src.rag.store import KnowledgeBaseError, SearchResult

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)


class FakeLLM:
    """Returns set structured results, and records what it was asked to explain."""

    def __init__(self, structured=None, text="LLM explanation", fail=False):
        self.structured, self.text, self.fail = structured, text, fail
        self.prompts = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.prompts.append(messages)
        if self.fail:
            raise RuntimeError("API down")
        if self.structured is not None:
            result, self.structured = self.structured, None
            return result
        return AIMessage(self.text)


class FakeMarket:
    def __init__(self, prices=None, stale=False):
        self.prices = {"AAPL": 100.0, "^GSPC": 7000.0, "^DJI": 40000.0, "^IXIC": 20000.0} if prices is None else prices
        self.stale = stale

    def get_quote(self, symbol):
        if symbol not in self.prices:
            raise MarketDataError(symbol)
        return Quote(symbol, self.prices[symbol], 99.0, 1.5, NOW, "fake", self.stale)

    def get_history(self, symbol, period="1y"):
        raise MarketDataError(symbol)


def state(*texts, **extra):
    return {"messages": [HumanMessage(t) for t in texts], **extra}


def result(title="Roth IRA", score=0.6):
    return SearchResult(f"{title} text", score, title, "tax", "IRS", "https://www.irs.gov/x")


def knowledge(monkeypatch, results=None, error=None, llm=None, categories=("tax",)):
    seen = {}

    def fake_search(store, query, cats, k, min_score):
        seen.update(query=query, categories=cats)
        if error:
            raise error
        return results or []

    monkeypatch.setattr("src.agents.knowledge.search", fake_search)
    return make_knowledge_agent("tax", list(categories), "task", llm, lambda: object()), seen


def test_knowledge_answer_cites_sources(monkeypatch):
    agent, seen = knowledge(monkeypatch, [result(), result()], llm=FakeLLM())
    out = agent(state("What is a Roth IRA?"))
    assert out["content"] == "LLM explanation"
    assert out["sources"] == ["Roth IRA (IRS): https://www.irs.gov/x"]
    assert seen["categories"] == ["tax"]


def test_knowledge_refuses_to_guess_without_results(monkeypatch):
    llm = FakeLLM()
    agent, _ = knowledge(monkeypatch, [], llm=llm)
    assert agent(state("banana bread?"))["content"] == NOT_FOUND
    assert llm.prompts == []


def test_knowledge_failure_is_reported(monkeypatch):
    agent, _ = knowledge(monkeypatch, error=KnowledgeBaseError("no index"))
    assert agent(state("x"))["error"] is True


def test_knowledge_without_llm_shows_retrieved_text(monkeypatch):
    agent, _ = knowledge(monkeypatch, [result()], llm=None)
    assert "Roth IRA text" in agent(state("x"))["content"]


def test_follow_up_is_rewritten_for_search():
    llm = FakeLLM(text="What is the Roth IRA limit?")
    assert standalone_question(llm, state("What is a Roth IRA?", "and the limit?")) == "What is the Roth IRA limit?"


def test_first_question_is_not_rewritten():
    llm = FakeLLM()
    assert standalone_question(llm, state("What is a Roth IRA?")) == "What is a Roth IRA?"
    assert llm.prompts == []


def test_rewrite_failure_uses_original_question():
    assert standalone_question(FakeLLM(fail=True), state("first", "second")) == "second"


def test_explain_prompt_carries_rules_level_and_facts():
    llm = FakeLLM()
    explain(llm, "task", "FACT-123", state("q", profile={"knowledge_level": "advanced"}))
    system, human = llm.prompts[0]
    assert RULES in system[1] and "experienced" in system[1]
    assert "FACT-123" in human[1]


def test_explain_shows_facts_when_llm_fails():
    assert explain(FakeLLM(fail=True), "task", "FACTS", state("q")) == "FACTS"


def test_portfolio_numbers_reach_the_llm_from_code():
    llm = FakeLLM(HoldingsRequest(holdings=[HoldingDetail(symbol="AAPL", shares=2, cost_per_share=50)]))
    out = make_portfolio_agent(llm, FakeMarket)(state("I own 2 AAPL at 50"))
    assert out["content"] == "LLM explanation"
    prompt = llm.prompts[-1][1][1]
    assert "$200.00" in prompt and "$100.00" in prompt and "AAPL" in prompt


def test_portfolio_without_llm_still_gives_figures():
    out = make_portfolio_agent(None, FakeMarket)(state("x", holdings=[{"symbol": "AAPL", "shares": 2}]))
    assert "Total value: $200.00" in out["content"]


def test_portfolio_prefers_holdings_from_the_tab():
    llm = FakeLLM()
    make_portfolio_agent(llm, FakeMarket)(state("analyze", holdings=[{"symbol": "AAPL", "shares": 1}]))
    assert len(llm.prompts) == 1  # explanation only, no extraction call


def test_portfolio_asks_when_no_holdings():
    out = make_portfolio_agent(FakeLLM(HoldingsRequest(holdings=[])), FakeMarket)(state("analyze my portfolio"))
    assert "shares" in out["content"]


def test_portfolio_bad_input_is_explained():
    out = make_portfolio_agent(None, FakeMarket)(state("x", holdings=[{"symbol": "AAPL", "shares": -1}]))
    assert "shares must be greater than 0" in out["content"]


def test_portfolio_no_prices_is_an_error():
    out = make_portfolio_agent(None, lambda: FakeMarket({}))(state("x", holdings=[{"symbol": "AAPL", "shares": 1}]))
    assert out["error"] is True


def test_market_quotes_requested_symbol():
    llm = FakeLLM(MarketRequest(symbols=["AAPL"]))
    out = make_market_agent(llm, FakeMarket)(state("AAPL price?"))
    assert "$100.00" in llm.prompts[-1][1][1]
    assert out["sources"] == ["Market data: fake"]


def test_market_general_question_uses_index_overview_without_dollars():
    out = make_market_agent(None, FakeMarket)(state("how is the market?"))
    assert "S&P 500: 7,000.00" in out["content"]
    assert "Nasdaq" in out["content"] and "$" not in out["content"]


def test_market_unknown_symbol_is_reported():
    llm = FakeLLM(MarketRequest(symbols=["AAPL", "NOPE"]))
    make_market_agent(llm, FakeMarket)(state("x"))
    assert "NOPE: no data available" in llm.prompts[-1][1][1]


def test_market_marks_stale_prices():
    out = make_market_agent(None, lambda: FakeMarket(stale=True))(state("x"))
    assert "may be out of date" in out["content"]


def test_market_includes_one_month_trend():
    class WithHistory(FakeMarket):
        def get_history(self, symbol, period="1y"):
            return PriceHistory(symbol, pd.Series([100.0, 110.0]), "fake")

    assert "+10.0% over about one month" in make_market_agent(None, WithHistory)(state("x"))["content"]


def test_market_all_failed_is_an_error():
    assert make_market_agent(None, lambda: FakeMarket({}))(state("x"))["error"] is True


def goal(**fields):
    return FakeLLM(GoalRequest(**fields))


def test_goals_uses_profile_risk_and_code_figures():
    llm = goal(target=50000, years=10, current_savings=1000, monthly_contribution=300)
    out = make_goals_agent(llm)(state("goal", profile={"risk_tolerance": "conservative"}))
    assert out["content"] == "LLM explanation"
    prompt = llm.prompts[-1][1][1]
    assert "Selected risk level: conservative" in prompt
    assert "of simulations reach the target" in prompt


def test_goals_stated_risk_overrides_profile():
    llm = goal(target=50000, years=10, risk_profile="aggressive")
    make_goals_agent(llm)(state("goal", profile={"risk_tolerance": "conservative"}))
    assert "Selected risk level: aggressive" in llm.prompts[-1][1][1]


def test_goals_discloses_every_assumption():
    llm = goal(target=50000, years=10)
    make_goals_agent(llm)(state("goal"))
    prompt = llm.prompts[-1][1][1]
    assert "current savings of $0" in prompt and "monthly contribution of $0" in prompt and "moderate risk" in prompt


def test_goals_asks_for_missing_details():
    out = make_goals_agent(goal(target=50000))(state("goal"))
    assert "how many years" in out["content"] and "target amount" not in out["content"]


def test_goals_explains_invalid_values():
    out = make_goals_agent(goal(target=50000, years=500))(state("goal"))
    assert "Years must be" in out["content"]


def test_goals_without_llm_asks_rather_than_guessing():
    assert "target amount" in make_goals_agent(None)(state("goal"))["content"]


class FakeNews:
    def __init__(self, articles=(), error=None):
        self.articles, self.error = list(articles), error
        self.query = None

    def search(self, query, symbol=None):
        self.query, self.symbol = query, symbol
        if self.error:
            raise self.error
        return self.articles


def test_news_summarizes_real_headlines_with_links():
    service = FakeNews([Article("Apple rises", "IBD", "https://x.test/a", NOW)])
    llm = FakeLLM(NewsRequest(query="Apple", symbol="aapl"))
    out = make_news_agent(llm, lambda: service)(state("Apple news?"))
    assert service.query == "Apple" and service.symbol == "AAPL"
    assert "Apple rises (IBD, 2026-10-30)" in llm.prompts[-1][1][1]
    assert out["sources"] == ["Apple rises (IBD): https://x.test/a"]


def test_news_empty_and_failed():
    assert "couldn't find" in make_news_agent(None, FakeNews)(state("x"))["content"]
    out = make_news_agent(None, lambda: FakeNews(error=NewsError("down")))(state("x"))
    assert out["error"] is True


def test_build_agents_creates_all_six_without_connecting():
    def boom():
        raise AssertionError("connected too early")

    assert set(build_agents(None, boom, boom, boom)) == set(AGENTS)


def test_money_format():
    assert money(1234.5) == "$1,234.50"


def test_invented_figures_finds_new_amounts_only():
    facts = "Value $1,234.50, up 6.0% and 20 percent"
    assert invented_figures("It is $1,234.50 and up 6% or 20%", facts) == set()
    assert invented_figures("You would have $40,000", facts) == {("$", 40000.0)}
    assert invented_figures("about 7.5 percent", facts) == {("%", 7.5)}
    assert invented_figures("in 5 years, or 60 months", facts) == set()


def test_explain_discards_answers_with_invented_figures():
    llm = FakeLLM(text="You will have $40,000.")
    assert explain(llm, "task", "Target is $60,000", state("q")) == "Target is $60,000"


def test_explain_allows_figures_the_user_stated():
    llm = FakeLLM(text="You mentioned $500 a month.")
    assert explain(llm, "t", "facts", state("I save $500 a month")) == "You mentioned $500 a month."


def test_explain_fallback_can_be_empty_for_optional_text():
    assert explain(FakeLLM(text="Costs $9"), "t", "facts", state("q"), fallback="") == ""


def test_weak_matches_are_not_used_or_cited(monkeypatch):
    agent, _ = knowledge(monkeypatch, [result("Top", 0.6), result("Close", 0.52), result("Weak", 0.4)], llm=None)
    out = agent(state("x"))
    assert [s.split(" (")[0] for s in out["sources"]] == ["Top", "Close"]
    assert "Weak" not in out["content"]


def test_news_ignores_an_invalid_symbol():
    service = FakeNews([Article("Apple rises", "IBD", "https://x.test/a", NOW)])
    make_news_agent(FakeLLM(NewsRequest(query="Apple", symbol="not a ticker")), lambda: service)(state("x"))
    assert service.symbol is None


def test_invented_years_are_caught():
    assert invented_figures("data up to October 2023", "as of 2026-10-30") == {("year", 2023)}
    assert invented_figures("as of October 30, 2026", "as of 2026-10-30") == set()
