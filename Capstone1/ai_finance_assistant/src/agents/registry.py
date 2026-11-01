"""The six agents. Each is a plain function: state in, one AgentOutput out."""
import logging
from collections.abc import Callable
from dataclasses import dataclass

from src.agents.goals import make_goals_agent
from src.agents.knowledge import make_knowledge_agent
from src.agents.market import make_market_agent
from src.agents.news import make_news_agent
from src.agents.portfolio import make_portfolio_agent
from src.core.state import AgentOutput, FinanceState
from src.data.market_data import get_market_data
from src.data.news import get_news_service
from src.rag.store import get_store as _default_store
from src.utils.metrics import annotate, record

logger = logging.getLogger(__name__)
AgentFn = Callable[[FinanceState], AgentOutput]


@dataclass(frozen=True)
class AgentInfo:
    title: str
    description: str  # the router's LLM reads this to choose an agent


AGENTS: dict[str, AgentInfo] = {
    "finance_qa": AgentInfo("Finance Q&A", "General investing and personal finance concepts."),
    "portfolio": AgentInfo("Portfolio Analysis", "Reviews the user's own holdings, allocation, and diversification."),
    "market": AgentInfo("Market Analysis", "Current prices, indexes, and market trends."),
    "goals": AgentInfo("Goal Planning", "Saving and investing toward goals such as retirement or a home."),
    "news": AgentInfo("News Synthesizer", "Summarizes recent financial news."),
    "tax": AgentInfo("Tax Education", "US tax concepts and account types such as 401(k) and Roth IRA."),
}


def build_agents(llm, get_market=None, get_store=None, get_news=None) -> dict[str, AgentFn]:
    """Create all agents. Services can be swapped for fakes in tests.

    Services are passed as functions, so nothing connects until an agent needs it.
    """
    get_market = get_market or get_market_data
    get_store = get_store or _default_store
    get_news = get_news or get_news_service
    qa_task = "Answer using the facts, as a clear explanation for a learner."
    tax_task = ("Answer using the facts, and state the tax year of any amounts. These are general US "
                "federal rules, so note when the answer depends on personal circumstances.")
    return {
        "finance_qa": make_knowledge_agent("finance_qa", None, qa_task, llm, get_store),
        "portfolio": make_portfolio_agent(llm, get_market),
        "market": make_market_agent(llm, get_market),
        "goals": make_goals_agent(llm),
        "news": make_news_agent(llm, get_news),
        "tax": make_knowledge_agent("tax", ["tax"], tax_task, llm, get_store),
    }


def as_node(name: str, fn: AgentFn):
    """Wrap an agent as a graph node, so one failing agent can't sink the turn."""

    def node(state: FinanceState) -> dict:
        try:
            output = fn(state)
        except Exception:
            logger.exception("Agent %s failed", name)
            record("agent_exception", agent=name)
            output = AgentOutput(
                agent=name,
                content=f"The {AGENTS[name].title} agent is unavailable right now.",
                sources=[],
                error=True,
            )
        annotate(agent=name, agent_error=output["error"], cited_sources=len(output["sources"]))
        return {"agent_outputs": [output]}  # a list, because the reducer concatenates

    return node
