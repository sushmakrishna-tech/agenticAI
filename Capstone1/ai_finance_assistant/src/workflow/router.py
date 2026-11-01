"""Decides which agent(s) handle a message: LLM first, keyword rules as backup."""
import logging
import re

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.agents.registry import AGENTS
from src.core.state import AgentName, FinanceState
from src.utils.metrics import annotate, record

logger = logging.getLogger(__name__)
HISTORY_WINDOW = 6  # recent messages, so follow-ups like "and the taxes?" make sense

KEYWORDS = {
    "tax": r"tax(es)?|roth|ira|401\(?k\)?|capital gains?|deductions?",
    "portfolio": r"portfolio|my holdings|diversif\w*|allocation|rebalanc\w*",
    "market": r"price|quote|stock market|s&p|nasdaq|dow|trading at",
    "goals": r"goals?|retire(ment)?|sav(e|ing) for|down payment|college fund",
    "news": r"news|headlines?",
}


class RouteDecision(BaseModel):
    """Structured output: forces the LLM to answer with valid agent names."""

    agents: list[AgentName] = Field(min_length=1)


def keyword_route(text: str) -> list[str]:
    matches = [
        name for name, pattern in KEYWORDS.items()
        if re.search(rf"\b({pattern})\b", text, re.IGNORECASE)
    ]
    return matches or ["finance_qa"]


def _prompt() -> str:
    agents = "\n".join(f"- {name}: {a.description}" for name, a in AGENTS.items())
    return (
        "Route the user's latest message to the fewest agents that can answer it. Most messages "
        "need one. Choose several only if it clearly asks about separate areas. Use the earlier "
        "messages to interpret follow-ups.\n"
        "- US tax rules, contribution limits, retirement account rules, and early withdrawals go "
        "to tax alone.\n"
        "- How a market, index, or stock has performed or is priced goes to market alone, unless "
        "news is requested.\n"
        "- A personal savings target with amounts or a timeline goes to goals alone.\n"
        "- Whether to buy or sell a specific stock goes to market.\n"
        "- Stated holdings with a share count, plus a question about tax on selling them, go to "
        "portfolio and tax together.\n"
        "- Anything unrelated to personal finance or investing goes to finance_qa alone, "
        "which will decline it.\n\n"
        f"Agents:\n{agents}"
    )


def make_router(llm):
    """Build the router node. llm may be None, which means keyword rules only."""

    def router(state: FinanceState) -> dict:
        messages = state["messages"]
        latest = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
        route, source = None, "keyword"
        if llm is not None:
            try:
                decision = llm.with_structured_output(RouteDecision).invoke(
                    [("system", _prompt()), *messages[-HISTORY_WINDOW:]]
                )
                route, source = list(dict.fromkeys(decision.agents)), "llm"
            except Exception:
                logger.exception("LLM routing failed; using keyword rules")
                record("routing_fallback")
        # agent_outputs=None clears the previous turn (see collect_outputs).
        route = route or keyword_route(str(latest.content))
        annotate(route=route, route_source=source)
        return {
            "route": route,
            "route_source": source,
            "agent_outputs": None,
        }

    return router
