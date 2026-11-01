"""The six agents. Each is a plain function: state in, one AgentOutput out.

These are STUBS: they prove routing and orchestration work. Real logic arrives
in build step 4, and each agent can then move into its own module.
"""
import logging
from collections.abc import Callable
from dataclasses import dataclass

from src.core.state import AgentOutput, FinanceState

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


def _stub(name: str) -> AgentFn:
    def run(state: FinanceState) -> AgentOutput:
        return AgentOutput(
            agent=name,
            content=f"[stub] {AGENTS[name].title} agent is not implemented yet.",
            sources=[],
            error=False,
        )

    return run


AGENT_FUNCTIONS: dict[str, AgentFn] = {name: _stub(name) for name in AGENTS}


def as_node(name: str, fn: AgentFn):
    """Wrap an agent as a graph node, so one failing agent can't sink the turn."""

    def node(state: FinanceState) -> dict:
        try:
            output = fn(state)
        except Exception:
            logger.exception("Agent %s failed", name)
            output = AgentOutput(
                agent=name,
                content=f"The {AGENTS[name].title} agent is unavailable right now.",
                sources=[],
                error=True,
            )
        return {"agent_outputs": [output]}  # a list, because the reducer concatenates

    return node
