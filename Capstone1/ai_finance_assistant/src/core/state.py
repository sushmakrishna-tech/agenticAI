"""The shared state: the one object every node in the graph reads and updates.

A node never mutates state. It returns a dict of changes, and LangGraph merges
them in. A *reducer* (the function inside Annotated[...]) says how to merge a
field; a field without one is simply overwritten.
"""
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

AgentName = Literal["finance_qa", "portfolio", "market", "goals", "news", "tax"]


class AgentOutput(TypedDict):
    agent: str
    content: str
    sources: list[str]
    error: bool


def collect_outputs(
    current: list[AgentOutput] | None, update: list[AgentOutput] | None
) -> list[AgentOutput]:
    """Combine results from agents running in parallel; None clears the list.

    Conversation state persists between turns, so the router sends None at the
    start of each turn. Otherwise last turn's answers would leak into this one.
    """
    if update is None:
        return []
    return (current or []) + update


class UserProfile(TypedDict, total=False):
    knowledge_level: Literal["beginner", "intermediate", "advanced"]
    risk_tolerance: Literal["conservative", "moderate", "aggressive"]
    goals: list[str]


class FinanceState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # appended, never replaced
    profile: UserProfile
    route: list[AgentName]
    route_source: Literal["llm", "keyword"]
    agent_outputs: Annotated[list[AgentOutput], collect_outputs]
    draft: str
    sources: list[str]
    compliance_flags: list[str]
