import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.workflow.router import RouteDecision, keyword_route, make_router


class FakeLLM:
    """Stands in for ChatOpenAI: returns a set decision or raises an error."""

    def __init__(self, result):
        self.result = result
        self.seen = None

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.seen = messages
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.mark.parametrize("text,expected", [
    ("What is a Roth IRA?", ["tax"]),
    ("Latest news headlines", ["news"]),
    ("What is an ETF?", ["finance_qa"]),
    ("What is the syntax for a dowel?", ["finance_qa"]),  # no partial-word matches
    ("Will selling for my down payment raise my taxes?", ["tax", "goals"]),
])
def test_keyword_route(text, expected):
    assert keyword_route(text) == expected


def test_llm_decision_is_used_and_deduplicated():
    llm = FakeLLM(RouteDecision(agents=["tax", "tax", "goals"]))
    result = make_router(llm)({"messages": [HumanMessage("anything")]})
    assert result["route"] == ["tax", "goals"]
    assert result["route_source"] == "llm"
    assert result["agent_outputs"] is None


def test_llm_failure_falls_back_to_keywords():
    router = make_router(FakeLLM(RuntimeError("API down")))
    result = router({"messages": [HumanMessage("What is a Roth IRA?")]})
    assert result["route"] == ["tax"]
    assert result["route_source"] == "keyword"


def test_router_sends_history_for_follow_ups():
    llm = FakeLLM(RouteDecision(agents=["tax"]))
    history = [HumanMessage("Should I sell AAPL?"), AIMessage("..."), HumanMessage("And the taxes?")]
    make_router(llm)({"messages": history})
    assert llm.seen[1:] == history


def test_invalid_agent_name_is_rejected():
    with pytest.raises(ValueError):
        RouteDecision(agents=["astrology"])
    with pytest.raises(ValueError):
        RouteDecision(agents=[])


def test_router_prompt_pairs_holdings_with_tax():
    from src.workflow.router import _prompt

    assert "portfolio and tax together" in _prompt()
