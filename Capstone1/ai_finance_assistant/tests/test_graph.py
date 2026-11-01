from langchain_core.messages import HumanMessage

from src.agents.registry import AGENTS
from src.core.state import AgentOutput, collect_outputs
from src.workflow.graph import build_graph


def fake(name, **overrides):
    def run(state):
        return AgentOutput(**{"agent": name, "content": f"{name} answer", "sources": [], "error": False, **overrides})
    return run


def make(**replacements):
    functions = {name: fake(name) for name in AGENTS} | replacements
    return build_graph(llm=None, agent_functions=functions)


def ask(graph, text, thread="t1"):
    return graph.invoke(
        {"messages": [HumanMessage(text)]},
        {"configurable": {"thread_id": thread}},
    )


def test_single_agent_turn():
    result = ask(make(), "What is an ETF?")
    assert result["route"] == ["finance_qa"]
    assert result["messages"][-1].content.startswith("finance_qa answer")
    assert "not personalized" in result["messages"][-1].content


def test_multi_agent_merge_has_stable_order():
    result = ask(make(), "Will selling for my down payment raise my taxes?")
    answer = result["messages"][-1].content
    assert answer.index("### Goal Planning") < answer.index("### Tax Education")


def test_memory_kept_but_outputs_reset_between_turns():
    graph = make()
    ask(graph, "What is a Roth IRA?")
    result = ask(graph, "Latest news")
    assert len(result["messages"]) == 4
    assert [o["agent"] for o in result["agent_outputs"]] == ["news"]
    assert "tax answer" not in result["draft"]


def test_threads_are_isolated():
    graph = make()
    ask(graph, "What is a Roth IRA?", thread="a")
    assert len(ask(graph, "What is an ETF?", thread="b")["messages"]) == 2


def test_failed_agent_does_not_sink_the_others():
    def boom(state):
        raise RuntimeError("API down")

    result = ask(make(goals=boom), "Will selling for my down payment raise my taxes?")
    answer = result["messages"][-1].content
    assert "Goal Planning agent is unavailable" in answer
    assert "tax answer" in answer


def test_sources_are_deduplicated():
    graph = make(
        tax=fake("tax", sources=["irs.gov"]),
        goals=fake("goals", sources=["irs.gov", "sec.gov"]),
    )
    result = ask(graph, "Will saving for my down payment raise my taxes?")
    assert result["sources"] == ["irs.gov", "sec.gov"]


def test_reducer_clears_on_none():
    out = AgentOutput(agent="tax", content="x", sources=[], error=False)
    assert collect_outputs([out], None) == []
    assert collect_outputs([out], [out]) == [out, out]


def test_merge_adds_a_link_between_sections_and_keeps_them():
    from src.workflow.graph import make_merge_node

    class LLM:
        def invoke(self, messages):
            from langchain_core.messages import AIMessage
            return AIMessage("These connect.")

    outputs = [AgentOutput(agent=a, content=f"{a} answer", sources=[], error=False) for a in ("tax", "goals")]
    state = {"agent_outputs": outputs, "messages": [HumanMessage("q")]}
    draft = make_merge_node(LLM())(state)["draft"]
    assert draft.startswith("These connect.")
    assert "goals answer" in draft and "tax answer" in draft


def test_merge_skips_the_link_when_an_agent_failed():
    from src.workflow.graph import make_merge_node

    class LLM:
        def invoke(self, messages):
            raise AssertionError("should not be called")

    outputs = [AgentOutput(agent="tax", content="ok", sources=[], error=False),
               AgentOutput(agent="goals", content="down", sources=[], error=True)]
    make_merge_node(LLM())({"agent_outputs": outputs, "messages": [HumanMessage("q")]})
