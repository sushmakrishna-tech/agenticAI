"""Wires the nodes into the workflow.

    START -> router -> one or more agents (in parallel) -> merge -> compliance -> END
"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.common import explain
from src.agents.registry import AGENTS, as_node, build_agents
from src.core.llm import get_llm
from src.core.state import FinanceState
from src.workflow.compliance import compliance_node
from src.workflow.router import make_router

_NOT_PASSED = object()


def make_merge_node(llm):
    def merge_node(state: FinanceState) -> dict:
        """Combine agent answers. Parallel agents finish in any order, so sort first."""
        order = list(AGENTS)
        outputs = sorted(state["agent_outputs"], key=lambda o: order.index(o["agent"]))
        if len(outputs) == 1:
            draft = outputs[0]["content"]
        else:
            draft = "\n\n".join(f"### {AGENTS[o['agent']].title}\n{o['content']}" for o in outputs)
            working = [o for o in outputs if not o["error"]]
            if llm is not None and len(working) > 1:
                # Sections stay intact, so a poor summary can't lose an agent's figures.
                link = explain(llm, "In two sentences, say how these answers connect to the "
                                    "user's question. Add no new numbers.", draft, state, fallback="")
                if link:
                    draft = f"{link}\n\n{draft}"
        sources = list(dict.fromkeys(s for o in outputs for s in o["sources"]))
        return {"draft": draft, "sources": sources}

    return merge_node


def build_graph(llm=_NOT_PASSED, agent_functions=None, checkpointer=None):
    """Compile the graph. Arguments let tests swap in fakes, with no API calls."""
    llm = get_llm() if llm is _NOT_PASSED else llm
    functions = agent_functions or build_agents(llm)

    graph = StateGraph(FinanceState)
    graph.add_node("router", make_router(llm))
    graph.add_node("merge", make_merge_node(llm))
    graph.add_node("compliance", compliance_node)
    for name, fn in functions.items():
        graph.add_node(name, as_node(name, fn))
        graph.add_edge(name, "merge")  # merge waits until all selected agents finish

    graph.add_edge(START, "router")
    # Returning a list of node names makes LangGraph run them in parallel.
    graph.add_conditional_edges("router", lambda s: s["route"], list(functions))
    graph.add_edge("merge", "compliance")
    graph.add_edge("compliance", END)

    # The checkpointer saves state per thread_id: this is conversation memory.
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
