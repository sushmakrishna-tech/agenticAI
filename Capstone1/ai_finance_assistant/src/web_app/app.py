"""Streamlit UI. Run from the project root: streamlit run src/web_app/app.py"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # makes `src` importable

import streamlit as st
from langchain_core.messages import HumanMessage

from src.agents.registry import AGENTS
from src.workflow.graph import build_graph


@st.cache_resource  # build once per server, or conversation memory would reset
def load_graph():
    return build_graph()


st.set_page_config(page_title="AI Finance Assistant", layout="wide")
st.title("AI Finance Assistant")

# Streamlit reruns this script on every interaction; session_state survives reruns.
st.session_state.setdefault("thread_id", str(uuid.uuid4()))
st.session_state.setdefault("chat", [])

chat_tab, portfolio_tab, market_tab, goals_tab = st.tabs(["Chat", "Portfolio", "Market", "Goals"])

with chat_tab:
    for turn in st.session_state.chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("caption"):
                st.caption(turn["caption"])

with portfolio_tab:
    st.info("The portfolio dashboard arrives in build step 5.")
with market_tab:
    st.info("The market overview arrives in build step 5.")
with goals_tab:
    st.info("The goal planner arrives in build step 5.")

# st.chat_input stays outside the tabs so it remains pinned to the page bottom.
if question := st.chat_input("Ask about investing, your portfolio, taxes..."):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    result = load_graph().invoke({"messages": [HumanMessage(content=question)]}, config)
    used = ", ".join(AGENTS[name].title for name in result["route"])
    st.session_state.chat += [
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": result["messages"][-1].content,
            "caption": f"Agents used: {used} (routed by {result['route_source']})",
        },
    ]
    st.rerun()
