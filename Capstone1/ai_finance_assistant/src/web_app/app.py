"""Streamlit UI. Run from the project root: streamlit run src/web_app/app.py"""
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # makes `src` importable

from src.utils.tracing import setup_tracing

setup_tracing()  # must run before LangChain and LangGraph are imported

import streamlit as st
from langchain_core.messages import HumanMessage

from openinference.instrumentation import using_session

from src.agents.registry import AGENTS
from src.utils.feedback import save_feedback
from src.workflow.graph import build_graph

logger = logging.getLogger(__name__)


@st.cache_resource  # build once per server, or conversation memory would reset
def load_graph():
    return build_graph()


st.set_page_config(page_title="AI Finance Assistant", layout="wide")
st.title("AI Finance Assistant")

# Streamlit reruns this script on every interaction; session_state survives reruns.
st.session_state.setdefault("thread_id", str(uuid.uuid4()))
st.session_state.setdefault("chat", [])

with st.sidebar:
    st.header("Your profile")
    profile = {
        "knowledge_level": st.selectbox("Knowledge level", ["beginner", "intermediate", "advanced"]),
        "risk_tolerance": st.selectbox("Risk tolerance", ["conservative", "moderate", "aggressive"], index=1),
    }

chat_tab, portfolio_tab, market_tab, goals_tab = st.tabs(["Chat", "Portfolio", "Market", "Goals"])

with chat_tab:
    for index, turn in enumerate(st.session_state.chat):
        with st.chat_message(turn["role"]):
            # Streamlit treats text between dollar signs as math, so escape them.
            st.markdown(turn["content"].replace("$", r"\$"))
            for source in turn.get("sources", []):
                st.caption(f"Source: {source}".replace("$", r"\$"))
            if turn.get("caption"):
                st.caption(turn["caption"])
            if turn.get("agents"):
                rating = st.feedback("thumbs", key=f"feedback_{index}")
                if rating is not None and not turn.get("rated"):
                    turn["rated"] = True
                    save_feedback(rating == 1, st.session_state.thread_id, turn["agents"])

with portfolio_tab:
    st.info("The portfolio dashboard arrives in build step 5.")
with market_tab:
    st.info("The market overview arrives in build step 5.")
with goals_tab:
    st.info("The goal planner arrives in build step 5.")

# st.chat_input stays outside the tabs so it remains pinned to the page bottom.
if question := st.chat_input("Ask about investing, your portfolio, taxes..."):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    st.session_state.chat.append({"role": "user", "content": question})
    try:
        # Groups all of a conversation's traces into one session in Arize.
        with st.spinner("Thinking..."), using_session(st.session_state.thread_id):
            result = load_graph().invoke(
                {"messages": [HumanMessage(content=question)], "profile": profile}, config
            )
    except Exception:
        logger.exception("Workflow failed")
        reply = {"role": "assistant", "content": "Something went wrong. Please try again."}
    else:
        used = ", ".join(AGENTS[name].title for name in result["route"])
        reply = {
            "role": "assistant",
            "content": result["messages"][-1].content,
            "sources": result.get("sources", []),
            "caption": f"Agents used: {used} (routed by {result['route_source']})",
            "agents": result["route"],
        }
    st.session_state.chat.append(reply)
    st.rerun()
