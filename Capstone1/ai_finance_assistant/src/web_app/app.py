"""Streamlit UI. Run from the project root: streamlit run src/web_app/app.py"""
import logging
import sqlite3
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # makes `src` importable

from src.utils.tracing import setup_tracing

setup_tracing()  # must run before LangChain and LangGraph are imported

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from openinference.instrumentation import using_attributes

from src.agents.registry import AGENTS
from src.data.accounts import AccountError, AccountStore, get_db_path
from src.utils.feedback import save_feedback
from src.web_app.dashboards import render_goals, render_market, render_portfolio
from src.workflow.graph import build_graph

logger = logging.getLogger(__name__)

KNOWLEDGE_LEVELS = ["beginner", "intermediate", "advanced"]
RISK_LEVELS = ["conservative", "moderate", "aggressive"]


@st.cache_resource  # build once per server, and share it across all sessions
def load_store(path: str) -> AccountStore:
    return AccountStore(path)


@st.cache_resource
def load_graph(path: str):
    # Streamlit reruns on different threads. SqliteSaver has its own lock, so this is safe.
    saver = SqliteSaver(sqlite3.connect(path, check_same_thread=False))
    saver.setup()
    return build_graph(checkpointer=saver)


def escape(text: str) -> str:
    """Streamlit treats text between dollar signs as math."""
    return text.replace("$", r"\$")


def render_login(store: AccountStore) -> None:
    log_in, sign_up = st.tabs(["Log in", "Sign up"])
    with log_in, st.form("login_form", clear_on_submit=True):
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.form_submit_button("Log in"):
            if user := store.authenticate(email, password):
                st.session_state.user = user
                st.rerun()
            st.error("Incorrect email or password.")
    with sign_up, st.form("signup_form", clear_on_submit=True):
        st.caption("This is a demo with no password reset. Don't reuse a real password or enter "
                   "real financial details.")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password (at least 8 characters)", type="password", key="signup_password")
        if st.form_submit_button("Create account"):
            try:
                st.session_state.user = store.create_user(email, password)
            except AccountError as error:
                st.error(str(error))
            else:
                st.rerun()


def render_sidebar(store: AccountStore, user: dict, graph) -> dict:
    """Returns the current profile."""
    with st.sidebar:
        st.caption(f"Logged in as {escape(user['email'])}")
        if st.button("Log out"):
            st.session_state.clear()  # nothing from this user may carry over to the next
            st.rerun()

        st.header("Your profile")
        saved = store.get_profile(user["id"])
        profile = {
            "knowledge_level": st.selectbox(
                "Knowledge level", KNOWLEDGE_LEVELS,
                index=KNOWLEDGE_LEVELS.index(saved.get("knowledge_level", "beginner")), key="profile_knowledge"),
            "risk_tolerance": st.selectbox(
                "Risk tolerance", RISK_LEVELS,
                index=RISK_LEVELS.index(saved.get("risk_tolerance", "moderate")), key="profile_risk"),
        }
        if profile != saved:
            store.save_profile(user["id"], profile)

        st.header("Conversations")
        if st.button("New conversation"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()
        current = st.session_state.thread_id
        for thread in store.list_threads(user["id"]):
            if st.button(escape(thread["title"]), key=f"thread_{thread['id']}",
                         type="primary" if thread["id"] == current else "secondary"):
                st.session_state.thread_id = thread["id"]
                st.rerun()
        if store.owns_thread(user["id"], current):
            with st.popover("Delete this conversation"):
                st.write("This can't be undone.")
                if st.button("Delete", key="confirm_delete"):
                    # Ownership is checked first, so only the owner can clear checkpoints.
                    if store.delete_thread(user["id"], current):
                        graph.checkpointer.delete_thread(current)
                    st.session_state.thread_id = str(uuid.uuid4())
                    st.rerun()
    return profile


def render_chat(store: AccountStore, user: dict) -> None:
    for turn in store.get_turns(user["id"], st.session_state.thread_id):
        with st.chat_message(turn["role"]):
            st.markdown(escape(turn["content"]))
            for source in turn["sources"]:
                st.caption(escape(f"Source: {source}"))
            if turn["caption"]:
                st.caption(turn["caption"])
            if turn["agents"]:
                if turn["rating"] is not None:
                    st.caption(f"You rated this answer {'👍' if turn['rating'] == 1 else '👎'}")
                    continue
                rating = st.feedback("thumbs", key=f"feedback_{turn['id']}")
                if rating is not None:
                    if store.rate_turn(user["id"], turn["id"], rating):
                        save_feedback(rating == 1, st.session_state.thread_id, turn["agents"])
                    st.rerun()


st.set_page_config(page_title="AI Finance Assistant", layout="wide")
st.title("AI Finance Assistant")

db_path = str(get_db_path())
store = load_store(db_path)

# Streamlit reruns this script on every interaction; session_state survives reruns.
# It doesn't survive a page refresh, so refreshing asks the user to log in again.
user = st.session_state.get("user")
if not user:
    render_login(store)
    st.stop()

st.session_state.setdefault("thread_id", str(uuid.uuid4()))
graph = load_graph(db_path)
profile = render_sidebar(store, user, graph)

chat_tab, portfolio_tab, market_tab, goals_tab = st.tabs(["Chat", "Portfolio", "Market", "Goals"])

with chat_tab:
    render_chat(store, user)
with portfolio_tab:
    render_portfolio(store, user)
with market_tab:
    render_market()
with goals_tab:
    render_goals(profile)

# st.chat_input stays outside the tabs so it remains pinned to the page bottom.
if question := st.chat_input("Ask about investing, your portfolio, taxes..."):
    thread_id = st.session_state.thread_id
    # Runs before the graph, so another user's thread can never be loaded or changed.
    store.ensure_thread(user["id"], thread_id, question)
    store.add_turn(user["id"], thread_id, "user", question)
    config = {"configurable": {"thread_id": thread_id}}
    try:
        # Groups a conversation's traces in Arize. The user ID is random, not the email.
        with st.spinner("Thinking..."), using_attributes(session_id=thread_id, user_id=user["id"]):
            result = graph.invoke(
                # Holdings are always sent, so removing them in the tab clears them here too.
                {"messages": [HumanMessage(content=question)], "profile": profile,
                 "holdings": store.get_holdings(user["id"])}, config
            )
    except Exception:
        logger.exception("Workflow failed")
        store.add_turn(user["id"], thread_id, "assistant", "Something went wrong. Please try again.")
    else:
        used = ", ".join(AGENTS[name].title for name in result["route"])
        store.add_turn(
            user["id"], thread_id, "assistant", result["messages"][-1].content,
            sources=result.get("sources", []),
            caption=f"Agents used: {used} (routed by {result['route_source']})",
            agents=result["route"],
        )
    st.rerun()
