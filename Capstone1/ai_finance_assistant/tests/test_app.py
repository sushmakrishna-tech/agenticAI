import sqlite3

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "app.db"
    monkeypatch.setenv("FINANCE_DB_PATH", str(path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("src.core.llm.get_openai_key", lambda: None)
    monkeypatch.setattr("src.rag.store.get_openai_key", lambda: None)  # no API calls in tests
    return path


def open_app():
    at = AppTest.from_file("../src/web_app/app.py", default_timeout=30).run()
    assert not at.exception
    return at


def submit(at, label):
    next(b for b in at.button if b.label == label).click().run()
    assert not at.exception


def sign_up(at, email="a@example.com", password="password1"):
    at.text_input(key="signup_email").set_value(email)
    at.text_input(key="signup_password").set_value(password)
    submit(at, "Create account")


def log_in(at, email="a@example.com", password="password1"):
    at.text_input(key="login_email").set_value(email)
    at.text_input(key="login_password").set_value(password)
    submit(at, "Log in")


def ask(at, question):
    at.chat_input[0].set_value(question).run()
    assert not at.exception


def page_text(at):
    return " ".join(m.value for m in at.markdown)


def test_the_app_requires_login(db):
    at = open_app()
    assert not at.chat_input
    assert not at.error


def test_chat_turn_through_the_ui(db):
    at = open_app()
    sign_up(at)
    ask(at, "What is a Roth IRA?")
    assert any("Tax Education" in c.value for c in at.caption)


def test_wrong_password_shows_one_generic_error(db):
    at = open_app()
    sign_up(at)
    submit(at, "Log out")
    log_in(at, password="wrong-password")
    assert at.error[0].value == "Incorrect email or password."
    assert not at.chat_input
    log_in(at, email="nobody@example.com")
    assert at.error[0].value == "Incorrect email or password."


def test_bad_signup_shows_the_reason(db):
    at = open_app()
    sign_up(at, password="short")
    assert "at least 8" in at.error[0].value
    assert not at.chat_input


def test_conversation_and_profile_return_after_a_new_login(db):
    at = open_app()
    sign_up(at)
    at.selectbox[1].set_value("aggressive").run()
    ask(at, "What is a Roth IRA?")

    # A new AppTest has empty session state, as a page refresh does.
    at = open_app()
    log_in(at)
    assert at.selectbox[1].value == "aggressive"
    assert "What is a Roth IRA?" not in page_text(at)  # starts on a new conversation
    submit(at, "What is a Roth IRA?")
    assert "What is a Roth IRA?" in page_text(at)
    assert any("Tax Education" in c.value for c in at.caption)


def test_agents_remember_a_saved_conversation(db):
    at = open_app()
    sign_up(at)
    ask(at, "What is a Roth IRA?")
    thread_id = at.session_state.thread_id
    ask(at, "What is an ETF?")
    with sqlite3.connect(db) as conn:
        checkpoints = conn.execute("SELECT count(*) FROM checkpoints WHERE thread_id = ?", (thread_id,)).fetchone()[0]
    assert checkpoints > 0


def test_users_only_see_their_own_conversations(db):
    at = open_app()
    sign_up(at)
    ask(at, "My private question")
    private_thread = at.session_state.thread_id
    submit(at, "Log out")
    assert "thread_id" not in at.session_state
    assert "user" not in at.session_state

    sign_up(at, email="b@example.com")
    assert all(b.label != "My private question" for b in at.button)
    assert "My private question" not in page_text(at)

    # Even with the ID in hand, the other user's turns don't load and the question is refused.
    at.session_state.thread_id = private_thread
    at.run()
    assert "My private question" not in page_text(at)
    at.chat_input[0].set_value("Let me in").run()
    assert at.exception
    assert "Let me in" not in page_text(at)


def test_deleting_a_conversation_also_clears_agent_memory(db):
    at = open_app()
    sign_up(at)
    ask(at, "What is a Roth IRA?")
    thread_id = at.session_state.thread_id
    submit(at, "Delete")
    assert all(b.label != "What is a Roth IRA?" for b in at.button)
    assert at.session_state.thread_id != thread_id
    with sqlite3.connect(db) as conn:
        for table in ("checkpoints", "writes", "threads"):
            column = "id" if table == "threads" else "thread_id"
            assert conn.execute(f"SELECT count(*) FROM {table} WHERE {column} = ?", (thread_id,)).fetchone()[0] == 0


def test_a_rating_is_saved_once_and_kept(db, monkeypatch, tmp_path):
    monkeypatch.setattr("src.utils.feedback.save_feedback.__defaults__", (tmp_path / "feedback.jsonl",))
    at = open_app()
    sign_up(at)
    ask(at, "What is a Roth IRA?")
    at.get("feedback")[0].set_value(1).run()
    assert not at.exception
    assert any("rated this answer" in c.value for c in at.caption)
    assert not at.get("feedback")
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT rating FROM turns WHERE rating IS NOT NULL").fetchall() == [(1,)]
