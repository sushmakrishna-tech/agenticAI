from streamlit.testing.v1 import AppTest



def test_chat_turn_through_the_ui(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("src.core.llm.get_openai_key", lambda: None)
    at = AppTest.from_file("../src/web_app/app.py", default_timeout=30).run()
    assert not at.exception
    at.chat_input[0].set_value("What is a Roth IRA?").run()
    assert not at.exception
    assert "Tax Education" in at.caption[0].value
