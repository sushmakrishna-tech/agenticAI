# AI Finance Assistant

A multi-agent assistant for US financial education, built with LangGraph and Streamlit.

**Status:** skeleton. Routing, memory, compliance, and the UI work; the six agents are stubs.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your OPENAI_API_KEY
```

## Run
```bash
streamlit run src/web_app/app.py
pytest --cov=src
```
Without an API key, the app still runs and routes using keyword rules.

## Docs
See [docs/DESIGN.md](docs/DESIGN.md) for the architecture.
