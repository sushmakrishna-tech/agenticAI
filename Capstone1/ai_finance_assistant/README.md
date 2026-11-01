# AI Finance Assistant

A multi-agent assistant for US financial education, built with LangGraph and Streamlit.

**Status:** skeleton. Routing, memory, compliance, market data, portfolio and goal calculations, and knowledge base search work; the six agents are still stubs.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your OPENAI_API_KEY
python -m src.rag.build_index   # builds the search index; calls OpenAI
```

## Run
```bash
streamlit run src/web_app/app.py
pytest --cov=src
```
Without an API key, the app still runs and routes using keyword rules.

## Docs
See [docs/DESIGN.md](docs/DESIGN.md) for the architecture.
