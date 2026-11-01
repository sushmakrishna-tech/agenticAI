# AI Finance Assistant

A multi-agent assistant for US financial education, built with LangGraph and Streamlit.

**Status:** skeleton. The six agents answer through chat using live data and the knowledge base. Dashboards, accounts, end-to-end evals, and the MCP server are planned.

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
Without an API key, the app still runs, routes using keyword rules, and shows calculated facts without explanations. Knowledge base questions need a key.

## Tracing (optional)
Set `ARIZE_TRACING=true`, `ARIZE_SPACE_ID`, and `ARIZE_API_KEY` in `.env`. Prompts and user input are sent to Arize, so use made-up data.

## Docs
See [docs/DESIGN.md](docs/DESIGN.md) for the architecture. See [docs/METRICS.md](docs/METRICS.md) for what is measured and [docs/EVAL_RESULTS.md](docs/EVAL_RESULTS.md) for the latest eval results.
