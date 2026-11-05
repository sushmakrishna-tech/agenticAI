# AI Finance Assistant

A multi-agent assistant for US financial education, built with LangGraph and Streamlit.

**Status:** in progress. The six agents answer through chat using live data and the knowledge base, users can log in and return to saved conversations, and the Portfolio, Market, and Goals tabs show dashboards. The MCP server is planned.

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

Accounts and conversations are saved in `finance_assistant.db`, which Git ignores. This is a demo with no password reset, so don't reuse a real password or enter real financial details.

## Tracing (optional)
Set `ARIZE_TRACING=true`, `ARIZE_SPACE_ID`, and `ARIZE_API_KEY` in `.env`. Prompts and user input are sent to Arize, so use made-up data.

## Docs
See [docs/DESIGN.md](docs/DESIGN.md) for the architecture. See [docs/GUARDRAILS.md](docs/GUARDRAILS.md) for safety checks, [docs/METRICS.md](docs/METRICS.md) for what is measured and [docs/EVAL_RESULTS.md](docs/EVAL_RESULTS.md) for the latest eval results.
