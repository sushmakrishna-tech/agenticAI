# AI Finance Assistant

A multi-agent assistant for US financial education, built with LangGraph and Streamlit.

**Status:** in progress. The six agents answer through chat using live data and the knowledge base, users can log in and return to saved conversations, the Portfolio, Market, and Goals tabs show dashboards, and an MCP server lets AI clients use the tools. Docker files are written but not yet built.

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

## MCP server
Lets an MCP client such as Claude Desktop call the assistant's tools: quotes, price history, portfolio analysis, goal projections, knowledge base and news search, and the full assistant. See [docs/DESIGN.md](docs/DESIGN.md#mcp-server) for the tool list.

```bash
python -m src.mcp_server.server   # stdio; normally started by the client
```

To add it to Claude Desktop, add this to `claude_desktop_config.json`, replacing the path:
```json
{
  "mcpServers": {
    "ai-finance-assistant": {
      "command": "/path/to/ai_finance_assistant/.venv/bin/python",
      "args": ["-m", "src.mcp_server.server"],
      "env": {"PYTHONPATH": "/path/to/ai_finance_assistant"}
    }
  }
}
```

## Docker
```bash
docker compose up --build   # app at http://localhost:8501
docker compose run --rm -i app python -m src.mcp_server.server   # MCP server over stdio
```
Keys are read from `.env`. The first start builds the search index, which calls OpenAI. Data is kept in the `finance-data` volume. The Docker files haven't been built yet, because Docker wasn't available during development.

## Tracing (optional)
Set `ARIZE_TRACING=true`, `ARIZE_SPACE_ID`, and `ARIZE_API_KEY` in `.env`. Prompts and user input are sent to Arize, so use made-up data.

## Docs
See [docs/DESIGN.md](docs/DESIGN.md) for the architecture. See [docs/GUARDRAILS.md](docs/GUARDRAILS.md) for safety checks, [docs/METRICS.md](docs/METRICS.md) for what is measured and [docs/EVAL_RESULTS.md](docs/EVAL_RESULTS.md) for the latest eval results.
