# Technical Design: AI Finance Assistant

Living document. Sections marked *(planned)* are not built yet.

## Scope
Education for US beginners, not personalized advice. Tax content covers US rules only.

## Stack
| Concern | Choice | Why |
|---|---|---|
| Orchestration and state | LangGraph | Named in the rubric; explicit graph, parallel branches, built-in memory |
| LLM | OpenAI, set in `config.yaml` | Key available; created in one file (`src/core/llm.py`) |
| Vector store | Chroma *(planned)* | Runs locally with metadata filtering for categories |
| Market data | yFinance, Alpha Vantage fallback *(planned)* | Free primary source plus a backup |
| UI | Streamlit | Named in the rubric; quick multi-tab app |

## Workflow
```
START -> router -> agent(s) in parallel -> merge -> compliance -> END
```
- **router:** the LLM picks one or more agents using structured output. Keyword rules take over if there is no key or the call fails.
- **agents:** each takes state and returns one `AgentOutput`.
- **merge:** combines outputs in a fixed order and de-duplicates sources.
- **compliance:** adds a disclaimer and flags advice-like wording.

## Agent communication
Agents do not call each other. They communicate only through shared state (`src/core/state.py`), which keeps them independently testable. Results are combined by the `collect_outputs` reducer; the router clears it each turn so old answers do not leak into the next one.

## Key decisions
1. **Code calculates, the LLM explains.** Financial figures come from tested Python, so the model cannot invent them.
2. **Failures stay contained.** A failing agent yields an error output, and other agents still answer.
3. **Conversation memory is per thread.** `InMemorySaver` keeps it for the life of the server process, which is enough for a demo but is lost on restart.
4. **Compliance is enforced by the graph,** so no agent can skip it.

## RAG *(planned)*
50-100 summaries of public sources (Investor.gov, FINRA, IRS), each tagged with category and source URL, retrieved with category filters and cited in answers.

## Performance *(planned)*
Benchmarks for router accuracy, retrieval hit rate, and latency, from a labeled question set.
