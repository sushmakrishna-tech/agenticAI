# Technical Design: AI Finance Assistant

Living document. It describes both what exists and what is planned, and the status below shows which is which.

## Scope
Education for US beginners, not personalized advice. Tax content covers US rules only.

## Build status
| Part | Status |
|---|---|
| LangGraph workflow, router, memory, compliance | Built |
| Market data layer | Built; Alpha Vantage tested only against simulated responses |
| Streamlit app | Chat works; dashboard tabs are placeholders |
| Six agents | Stubs that return placeholder text |
| Portfolio metrics, goal projections | Built and tested, not yet connected to agents or the UI |
| Knowledge base, MCP server | Planned |

## System overview
```mermaid
flowchart TB
    user(["User"]) --> ui["Streamlit app<br/>Chat, Portfolio, Market, Goals"]
    ui <--> workflow

    subgraph workflow["LangGraph workflow"]
        direction LR
        router["Router"] --> agents["Six specialist agents"]
        agents --> merge["Merge"]
        merge --> compliance["Compliance"]
    end

    workflow --> openai[("OpenAI API")]
    workflow -.-> tools
    mcp["MCP server"] -.-> tools

    subgraph tools["Tools the agents will use"]
        direction LR
        market["Market data service"]
        calculators["Portfolio and goal calculators"]
        rag["Knowledge base<br/>Chroma"]
    end

    tools --> providers[("yFinance<br/>Alpha Vantage")]

    classDef built fill:#d1fae5,stroke:#047857,color:#064e3b
    classDef stub fill:#fef3c7,stroke:#b45309,color:#78350f
    classDef planned fill:#e5e7eb,stroke:#6b7280,color:#1f2937,stroke-dasharray:5 5
    classDef external fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a
    class ui,router,merge,compliance,market built
    class agents stub
    class calculators,rag,mcp planned
    class openai,providers external
    style workflow fill:#f8fafc,stroke:#94a3b8,color:#0f172a
    style tools fill:#f8fafc,stroke:#94a3b8,color:#0f172a
```
**Key:** green is built, yellow is a stub, gray is planned, and blue is an external service. Dashed arrows are connections that don't exist yet. Only the router calls OpenAI today, and only the market data service uses yFinance and Alpha Vantage.

## Stack
| Concern | Choice | Why |
|---|---|---|
| Orchestration and state | LangGraph | Named in the rubric; explicit graph, parallel branches, built-in memory |
| LLM | OpenAI, set in `config.yaml` | Key available; created in one file (`src/core/llm.py`) |
| Vector store | Chroma (planned) | Runs locally with metadata filtering for categories |
| Market data | yFinance, Alpha Vantage fallback | Free primary source plus a backup |
| UI | Streamlit | Named in the rubric; quick multi-tab app |

## Workflow
This graph matches `src/workflow/graph.py`. The router can select any combination of agents, and those agents run in parallel.

```mermaid
flowchart LR
    start(["START"]) --> router{"Router"}
    router --> qa["Finance Q&A"]
    router --> portfolio["Portfolio Analysis"]
    router --> market["Market Analysis"]
    router --> goals["Goal Planning"]
    router --> news["News Synthesizer"]
    router --> tax["Tax Education"]
    qa --> merge["Merge"]
    portfolio --> merge
    market --> merge
    goals --> merge
    news --> merge
    tax --> merge
    merge --> compliance["Compliance"]
    compliance --> finish(["END"])

    classDef built fill:#d1fae5,stroke:#047857,color:#064e3b
    classDef stub fill:#fef3c7,stroke:#b45309,color:#78350f
    class router,merge,compliance built
    class qa,portfolio,market,goals,news,tax stub
```
- **Router:** the LLM picks one or more agents using structured output. Keyword rules take over if there is no key or the call fails.
- **Agents:** each takes state and returns one `AgentOutput`.
- **Merge:** combines outputs in a fixed order and de-duplicates sources.
- **Compliance:** adds a disclaimer and flags advice-like wording.

## One question, start to finish
This example follows a question that spans two agents.

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit
    participant Router
    participant Goals as Goal Planning
    participant Tax as Tax Education
    participant Merge
    participant Compliance

    User->>UI: Will selling stock for my down payment raise my taxes?
    UI->>Router: New message and thread_id
    Note over Router: Clears last turn's outputs<br/>Chooses goals and tax
    par Agents run in parallel
        Router->>Goals: Shared state
        Goals-->>Merge: AgentOutput
    and
        Router->>Tax: Shared state
        Tax-->>Merge: AgentOutput
    end
    Merge->>Compliance: Combined draft and sources
    Note over Compliance: Flags advice wording<br/>Adds disclaimer
    Compliance-->>UI: Final answer, saved to conversation memory
    UI-->>User: Answer and the agents used
```

## Agent communication
Agents do not call each other. They communicate only through shared state (`src/core/state.py`), which keeps them independently testable. Results are combined by the `collect_outputs` reducer; the router clears it each turn so old answers do not leak into the next one.

## Key decisions
1. **Code calculates, the LLM explains.** Financial figures will come from tested Python, so the model cannot invent them. This is a design rule until the real agents are built.
2. **Failures stay contained.** A failing agent yields an error output, and other agents still answer.
3. **Conversation memory is per thread.** `InMemorySaver` keeps it for the life of the server process, which is enough for a demo but is lost on restart.
4. **Compliance is enforced by the graph,** so no agent can skip it.

## Market data
```mermaid
flowchart LR
    request["Quote or history request"] --> valid{"Valid symbol?"}
    valid -- No --> invalid["InvalidSymbolError"]
    valid -- Yes --> cache{"Fresh in cache?"}
    cache -- Yes --> fresh["Return cached data"]
    cache -- No --> yf{"yFinance answers?"}
    yf -- Yes --> store["Save to cache and return"]
    yf -- No --> av{"Alpha Vantage answers?"}
    av -- Yes --> store
    av -- No --> old{"Older copy in cache?"}
    old -- Yes --> stale["Return it with stale=True"]
    old -- No --> error["MarketDataError"]

    classDef good fill:#d1fae5,stroke:#047857,color:#064e3b
    classDef warn fill:#fef3c7,stroke:#b45309,color:#78350f
    classDef bad fill:#fee2e2,stroke:#b91c1c,color:#7f1d1d
    class fresh,store good
    class stale warn
    class invalid,error bad
```
Quotes stay fresh for 60 seconds and history for an hour. A rate-limited provider is skipped for a cooldown period. The free Alpha Vantage series covers about 100 trading days, so long-range requests served by the fallback come back shorter.

## Portfolio metrics
`src/utils/portfolio_metrics.py` calculates value, weights, gains, effective number of holdings, volatility, max drawdown, Sharpe ratio, and beta against the S&P 500. Risk metrics apply today's weights to past daily returns, so they describe the current mix rather than the user's actual results. Returns are not annualized below 200 trading days. The metrics can't see inside funds, so a broad ETF counts as one holding and observations word concentration accordingly. Holdings' returns include dividends while the S&P 500 index does not, which slightly flatters comparisons against it. Observations state facts and never say what to do. Missing prices or history are reported, not hidden.

## Goal projections
`src/utils/goal_projections.py` gives a straight-line estimate and a seeded Monte Carlo simulation of monthly lognormal returns. Targets grow with inflation, and contributions stay flat. Return and volatility assumptions are illustrative values in `config.yaml`, and taxes and fees are ignored.

## RAG (planned)
50-100 summaries of public sources (Investor.gov, FINRA, IRS), each tagged with category and source URL, retrieved with category filters and cited in answers.

```mermaid
flowchart LR
    sources["Public sources<br/>Investor.gov, FINRA, IRS"] --> summaries["Summaries with<br/>category and URL"]
    summaries --> chunks["Chunk and embed"]
    chunks --> chroma[("Chroma")]
    question["Agent's question"] --> search["Search, filtered<br/>by category"]
    chroma --> search
    search --> answer["Answer with citations"]

    classDef planned fill:#e5e7eb,stroke:#6b7280,color:#1f2937,stroke-dasharray:5 5
    class sources,summaries,chunks,chroma,question,search,answer planned
```

## Performance (planned)
Benchmarks for router accuracy, retrieval hit rate, and latency, from a labeled question set.
