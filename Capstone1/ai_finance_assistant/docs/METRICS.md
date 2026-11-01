# LLM Metrics

This guide explains the metrics commonly tracked for LLM applications, then lists what this project tracks and where to find it. For the latest scores, see [EVAL_RESULTS.md](EVAL_RESULTS.md).

## Common LLM metrics

### 1. Performance
| Metric | What it tells you |
|---|---|
| Total latency | How long a full answer takes; track p50 and p95, since averages hide slow requests |
| Time to first token | How soon text starts appearing, if you stream answers |
| Tokens per second | How fast the model generates text |
| Throughput | Requests handled per minute |

### 2. Usage and cost
| Metric | What it tells you |
|---|---|
| Input and output tokens | How much each call uses, since the two are usually priced differently |
| Cost per call or per question | Where the money goes |
| Cached tokens | How much prompt caching is saving, if you use it |

### 3. Reliability
| Metric | What it tells you |
|---|---|
| Error rate | Failed calls, by type such as timeout or rate limit |
| Retries and fallbacks | How often the backup path runs |
| Finish reason | Whether an answer ended normally or was cut off by a token limit |
| Structured output failures | How often the model's reply didn't fit the required format |

### 4. Answer quality
These usually come from evals, not from the API call itself.

| Metric | What it tells you |
|---|---|
| Faithfulness | Whether the answer is supported by its sources |
| Answer relevance | Whether it actually answers the question |
| Retrieval quality | Whether search found the right chunks |
| Safety | Whether it stays within your rules, such as refusing to give personal advice |
| User feedback | Thumbs up or down, and follow-up complaints |

### 5. Agent workflow
Multi-agent apps should track these too.

| Metric | What it tells you |
|---|---|
| Routing accuracy | Whether the right agents were picked |
| LLM calls per question | Whether a workflow is making more calls than it should |
| Latency by step | Which agent or step is the slowdown |

## Coverage in this project

| Metric | Status | Where |
|---|---|---|
| Total latency, p50 and p95 | Tracked | Agent evals; per-trace timing in Arize |
| Latency by step | Tracked | Arize spans |
| Time to first token | Not tracked | The app doesn't stream answers |
| Tokens per second | Not tracked | |
| Throughput | Not tracked | Not meaningful for a single-user local demo |
| Input and output tokens | Tracked | Arize LLM spans; `UsageTracker` during evals |
| Cost per question | Tracked | Agent evals, for chat calls only; excludes embeddings and judge calls |
| Cached tokens | Not tracked | |
| LLM error count | Tracked | `UsageTracker` during evals; not broken down by type |
| Fallbacks | Tracked | Event counts below |
| Retries | Not tracked | The app doesn't add its own retries |
| Finish reason | Tracked | `UsageTracker` during evals, reported as answers cut off by the token limit |
| Structured output failures | Tracked | `extraction_failed` and `routing_fallback`; these also count other failures of those calls |
| Faithfulness | Tracked | Agent evals, using an LLM judge |
| Answer relevance | Tracked | Agent evals, using an LLM judge |
| Retrieval quality | Tracked | Retrieval eval; result count and top score on spans |
| Safety | Tracked | Agent evals with an LLM judge; compliance flags on spans |
| User feedback | Tracked | Thumbs in the chat; `feedback.jsonl` and `user_feedback` spans |
| Routing accuracy | Tracked | Agent evals |
| LLM calls per question | Tracked | `UsageTracker` during evals |

## What this project tracks

### Evals
Run these by hand. Both call OpenAI, so each run costs a small amount.

**Agent evals:** `python -m src.workflow.evaluate` writes `docs/EVAL_RESULTS.md`. Questions are in `tests/data/agent_eval.yaml`.

| Metric | How it is measured |
|---|---|
| Routing, exact match | Chosen agents exactly match the expected agents |
| Routing, includes expected | Chosen agents include all expected agents, and extras are allowed |
| Faithfulness | LLM judge compares each agent's answer with its source facts |
| Answer relevance | LLM judge checks whether the answer responds to the question |
| Advice safety | LLM judge reviews answers to questions pressing for advice or predictions |
| Prediction refusal | The answer to a prediction question contains "can't predict", checked by exact text |
| Out-of-scope declined | The answer contains the knowledge base's decline message |
| Latency, p50 and p95 | Wall-clock time per question |
| LLM calls per question | Count from `UsageTracker` |
| Chat cost per question | Tokens multiplied by the prices in `config.yaml` |
| LLM errors | Count from `UsageTracker` |
| Answers cut off | Finish reason of `length` |
| Agent errors | Agent outputs marked as errors |

**Retrieval eval:** `python -m src.rag.evaluate` prints its results. Questions are in `tests/data/retrieval_eval.yaml`.

| Metric | How it is measured |
|---|---|
| Hit at 1 | The expected article ranks first |
| Hit at k | The expected article appears in the top 4 |
| Lowest correct score | Weakest match for a correct article, compared with the 0.3 cutoff |
| Correct below cutoff | Correct articles that the cutoff would drop |
| Highest off-topic score | Strongest match for an unrelated question |
| Off-topic above cutoff | Unrelated questions that pass the cutoff |
| Median latency | Search time, including the embedding call |

### Event counts
`src/utils/metrics.py` counts these and marks them on the active span. Counts are held in memory, so they reset when the process restarts. Agent evals report the counts from their run.

| Event | Meaning |
|---|---|
| `answer_rejected` | An answer had a dollar amount, percentage, or year missing from its source data, so the verified facts were shown |
| `routing_fallback` | The LLM router failed, so keyword rules took over |
| `extraction_failed` | Structured details couldn't be read from the conversation |
| `explanation_failed` | The explanation call failed, so plain facts were shown |
| `retrieval_empty` | Search found nothing above the cutoff |
| `market_provider_fallback` | A provider after the first one answered |
| `market_stale_served` | All providers failed, so an older cached value was used |
| `market_data_unavailable` | No provider or cached value could answer |
| `agent_exception` | An agent raised an unexpected error |
| `compliance_flagged` | An answer matched an advice-wording pattern |
| `feedback_positive` | A user clicked thumbs up |
| `feedback_negative` | A user clicked thumbs down |

### Arize traces
Tracing is off unless `ARIZE_TRACING=true`. Each question becomes one trace, and traces are grouped into sessions by conversation.

**Captured automatically:** the LangGraph structure, step timing, prompts and responses, model name, and token counts.

**Added by this project,** with custom fields prefixed `app.`:

| Span | Span kind | Fields |
|---|---|---|
| `router` | Automatic | `route`, `route_source` |
| Each agent | Automatic | `agent`, `agent_error`, `cited_sources` |
| `compliance` | Automatic | `compliance_flags` |
| `knowledge_search` | RETRIEVER | `agent`, `result_count`, `top_score` |
| `market_data` | TOOL | `request`, `symbol`, `cache_hit`, `provider` |
| `news_search` | TOOL | `has_symbol`, `result_count` |
| `portfolio_metrics` | TOOL | `holdings`, `unavailable` |
| `goal_simulation` | TOOL | `risk_profile`, `years` |
| `user_feedback` | EVALUATOR | `positive`, `agents` |

Events from the table above also appear as `app.<event>` fields on the active span. For example, an `answer_rejected` event adds `app.answer_rejected` and `app.invented_figures`.

### User feedback
Thumbs ratings are saved to `feedback.jsonl`, which Git ignores. Each entry holds the time, rating, conversation ID, and agents used, with no conversation text.

## Reading the results
- **LLM judges can be wrong.** Review failures by hand before trusting or acting on a score.
- **The questions aren't independent.** The project author wrote them, so scores are likely optimistic.
- **Live data changes.** Market data and news differ between runs, so small score changes may not mean anything.
- **Traces can contain personal data.** Prompts and responses go to Arize when tracing is on, so use made-up data.

## Known gaps
- Feedback spans aren't linked to the trace of the answer being rated.
- LLM errors aren't broken down by type.
- Cost excludes embeddings and judge calls.
- Event counts don't persist across restarts.
- Tracking `answer_rejected` shows how often an invented figure is caught, but the check doesn't review other wording.
