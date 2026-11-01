"""Finance Q&A and Tax Education: answer from the knowledge base and cite it."""
import logging

from langchain_core.messages import HumanMessage

from src.agents.common import explain, latest_question, output, user_history
from src.core.config import get_config
from src.rag.store import KnowledgeBaseError, search
from src.utils.metrics import annotate, record, traced

logger = logging.getLogger(__name__)
SCORE_MARGIN = 0.1
NOT_FOUND = (
    "I couldn't find this in my verified knowledge base, so I won't guess. "
    "Try rephrasing, or check Investor.gov, FINRA.org, or IRS.gov directly."
)


def standalone_question(llm, state) -> str:
    """Rewrite follow-ups like "what about the limit?" so they can be searched."""
    question = latest_question(state)
    earlier = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if llm is None or len(earlier) < 2:
        return question
    try:
        rewritten = llm.invoke([
            ("system", "Rewrite the user's last message as one standalone search question, "
                       "using earlier messages for context. Return only the question."),
            ("human", user_history(state)),
        ]).content
        return str(rewritten).strip() or question
    except Exception:
        logger.exception("Could not rewrite the question; searching it as written")
        return question


def make_knowledge_agent(name: str, categories, task: str, llm, get_store):
    def run(state):
        cfg = get_config()["rag"]
        try:
            query = standalone_question(llm, state)
            with traced("knowledge_search", "RETRIEVER", agent=name):
                results = search(get_store(), query, categories, cfg["top_k"], cfg["min_score"])
                annotate(result_count=len(results), top_score=results[0].score if results else 0.0)
        except KnowledgeBaseError as exc:
            logger.warning("%s", exc)
            return output(name, "The knowledge base is unavailable right now.", error=True)
        if not results:
            record("retrieval_empty")
            return output(name, NOT_FOUND)
        # Keep only close contenders, so a weak fourth match isn't used or cited.
        results = [r for r in results if r.score >= results[0].score - SCORE_MARGIN]
        facts = "\n\n".join(r.text for r in results)
        sources = list(dict.fromkeys(f"{r.title} ({r.source_name}): {r.source_url}" for r in results))
        return output(name, explain(llm, task, facts, state), sources, facts=facts)

    return run
