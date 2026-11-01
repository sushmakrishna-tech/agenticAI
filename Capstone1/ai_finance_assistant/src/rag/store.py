"""Builds the Chroma index and searches it."""
import logging
import threading
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.core.config import PROJECT_ROOT, get_config, get_openai_key
from src.rag.articles import load_articles
from src.rag.chunking import chunk_article

logger = logging.getLogger(__name__)
BATCH_SIZE = 200
_store = None
_store_lock = threading.Lock()


class KnowledgeBaseError(Exception):
    """Search could not run. Different from a search that found nothing."""


@dataclass(frozen=True)
class SearchResult:
    text: str
    score: float  # 0 to 1, higher is closer in meaning
    title: str
    category: str
    source_name: str
    source_url: str


def _open_store(embeddings, persist_dir: str | None, collection: str) -> Chroma:
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_index(articles, embeddings, persist_dir=None, collection="finance_kb",
                chunk_size=900, overlap=120) -> tuple[Chroma, int]:
    """Replace the index, so edited or deleted articles can't leave old chunks behind."""
    store = _open_store(embeddings, persist_dir, collection)
    store.delete_collection()
    store = _open_store(embeddings, persist_dir, collection)
    chunks = [c for a in articles for c in chunk_article(a, chunk_size, overlap)]
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        store.add_documents(batch, ids=[c.id for c in batch])
    return store, len(chunks)


def search(store: Chroma, query: str, categories: list[str] | None = None,
           k: int = 4, min_score: float = 0.3) -> list[SearchResult]:
    """Closest chunks to the query, keeping only reasonably good matches."""
    if not query.strip():
        return []
    where = {"category": {"$in": categories}} if categories else None
    try:
        # Cosine distance runs 0 (same) to 2 (opposite), so 1 - distance is similarity.
        hits = store.similarity_search_with_score(query, k=k, filter=where)
    except Exception as exc:
        raise KnowledgeBaseError(f"Knowledge base search failed: {type(exc).__name__}") from exc
    results = []
    for doc, distance in hits:
        score = 1 - distance
        if score >= min_score:
            meta = doc.metadata
            results.append(SearchResult(doc.page_content, score, meta["title"], meta["category"],
                                        meta["source_name"], meta["source_url"]))
    return results


def _embeddings() -> OpenAIEmbeddings:
    key = get_openai_key()
    if not key:
        raise KnowledgeBaseError("OPENAI_API_KEY is required for the knowledge base")
    return OpenAIEmbeddings(model=get_config()["rag"]["embedding_model"], api_key=key)


def build_default_index() -> tuple[int, int]:
    """Index the articles in config.yaml. Calls OpenAI, so it costs a small amount."""
    cfg = get_config()["rag"]
    articles = load_articles(PROJECT_ROOT / cfg["articles_dir"])
    _, chunks = build_index(articles, _embeddings(), str(PROJECT_ROOT / cfg["persist_dir"]),
                            cfg["collection"], cfg["chunk_size"], cfg["chunk_overlap"])
    return len(articles), chunks


def get_store() -> Chroma:
    """The saved index, shared by all agents.

    Agents run in parallel and Chroma's startup isn't thread-safe, so the lock
    ensures only one thread opens it. A failure isn't saved, so a later call retries.
    """
    global _store
    with _store_lock:
        if _store is None:
            cfg = get_config()["rag"]
            store = _open_store(_embeddings(), str(PROJECT_ROOT / cfg["persist_dir"]), cfg["collection"])
            if not store.get(limit=1)["ids"]:
                raise KnowledgeBaseError("The search index is empty. Run: python -m src.rag.build_index")
            _store = store
        return _store
