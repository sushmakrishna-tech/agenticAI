import math
import uuid
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from src.rag.articles import Article, ArticleError, load_articles, parse_article
from src.rag.chunking import chunk_article
from src.rag.store import KnowledgeBaseError, build_index, search

VOCAB = ["roth", "ira", "stock", "bond", "tax", "fund", "risk", "goal"]


class WordEmbeddings(Embeddings):
    """Offline stand-in: texts sharing words get similar vectors, with no API calls."""

    def _embed(self, text):
        low = text.lower()
        vector = [float(low.count(w)) for w in VOCAB] + [0.01]
        size = math.sqrt(sum(v * v for v in vector))
        return [v / size for v in vector]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def article(id="a", title="Title", category="tax", body="Body text.", url="https://www.irs.gov/x"):
    return Article(id, title, category, "IRS", url, "2026-11-01", body)


def write(tmp_path, name="a.md", **overrides):
    meta = {"title": "T", "category": "tax", "source_name": "IRS",
            "source_url": "https://www.irs.gov/x", "retrieved": "2026-11-01"} | overrides
    header = "".join(f"{k}: {v}\n" for k, v in meta.items() if v is not None)
    path = tmp_path / name
    path.write_text(f"---\n{header}---\n{overrides.get('body', 'Body')}\n", encoding="utf-8")
    return path


def test_parse_valid_article(tmp_path):
    parsed = parse_article(write(tmp_path))
    assert parsed.id == "a" and parsed.category == "tax" and parsed.body == "Body"


@pytest.mark.parametrize("overrides", [
    {"title": None},
    {"source_url": None},
    {"category": "astrology"},
    {"source_url": "http://www.irs.gov/x"},
    {"source_url": "https://irs.gov.evil.com/x"},
    {"source_url": "https://example.com/x"},
])
def test_bad_metadata_rejected(tmp_path, overrides):
    with pytest.raises(ArticleError):
        parse_article(write(tmp_path, **overrides))


@pytest.mark.parametrize("text", ["no header", "---\n: : bad: [\n---\nBody", "---\ntitle: x\n---\nBody"])
def test_malformed_files_rejected(tmp_path, text):
    path = tmp_path / "a.md"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ArticleError):
        parse_article(path)


def test_empty_body_rejected(tmp_path):
    path = tmp_path / "a.md"
    path.write_text("---\ntitle: T\ncategory: tax\nsource_name: IRS\nsource_url: https://www.irs.gov/x\nretrieved: 2026-11-01\n---\n\n", encoding="utf-8")
    with pytest.raises(ArticleError):
        parse_article(path)


def test_duplicate_ids_rejected(tmp_path):
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    write(tmp_path / "one")
    write(tmp_path / "two")
    with pytest.raises(ArticleError):
        load_articles(tmp_path)


def test_chunks_carry_context_and_citation():
    body = "Intro text.\n\n## Limits\nThe limit is $7,000."
    chunks = chunk_article(article(title="Roth IRAs", body=body))
    assert [c.metadata["section"] for c in chunks] == ["", "Limits"]
    assert chunks[1].page_content.startswith("Roth IRAs - Limits\n\n")
    assert chunks[1].metadata["source_url"] == "https://www.irs.gov/x"
    assert [c.id for c in chunks] == ["a:0", "a:1"]


def test_long_section_splits_within_size_with_overlap():
    body = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_article(article(body=body), chunk_size=300, overlap=60)
    texts = [c.page_content.split("\n\n", 1)[1] for c in chunks]
    assert len(chunks) > 5
    assert all(len(t) <= 300 for t in texts)
    assert texts[0][-15:] in texts[1]


def store_with(*articles):
    return build_index(articles, WordEmbeddings(), collection=f"t{uuid.uuid4().hex}")


def sample():
    return (article("r", "Roth", "tax", "roth ira roth ira tax"),
            article("s", "Stocks", "products", "stock stock stock risk", "https://www.finra.org/s"))


def test_search_returns_best_match_with_citation():
    store, count = store_with(*sample())
    assert count == 2
    top = search(store, "roth ira", k=2, min_score=0.1)[0]
    assert top.title == "Roth" and top.source_url == "https://www.irs.gov/x"


def test_category_filter_excludes_other_categories():
    store, _ = store_with(*sample())
    results = search(store, "roth ira stock", ["products"], min_score=0)
    assert {r.category for r in results} == {"products"}


def test_weak_matches_are_dropped():
    store, _ = store_with(*sample())
    assert search(store, "goal", min_score=0.5) == []


def test_blank_query_returns_nothing():
    store, _ = store_with(*sample())
    assert search(store, "   ") == []


def test_rebuild_removes_deleted_articles():
    name = f"t{uuid.uuid4().hex}"
    first, second = sample()
    build_index([first, second], WordEmbeddings(), collection=name)
    store, _ = build_index([second], WordEmbeddings(), collection=name)
    assert {r.title for r in search(store, "roth ira stock", min_score=0)} == {"Stocks"}


def test_search_failure_is_a_clear_error():
    class Broken:
        def similarity_search_with_score(self, *a, **k):
            raise RuntimeError("down")

    with pytest.raises(KnowledgeBaseError):
        search(Broken(), "roth")


def test_parallel_agents_share_one_store_opened_once(monkeypatch):
    import threading
    import time

    from src.rag import store as store_module

    opened = []

    class Store:
        def get(self, limit):
            return {"ids": ["x"]}

    def slow_open(*args):
        opened.append(1)
        time.sleep(0.05)  # widens the window a race would need
        return Store()

    monkeypatch.setattr(store_module, "_store", None)
    monkeypatch.setattr(store_module, "_open_store", slow_open)
    monkeypatch.setattr(store_module, "_embeddings", lambda: None)
    results = []
    threads = [threading.Thread(target=lambda: results.append(store_module.get_store())) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(opened) == 1
    assert len(results) == 6 and len(set(map(id, results))) == 1


def test_empty_index_is_a_clear_error_and_retries(monkeypatch):
    from src.rag import store as store_module

    class Empty:
        def get(self, limit):
            return {"ids": []}

    monkeypatch.setattr(store_module, "_store", None)
    monkeypatch.setattr(store_module, "_open_store", lambda *a: Empty())
    monkeypatch.setattr(store_module, "_embeddings", lambda: None)
    with pytest.raises(KnowledgeBaseError):
        store_module.get_store()
    assert store_module._store is None
