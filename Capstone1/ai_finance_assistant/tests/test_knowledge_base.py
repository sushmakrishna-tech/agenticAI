"""Checks the real articles, with no API calls."""
from pathlib import Path

import pytest
import yaml

from src.core.config import PROJECT_ROOT, get_config
from src.rag.articles import CATEGORIES, load_articles
from src.rag.chunking import chunk_article

ARTICLES = load_articles(PROJECT_ROOT / get_config()["rag"]["articles_dir"])


def test_meets_the_brief_size():
    assert 50 <= len(ARTICLES) <= 100


def test_every_category_has_articles():
    assert {a.category for a in ARTICLES} == CATEGORIES


def test_every_article_chunks_within_the_size_limit():
    cfg = get_config()["rag"]
    for article in ARTICLES:
        chunks = chunk_article(article, cfg["chunk_size"], cfg["chunk_overlap"])
        assert chunks
        assert all(len(c.page_content.split("\n\n", 1)[1]) <= cfg["chunk_size"] for c in chunks)


def test_eval_questions_point_at_real_articles():
    data = yaml.safe_load((PROJECT_ROOT / "tests/data/retrieval_eval.yaml").read_text(encoding="utf-8"))
    ids = {a.id for a in ARTICLES}
    assert {e for q in data["questions"] for e in q["expect"]} <= ids
