"""Loads knowledge base articles: Markdown files with a metadata header.

Validation is strict because citations are a promise: an article without a real
source must fail loudly when indexing, not quietly produce uncited answers.
"""
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import yaml

CATEGORIES = {"basics", "products", "markets", "risk_portfolio", "planning", "tax"}
TRUSTED_DOMAINS = {"finra.org", "investor.gov", "irs.gov"}
REQUIRED = {"title", "category", "source_name", "source_url", "retrieved"}


class ArticleError(ValueError):
    """An article file is malformed."""


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    category: str
    source_name: str
    source_url: str
    retrieved: str
    body: str


def parse_article(path: Path) -> Article:
    text = path.read_text(encoding="utf-8")
    parts = text.split("---\n", 2)
    if len(parts) != 3 or parts[0].strip():
        raise ArticleError(f"{path.name}: missing metadata header")
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as exc:
        raise ArticleError(f"{path.name}: invalid metadata") from exc
    missing = REQUIRED - set(meta)
    if missing:
        raise ArticleError(f"{path.name}: missing {sorted(missing)}")
    if meta["category"] not in CATEGORIES:
        raise ArticleError(f"{path.name}: unknown category {meta['category']!r}")
    url = urlparse(str(meta["source_url"]))
    host = (url.hostname or "").removeprefix("www.")
    if url.scheme != "https" or host not in TRUSTED_DOMAINS:
        raise ArticleError(f"{path.name}: source must be https on {sorted(TRUSTED_DOMAINS)}")
    body = parts[2].strip()
    if not body:
        raise ArticleError(f"{path.name}: empty body")
    return Article(path.stem, str(meta["title"]), meta["category"], str(meta["source_name"]),
                   str(meta["source_url"]), str(meta["retrieved"]), body)


def load_articles(directory: Path) -> list[Article]:
    articles = [parse_article(p) for p in sorted(Path(directory).rglob("*.md"))]
    ids = [a.id for a in articles]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ArticleError(f"Duplicate article ids: {sorted(duplicates)}")
    return articles
