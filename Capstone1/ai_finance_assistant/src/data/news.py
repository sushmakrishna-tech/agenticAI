"""Recent financial headlines from Yahoo: its ticker feed, or yFinance's search.

There is no separate backup provider because Alpha Vantage's news feed is
premium-only. On failure, older cached articles are used, and otherwise NewsError.

Yahoo's results often include unrelated stories, so a search for a specific
company keeps only headlines that mention it.
"""
import logging
import re
import threading
import time
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from urllib.parse import urlparse

import requests
import yfinance as yf

from src.core.config import get_config

logger = logging.getLogger(__name__)
GENERAL_QUERIES = {"stock market", "market", "markets", "financial news"}
FEED_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"


class NewsError(Exception):
    """No news could be fetched, and nothing was cached."""


@dataclass(frozen=True)
class Article:
    title: str
    publisher: str
    url: str
    published: datetime | None


def _parse(item: dict) -> Article | None:
    title, url = item.get("title"), item.get("link")
    # Links end up clickable in the UI, so accept web addresses only.
    if not title or urlparse(str(url)).scheme != "https":
        return None
    stamp = item.get("providerPublishTime")
    published = datetime.fromtimestamp(stamp, timezone.utc) if isinstance(stamp, (int, float)) else None
    return Article(str(title), str(item.get("publisher") or "Unknown"), str(url), published)


def _yahoo_search(query: str, count: int) -> list[dict]:
    return yf.Search(query, news_count=count).news


def parse_feed(xml_text: str) -> list[dict]:
    if "<!DOCTYPE" in xml_text[:1000]:  # feeds never need one, and it enables XML attacks
        raise ValueError("Unexpected document type in feed")
    items = []
    for item in ElementTree.fromstring(xml_text).iter("item"):
        link = item.findtext("link") or ""
        try:
            stamp = parsedate_to_datetime(item.findtext("pubDate") or "").timestamp()
        except (TypeError, ValueError):
            stamp = None
        items.append({"title": item.findtext("title"), "link": link, "providerPublishTime": stamp,
                      "publisher": (urlparse(link).hostname or "Unknown").removeprefix("www.")})
    return items


def _yahoo_feed(symbol: str) -> list[dict]:
    response = requests.get(FEED_URL, params={"s": symbol, "region": "US", "lang": "en-US"},
                            headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    response.raise_for_status()
    return parse_feed(response.text)


def is_relevant(title: str, query: str, symbol: str | None) -> bool:
    if not symbol and query.lower().strip() in GENERAL_QUERIES:
        return True
    terms = [w for w in re.findall(r"[a-z0-9&]+", query.lower()) if len(w) > 2]
    if symbol:
        terms.append(symbol.lower())
    words = set(re.findall(r"[a-z0-9&]+", title.lower()))
    return any(term in words for term in terms)


class NewsService:
    def __init__(self, fetch=_yahoo_search, fetch_symbol=_yahoo_feed, ttl: float = 900,
                 max_articles: int = 6, clock: Callable[[], float] = time.monotonic):
        self.fetch, self.fetch_symbol = fetch, fetch_symbol
        self.ttl, self.max_articles, self.clock = ttl, max_articles, clock
        self._cache: dict[str, tuple[float, list[Article]]] = {}
        self._lock = threading.Lock()

    def search(self, query: str, symbol: str | None = None) -> list[Article]:
        key = (" ".join(query.lower().split()), symbol)
        with self._lock:
            cached = self._cache.get(key)
        if cached and self.clock() - cached[0] < self.ttl:
            return cached[1]
        try:
            items = self.fetch_symbol(symbol) if symbol else self.fetch(query, self.max_articles * 3)
            parsed = [a for a in map(_parse, items) if a]
            articles = [a for a in parsed if is_relevant(a.title, query, symbol)][: self.max_articles]
        except Exception as exc:
            logger.warning("News search failed: %s", type(exc).__name__)
            if cached:
                return cached[1]
            raise NewsError("News is unavailable right now") from exc
        with self._lock:
            self._cache[key] = (self.clock(), articles)
        return articles


@lru_cache
def get_news_service() -> NewsService:
    cfg = get_config()["news"]
    return NewsService(ttl=cfg["cache_ttl_seconds"], max_articles=cfg["max_articles"])
