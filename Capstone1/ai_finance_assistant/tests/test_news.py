import pytest

from src.data.news import NewsError, NewsService, _parse, is_relevant, parse_feed

ITEM = {"title": "Apple rises", "publisher": "IBD", "link": "https://x.test/a", "providerPublishTime": 1793538054}


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_parse_good_item():
    article = _parse(ITEM)
    assert article.title == "Apple rises" and article.published.year == 2026


def test_parse_rejects_unsafe_or_incomplete_items():
    assert _parse({**ITEM, "link": "javascript:alert(1)"}) is None
    assert _parse({**ITEM, "title": None}) is None
    assert _parse({"title": "x", "link": "https://x.test", "providerPublishTime": "bad"}).published is None


def test_cache_then_refresh():
    calls = []
    clock = Clock()
    service = NewsService(lambda q, n: calls.append(q) or [ITEM], ttl=10, clock=clock)
    service.search("Apple")
    service.search(" apple ")
    assert len(calls) == 1
    clock.t = 11
    service.search("apple")
    assert len(calls) == 2


def test_failure_uses_old_articles_then_errors():
    state = {"fail": False}

    def fetch(q, n):
        if state["fail"]:
            raise RuntimeError("down")
        return [ITEM]

    clock = Clock()
    service = NewsService(fetch, ttl=10, clock=clock)
    service.search("apple")
    state["fail"], clock.t = True, 11
    assert service.search("apple")[0].title == "Apple rises"
    try:
        service.search("new query")
        raise AssertionError("expected NewsError")
    except NewsError:
        pass


def test_article_limit():
    service = NewsService(lambda q, n: [ITEM] * 9, max_articles=3)
    assert len(service.search("apple")) == 3


FEED = """<rss><channel><item><title>Nvidia climbs</title><link>https://www.fool.com/a</link>
<pubDate>Sun, 01 Nov 2026 17:13:58 +0000</pubDate></item>
<item><title>No date</title><link>https://x.test/b</link></item></channel></rss>"""


def test_parse_feed():
    items = parse_feed(FEED)
    assert items[0]["title"] == "Nvidia climbs" and items[0]["publisher"] == "fool.com"
    assert items[0]["providerPublishTime"] > 0
    assert items[1]["providerPublishTime"] is None


def test_feed_with_doctype_is_rejected():
    with pytest.raises(ValueError):
        parse_feed('<!DOCTYPE x [<!ENTITY a "b">]><rss></rss>')


@pytest.mark.parametrize("title,query,symbol,expected", [
    ("Nvidia climbs", "Nvidia", "NVDA", True),
    ("Why NVDA fell", "Nvidia", "NVDA", True),
    ("Micron could grow", "Nvidia", "NVDA", False),
    ("Anything at all", "stock market", None, True),
    ("Fed holds interest rates", "interest rates", None, True),
    ("Nvidiatech is a different name", "Nvidia", None, False),
])
def test_is_relevant(title, query, symbol, expected):
    assert is_relevant(title, query, symbol) is expected


def test_symbol_uses_the_ticker_feed_and_filters_unrelated_stories():
    items = [{**ITEM, "title": "Micron grows"}, {**ITEM, "title": "Nvidia climbs"}]
    service = NewsService(lambda q, n: [], lambda s: items)
    assert [a.title for a in service.search("Nvidia", "NVDA")] == ["Nvidia climbs"]


def test_no_relevant_stories_returns_nothing():
    service = NewsService(lambda q, n: [{**ITEM, "title": "Micron grows"}])
    assert service.search("Nvidia") == []
