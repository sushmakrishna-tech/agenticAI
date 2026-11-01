from datetime import datetime, timezone

import pandas as pd
import pytest
import requests

from src.data import market_data as md
from src.data.market_data import (
    AlphaVantageProvider, InvalidSymbolError, MarketDataError, MarketDataService,
    PriceHistory, ProviderError, Quote, RateLimitError, YFinanceProvider,
)

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)


class FakeProvider:
    def __init__(self, name="fake", error=None):
        self.name, self.error, self.calls = name, error, 0

    def get_quote(self, symbol):
        self.calls += 1
        if self.error:
            raise self.error
        return Quote(symbol, 100.0, 99.0, 1.01, NOW, self.name)

    def get_history(self, symbol, period):
        self.calls += 1
        if self.error:
            raise self.error
        return PriceHistory(symbol, pd.Series([1.0, 2.0]), self.name)


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def service(*providers):
    clock = Clock()
    return MarketDataService(list(providers), quote_ttl=60, history_ttl=3600, cooldown=30, clock=clock), clock


def test_cache_avoids_second_call_until_ttl_expires():
    p = FakeProvider()
    svc, clock = service(p)
    svc.get_quote("aapl")
    svc.get_quote("AAPL")
    assert p.calls == 1
    clock.t = 61
    svc.get_quote("AAPL")
    assert p.calls == 2


def test_falls_back_to_second_provider():
    svc, _ = service(FakeProvider("a", ProviderError("down")), FakeProvider("b"))
    assert svc.get_quote("AAPL").source == "b"


def test_stale_data_is_served_and_labeled_when_all_fail():
    p = FakeProvider()
    svc, clock = service(p)
    assert svc.get_quote("AAPL").stale is False
    p.error, clock.t = ProviderError("down"), 61
    assert svc.get_quote("AAPL").stale is True


def test_recovers_after_outage():
    p = FakeProvider()
    svc, clock = service(p)
    svc.get_quote("AAPL")
    p.error, clock.t = ProviderError("down"), 61
    svc.get_quote("AAPL")
    p.error = None
    assert svc.get_quote("AAPL").stale is False


def test_all_fail_and_no_cache_raises():
    svc, _ = service(FakeProvider(error=ProviderError("down")))
    with pytest.raises(MarketDataError):
        svc.get_quote("AAPL")


def test_rate_limited_provider_is_skipped_until_cooldown_ends():
    limited = FakeProvider("a", RateLimitError("slow down"))
    backup = FakeProvider("b")
    svc, clock = service(limited, backup)
    svc.get_quote("AAPL")
    svc.get_quote("MSFT")
    assert limited.calls == 1
    clock.t = 31
    svc.get_quote("TSLA")
    assert limited.calls == 2


def test_history_cached_per_period():
    p = FakeProvider()
    svc, _ = service(p)
    svc.get_history("AAPL", "1mo")
    svc.get_history("AAPL", "1mo")
    svc.get_history("AAPL", "1y")
    assert p.calls == 2


def test_bad_period_rejected():
    svc, _ = service(FakeProvider())
    with pytest.raises(ValueError):
        svc.get_history("AAPL", "5000y")


@pytest.mark.parametrize("good,expected", [(" aapl ", "AAPL"), ("BRK-B", "BRK-B"), ("^GSPC", "^GSPC")])
def test_symbols_normalized(good, expected):
    assert md.normalize_symbol(good) == expected


@pytest.mark.parametrize("bad", ["", "not a ticker", "A" * 16, "AAPL;DROP", "../etc"])
def test_bad_symbols_never_reach_a_provider(bad):
    p = FakeProvider()
    svc, _ = service(p)
    with pytest.raises(InvalidSymbolError):
        svc.get_quote(bad)
    assert p.calls == 0


# --- yFinance provider ---
def fake_ticker(monkeypatch, frame=None, error=None):
    class Ticker:
        def __init__(self, symbol):
            pass

        def history(self, period):
            if error:
                raise error
            return frame

    monkeypatch.setattr(md.yf, "Ticker", Ticker)


def test_yfinance_quote(monkeypatch):
    idx = pd.date_range("2026-10-29", periods=2, tz="UTC")
    fake_ticker(monkeypatch, pd.DataFrame({"Close": [100.0, 110.0]}, index=idx))
    quote = YFinanceProvider().get_quote("AAPL")
    assert quote.price == 110
    assert quote.change_pct == pytest.approx(10)


def test_yfinance_empty_means_provider_error(monkeypatch):
    fake_ticker(monkeypatch, pd.DataFrame())
    with pytest.raises(ProviderError):
        YFinanceProvider().get_quote("NOPE")


def test_yfinance_rate_limit_detected(monkeypatch):
    class YFRateLimitError(Exception):
        pass

    fake_ticker(monkeypatch, error=YFRateLimitError())
    with pytest.raises(RateLimitError):
        YFinanceProvider().get_quote("AAPL")


def test_yfinance_other_failure(monkeypatch):
    fake_ticker(monkeypatch, error=KeyError("boom"))
    with pytest.raises(ProviderError):
        YFinanceProvider().get_history("AAPL", "1mo")


# --- Alpha Vantage provider ---
class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload=None, error=None):
        self.payload, self.error = payload, error

    def get(self, url, params, timeout):
        if self.error:
            raise self.error
        return FakeResponse(self.payload)


def av(payload=None, error=None):
    return AlphaVantageProvider("SECRET", session=FakeSession(payload, error))


def test_alpha_vantage_quote():
    quote = av({"Global Quote": {
        "05. price": "110", "08. previous close": "100", "07. latest trading day": "2026-10-30",
    }}).get_quote("AAPL")
    assert quote.price == 110 and quote.change_pct == pytest.approx(10)


def test_alpha_vantage_unknown_symbol():
    with pytest.raises(ProviderError):
        av({"Global Quote": {}}).get_quote("NOPE")


@pytest.mark.parametrize("field", ["Note", "Information"])
def test_alpha_vantage_rate_limit_inside_http_200(field):
    with pytest.raises(RateLimitError):
        av({field: "limit reached"}).get_quote("AAPL")


def test_alpha_vantage_error_message():
    with pytest.raises(ProviderError):
        av({"Error Message": "bad"}).get_history("AAPL", "1mo")


def test_alpha_vantage_without_key_fails_cleanly():
    with pytest.raises(ProviderError):
        AlphaVantageProvider(None).get_quote("AAPL")


def test_alpha_vantage_errors_never_expose_the_key():
    leaky = requests.ConnectionError("failed: https://x/query?apikey=SECRET")
    with pytest.raises(ProviderError) as info:
        av(error=leaky).get_quote("AAPL")
    assert "SECRET" not in str(info.value)
    assert info.value.__suppress_context__ and info.value.__cause__ is None


def test_alpha_vantage_history_sorted_and_trimmed():
    series = {"2026-10-30": {"4. close": "3"}, "2026-10-29": {"4. close": "2"}, "2026-01-01": {"4. close": "1"}}
    history = av({"Time Series (Daily)": series}).get_history("AAPL", "1mo")
    assert history.closes.tolist() == [2.0, 3.0]


def test_alpha_vantage_malformed_history():
    with pytest.raises(ProviderError):
        av({"Time Series (Daily)": {"2026-10-30": {}}}).get_history("AAPL", "1mo")
