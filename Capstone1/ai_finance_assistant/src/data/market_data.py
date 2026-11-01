"""Market data with layered fallbacks.

Lookup order: fresh cache -> yFinance -> Alpha Vantage -> stale cache -> error.
Each provider exposes the same two methods, so the service doesn't care which
one answered, and tests can swap in fakes without touching the network.
"""
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Protocol

import pandas as pd
import requests
import yfinance as yf

from src.core.config import get_alpha_vantage_key, get_config

logger = logging.getLogger(__name__)

PERIOD_DAYS = {"1mo": 30, "3mo": 91, "6mo": 182, "1y": 365}
_SYMBOL = re.compile(r"^[A-Z0-9^][A-Z0-9.^=\-]{0,14}$")  # AAPL, BRK-B, ^GSPC


class MarketDataError(Exception):
    """No provider could answer, and nothing usable was cached."""


class InvalidSymbolError(MarketDataError):
    """The text can't be a ticker symbol, so no provider was called."""


class ProviderError(Exception):
    """One provider failed; the service moves on to the next."""


class RateLimitError(ProviderError):
    """The provider asked us to slow down, so skip it for a while."""


@dataclass(frozen=True)
class Quote:
    symbol: str
    price: float
    previous_close: float | None
    change_pct: float | None
    as_of: datetime  # the market timestamp of this price
    source: str
    stale: bool = False  # True when served from old cache after providers failed


@dataclass(frozen=True)
class PriceHistory:
    symbol: str
    closes: pd.Series  # daily closes, oldest first
    source: str
    stale: bool = False


class Provider(Protocol):
    name: str

    def get_quote(self, symbol: str) -> Quote: ...

    def get_history(self, symbol: str, period: str) -> PriceHistory: ...


def _change_pct(price: float, previous: float | None) -> float | None:
    return (price - previous) / previous * 100 if previous else None


class YFinanceProvider:
    name = "yfinance"

    def _closes(self, symbol: str, period: str) -> pd.Series:
        try:
            frame = yf.Ticker(symbol).history(period=period)
        except Exception as exc:
            if "rate" in type(exc).__name__.lower():
                raise RateLimitError(f"yfinance rate limited: {symbol}") from exc
            raise ProviderError(f"yfinance failed for {symbol}: {type(exc).__name__}") from exc
        if frame.empty or "Close" not in frame:
            raise ProviderError(f"yfinance has no data for {symbol}")
        return frame["Close"].dropna()

    def get_quote(self, symbol: str) -> Quote:
        closes = self._closes(symbol, "5d")
        if closes.empty:
            raise ProviderError(f"yfinance has no prices for {symbol}")
        price = float(closes.iloc[-1])
        previous = float(closes.iloc[-2]) if len(closes) > 1 else None
        return Quote(symbol, price, previous, _change_pct(price, previous),
                     closes.index[-1].to_pydatetime(), self.name)

    def get_history(self, symbol: str, period: str) -> PriceHistory:
        return PriceHistory(symbol, self._closes(symbol, period), self.name)


class AlphaVantageProvider:
    """Backup source. The free daily series returns about 100 trading days, so
    a longer request comes back shorter: check the dates on the result."""

    name = "alpha_vantage"
    URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str | None, timeout: float = 10, session=None):
        self.api_key = api_key
        self.timeout = timeout
        self.session = session or requests.Session()

    def _call(self, **params) -> dict:
        if not self.api_key:
            raise ProviderError("Alpha Vantage key not configured")
        try:
            response = self.session.get(
                self.URL, params={**params, "apikey": self.api_key}, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            # "from None" on purpose: the original error text can contain the
            # URL, which contains the API key, and this would end up in logs.
            raise ProviderError(f"Alpha Vantage request failed: {type(exc).__name__}") from None
        # Alpha Vantage reports rate limits inside an HTTP 200 response.
        if "Note" in data or "Information" in data:
            raise RateLimitError("Alpha Vantage rate limit reached")
        if "Error Message" in data:
            raise ProviderError("Alpha Vantage rejected the request")
        return data

    def get_quote(self, symbol: str) -> Quote:
        data = self._call(function="GLOBAL_QUOTE", symbol=symbol).get("Global Quote") or {}
        try:
            price = float(data["05. price"])
            previous = float(data["08. previous close"])
            day = datetime.fromisoformat(data["07. latest trading day"])
        except (KeyError, ValueError):
            raise ProviderError(f"Alpha Vantage has no quote for {symbol}") from None
        return Quote(symbol, price, previous, _change_pct(price, previous),
                     day.replace(tzinfo=timezone.utc), self.name)

    def get_history(self, symbol: str, period: str) -> PriceHistory:
        data = self._call(function="TIME_SERIES_DAILY", symbol=symbol).get("Time Series (Daily)")
        if not data:
            raise ProviderError(f"Alpha Vantage has no history for {symbol}")
        try:
            closes = pd.Series(
                {pd.Timestamp(day): float(row["4. close"]) for day, row in data.items()}
            ).sort_index()
        except (KeyError, ValueError):
            raise ProviderError(f"Alpha Vantage sent malformed history for {symbol}") from None
        cutoff = closes.index[-1] - timedelta(days=PERIOD_DAYS[period])
        return PriceHistory(symbol, closes[closes.index >= cutoff], self.name)


class MarketDataService:
    """Applies caching and fallbacks. Thread-safe, because agents run in parallel."""

    def __init__(self, providers: list[Provider], quote_ttl: float = 60, history_ttl: float = 3600,
                 cooldown: float = 60, clock: Callable[[], float] = time.monotonic):
        self.providers = providers
        self.quote_ttl = quote_ttl
        self.history_ttl = history_ttl
        self.cooldown = cooldown
        self.clock = clock  # injectable, so tests can "wait" an hour instantly
        self._cache: dict[tuple, tuple[float, object]] = {}
        self._blocked_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def _fetch(self, key: tuple, ttl: float, call: Callable[[Provider], object]):
        with self._lock:
            cached = self._cache.get(key)
        if cached and self.clock() - cached[0] < ttl:
            return cached[1]

        for provider in self.providers:
            with self._lock:
                if self.clock() < self._blocked_until.get(provider.name, 0):
                    continue
            try:
                result = call(provider)
            except RateLimitError as exc:
                logger.warning("%s; pausing this provider", exc)
                with self._lock:
                    self._blocked_until[provider.name] = self.clock() + self.cooldown
            except ProviderError as exc:
                logger.warning("%s", exc)
            else:
                with self._lock:
                    self._cache[key] = (self.clock(), result)
                return result

        if cached:
            logger.warning("All providers failed; serving stale data for %s", key)
            return replace(cached[1], stale=True)
        raise MarketDataError(f"No market data available for {key[1]}")

    def get_quote(self, symbol: str) -> Quote:
        symbol = normalize_symbol(symbol)
        return self._fetch(("quote", symbol), self.quote_ttl, lambda p: p.get_quote(symbol))

    def get_history(self, symbol: str, period: str = "1y") -> PriceHistory:
        symbol = normalize_symbol(symbol)
        if period not in PERIOD_DAYS:
            raise ValueError(f"period must be one of {sorted(PERIOD_DAYS)}")
        return self._fetch(("history", symbol, period), self.history_ttl,
                           lambda p: p.get_history(symbol, period))


def normalize_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    if not _SYMBOL.fullmatch(cleaned):
        raise InvalidSymbolError(f"Not a valid ticker symbol: {symbol!r}")
    return cleaned


@lru_cache
def get_market_data() -> MarketDataService:
    """The shared service, so all agents and tabs use one cache."""
    cfg = get_config()["market_data"]
    return MarketDataService(
        [YFinanceProvider(),
         AlphaVantageProvider(get_alpha_vantage_key(), cfg["request_timeout_seconds"])],
        quote_ttl=cfg["quote_ttl_seconds"],
        history_ttl=cfg["history_ttl_seconds"],
        cooldown=cfg["rate_limit_cooldown_seconds"],
    )
