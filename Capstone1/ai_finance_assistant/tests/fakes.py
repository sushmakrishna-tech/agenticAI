"""A market with no network, shared by the dashboard and app tests."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.data.market_data import MarketDataError, PriceHistory, Quote, normalize_symbol

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)
PRICES = {"AAPL": 200.0, "VOO": 500.0, "SPY": 600.0, "^GSPC": 7000.0, "^DJI": 45000.0, "^IXIC": 22000.0}


def history(symbol: str, days: int = 120) -> pd.Series:
    rng = np.random.default_rng(sum(map(ord, symbol)))
    closes = PRICES[symbol] * np.cumprod(1 + rng.normal(0.0005, 0.01, days))
    return pd.Series(closes, index=pd.bdate_range("2026-05-01", periods=days))


class FakeMarket:
    def __init__(self, prices=None):
        self.prices = PRICES if prices is None else prices

    def get_quote(self, symbol):
        symbol = normalize_symbol(symbol)
        if symbol not in self.prices:
            raise MarketDataError(symbol)
        return Quote(symbol, self.prices[symbol], self.prices[symbol] * 0.99, 1.01, NOW, "fake")

    def get_history(self, symbol, period="1y"):
        symbol = normalize_symbol(symbol)
        if symbol not in self.prices:
            raise MarketDataError(symbol)
        return PriceHistory(symbol, history(symbol), "fake")
