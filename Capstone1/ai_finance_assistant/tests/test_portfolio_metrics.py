from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.data.market_data import InvalidSymbolError, MarketDataError, PriceHistory, Quote, normalize_symbol
from src.utils.portfolio_metrics import Holding, analyze_portfolio, compute_risk, merge_holdings

NOW = datetime(2026, 10, 30, tzinfo=timezone.utc)
DATES = pd.bdate_range("2025-10-01", periods=260)


def steady(daily=0.001, n=260, tz=None):
    index = DATES[:n].tz_localize(tz) if tz else DATES[:n]
    return pd.Series(100 * (1 + daily) ** np.arange(n), index=index)


def wavy(scale=1.0, seed=1, n=260):
    returns = np.random.default_rng(seed).normal(0, 0.01, n) * scale
    return pd.Series(100 * np.cumprod(1 + returns), index=DATES[:n])


class FakeMarket:
    def __init__(self, prices, histories=None, stale=()):
        self.prices, self.histories, self.stale = prices, histories or {}, stale

    def get_quote(self, symbol):
        normalize_symbol(symbol)
        if symbol not in self.prices:
            raise MarketDataError(symbol)
        return Quote(symbol, self.prices[symbol], None, None, NOW, "fake", symbol in self.stale)

    def get_history(self, symbol, period="1y"):
        if symbol not in self.histories:
            raise MarketDataError(symbol)
        return PriceHistory(symbol, self.histories[symbol], "fake")


def test_values_weights_and_gains_by_hand():
    market = FakeMarket({"AAA": 100.0, "BBB": 50.0})
    result = analyze_portfolio([Holding("AAA", 3, 80), Holding("BBB", 2, 60)], market)
    assert result.total_value == 400
    assert [p.symbol for p in result.positions] == ["AAA", "BBB"]
    assert result.positions[0].weight_pct == 75
    assert result.positions[0].gain == 60
    assert result.positions[1].gain == -20
    assert result.total_gain == 40
    assert result.total_gain_pct == pytest.approx(40 / 360 * 100)
    # 1 / (0.75^2 + 0.25^2) = 1.6
    assert result.effective_holdings == pytest.approx(1.6)


def test_unknown_cost_stays_out_of_gains():
    market = FakeMarket({"AAA": 100.0, "BBB": 50.0})
    result = analyze_portfolio([Holding("AAA", 1, 50), Holding("BBB", 1)], market)
    assert result.total_gain == 50
    assert result.total_gain_pct == 100
    assert result.positions[1].gain is None


def test_no_costs_means_no_gain_figure():
    result = analyze_portfolio([Holding("AAA", 1)], FakeMarket({"AAA": 10.0}))
    assert result.total_gain is None and result.total_gain_pct is None


def test_zero_cost_does_not_divide_by_zero():
    result = analyze_portfolio([Holding("AAA", 1, 0)], FakeMarket({"AAA": 10.0}))
    assert result.positions[0].gain == 10
    assert result.positions[0].gain_pct is None
    assert result.total_gain_pct is None


def test_repeat_purchases_merge_with_average_cost():
    merged = merge_holdings([Holding("aapl", 1, 100), Holding("AAPL", 3, 200)])
    assert merged == [Holding("AAPL", 4, 175)]


def test_merge_with_unknown_cost_loses_cost():
    assert merge_holdings([Holding("A", 1, 100), Holding("A", 1)]) == [Holding("A", 2, None)]


@pytest.mark.parametrize("kwargs", [
    {"shares": 0}, {"shares": -1}, {"shares": float("nan")}, {"shares": 1, "cost_per_share": -5},
])
def test_bad_holdings_rejected(kwargs):
    with pytest.raises(ValueError):
        Holding("AAA", **kwargs)


def test_empty_portfolio_rejected():
    with pytest.raises(ValueError):
        analyze_portfolio([], FakeMarket({}))


def test_all_prices_missing_raises():
    with pytest.raises(MarketDataError):
        analyze_portfolio([Holding("AAA", 1)], FakeMarket({}))


def test_missing_price_is_reported_and_left_out():
    result = analyze_portfolio([Holding("AAA", 1), Holding("NOPE", 1)], FakeMarket({"AAA": 10.0}))
    assert result.total_value == 10
    assert result.unavailable == ["NOPE"]
    assert any("NOPE" in note for note in result.observations)


def test_invalid_symbol_is_treated_as_unavailable():
    assert issubclass(InvalidSymbolError, MarketDataError)
    result = analyze_portfolio([Holding("AAA", 1), Holding("bad symbol", 1)], FakeMarket({"AAA": 10.0}))
    assert result.unavailable == ["BAD SYMBOL"]


def test_steady_growth_risk_by_hand():
    risk = compute_risk({"A": steady(0.001)}, {"A": 100}, None, 0.04)
    assert risk.days == 259
    assert risk.period_return_pct == pytest.approx((1.001**259 - 1) * 100)
    assert risk.annualized_return_pct == pytest.approx((1.001**252 - 1) * 100)
    assert risk.volatility_pct == pytest.approx(0, abs=1e-6)
    assert risk.max_drawdown_pct == pytest.approx(0)
    assert risk.beta is None


def test_max_drawdown_by_hand():
    prices = [100.0] * 15 + [200.0, 100.0, 150.0] + [150.0] * 10
    series = pd.Series(prices, index=DATES[: len(prices)])
    assert compute_risk({"A": series}, {"A": 1}, None, 0).max_drawdown_pct == pytest.approx(-50)


def test_drawdown_counts_a_fall_from_the_starting_value():
    series = pd.Series([100.0, 50.0] + [50.0] * 25, index=DATES[:27])
    assert compute_risk({"A": series}, {"A": 1}, None, 0).max_drawdown_pct == pytest.approx(-50)


def test_portfolio_uses_weighted_returns():
    risk = compute_risk(
        {"A": steady(0.01, 30), "B": steady(0.0, 30)}, {"A": 75, "B": 25}, None, 0
    )
    assert risk.period_return_pct == pytest.approx((1.0075**29 - 1) * 100)


def test_beta_by_hand():
    bench = wavy()
    bench_returns = bench.pct_change().dropna()
    double = pd.Series(100 * np.cumprod(np.r_[1, 1 + 2 * bench_returns]), index=bench.index)
    assert compute_risk({"A": double}, {"A": 1}, bench, 0).beta == pytest.approx(2)
    assert compute_risk({"A": bench}, {"A": 1}, bench, 0).beta == pytest.approx(1)


def test_sharpe_by_hand():
    series = wavy()
    risk = compute_risk({"A": series}, {"A": 1}, None, 0.04)
    expected = (risk.annualized_return_pct - 4) / risk.volatility_pct
    assert risk.sharpe_ratio == pytest.approx(expected)


def test_short_history_is_not_annualized():
    risk = compute_risk({"A": wavy(n=60)}, {"A": 1}, None, 0.04)
    assert risk.days == 59
    assert risk.annualized_return_pct is None and risk.sharpe_ratio is None
    assert risk.volatility_pct > 0


def test_too_little_history_gives_no_risk():
    assert compute_risk({"A": wavy(n=10)}, {"A": 1}, None, 0) is None
    assert compute_risk({}, {}, None, 0) is None


def test_flat_benchmark_gives_no_beta_without_crashing():
    flat = pd.Series(100.0, index=DATES)
    assert compute_risk({"A": wavy()}, {"A": 1}, flat, 0).beta is None


def test_timezone_aware_and_naive_sources_line_up():
    risk = compute_risk(
        {"A": steady(tz="America/New_York"), "B": steady()}, {"A": 1, "B": 1}, steady(tz="UTC"), 0
    )
    assert risk.days == 259
    assert risk.benchmark_return_pct is not None


def test_only_shared_dates_are_used():
    risk = compute_risk({"A": wavy(), "B": wavy(seed=2, n=60)}, {"A": 1, "B": 1}, None, 0)
    assert risk.days == 59


def test_observations_flag_concentration_and_stale_prices():
    prices = {"A": 90.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0}
    result = analyze_portfolio([Holding(s, 1) for s in prices], FakeMarket(prices, stale={"E"}))
    text = " ".join(result.observations)
    assert "A is 90.0%" in text
    assert "three largest" in text
    assert "out of date for: E" in text
    assert "symbol(s)" not in text


def test_observations_flag_few_holdings_high_beta_short_history():
    bench = wavy(n=60)
    bench_returns = bench.pct_change().dropna()
    risky = pd.Series(100 * np.cumprod(np.r_[1, 1 + 2 * bench_returns]), index=bench.index)
    market = FakeMarket({"A": 1.0}, {"A": risky, "^GSPC": bench})
    text = " ".join(analyze_portfolio([Holding("A", 1)], market).observations)
    assert "holds 1 symbol" in text and "Beta of 2.00" in text and "only 59" in text


def test_observations_never_tell_the_user_what_to_do():
    market = FakeMarket({"A": 90.0, "B": 10.0})
    text = " ".join(analyze_portfolio([Holding("A", 1), Holding("B", 1)], market).observations).lower()
    assert not any(word in text for word in ("should", "sell", "buy", "must"))


def test_missing_history_is_reported():
    result = analyze_portfolio([Holding("AAA", 1)], FakeMarket({"AAA": 10.0}))
    assert result.risk is None
    assert result.excluded_from_risk == ["AAA"]


def test_risk_includes_benchmark_when_available():
    market = FakeMarket({"AAA": 10.0}, {"AAA": wavy(), "^GSPC": wavy(seed=3)})
    assert analyze_portfolio([Holding("AAA", 1)], market).risk.benchmark_return_pct is not None
