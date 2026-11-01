"""Portfolio calculations: plain Python, so the LLM explains numbers but never makes them.

Risk metrics use today's weights applied to past daily returns. They describe
how the current mix would have behaved, not what the user actually earned.
"""
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from src.core.config import get_config
from src.data.market_data import MarketDataError, MarketDataService, normalize_symbol

TRADING_DAYS = 252
MIN_RISK_DAYS = 20  # fewer daily returns than this is too little to mean anything
MIN_ANNUALIZE_DAYS = 200  # below this, annualizing would exaggerate a short period
SINGLE_HOLDING_LIMIT = 25.0
TOP_THREE_LIMIT = 60.0
FEW_HOLDINGS = 5
HIGH_BETA = 1.2


@dataclass(frozen=True)
class Holding:
    symbol: str
    shares: float
    cost_per_share: float | None = None  # optional: many beginners won't know it

    def __post_init__(self):
        if not np.isfinite(self.shares) or self.shares <= 0:
            raise ValueError(f"{self.symbol}: shares must be greater than 0")
        if self.cost_per_share is not None and (
            not np.isfinite(self.cost_per_share) or self.cost_per_share < 0
        ):
            raise ValueError(f"{self.symbol}: cost per share cannot be negative")


@dataclass(frozen=True)
class Position:
    symbol: str
    shares: float
    price: float
    value: float
    weight_pct: float
    day_change_pct: float | None
    cost_basis: float | None
    gain: float | None
    gain_pct: float | None
    as_of: datetime
    stale: bool


@dataclass(frozen=True)
class RiskMetrics:
    days: int
    start: str
    end: str
    period_return_pct: float
    annualized_return_pct: float | None
    volatility_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float | None
    beta: float | None
    benchmark_return_pct: float | None


@dataclass(frozen=True)
class PortfolioAnalysis:
    positions: list[Position]
    total_value: float
    total_gain: float | None  # covers only positions with a known cost
    total_gain_pct: float | None
    effective_holdings: float  # counts a fund as one holding, so it understates funds
    risk: RiskMetrics | None
    unavailable: list[str] = field(default_factory=list)  # no price, left out of totals
    excluded_from_risk: list[str] = field(default_factory=list)  # priced, but no history
    observations: list[str] = field(default_factory=list)


def merge_holdings(holdings: list[Holding]) -> list[Holding]:
    """Combine repeat purchases of a symbol, averaging the cost."""
    lots: dict[str, list[Holding]] = {}
    for holding in holdings:
        lots.setdefault(holding.symbol.strip().upper(), []).append(holding)
    merged = []
    for symbol, group in lots.items():
        shares = sum(h.shares for h in group)
        known = all(h.cost_per_share is not None for h in group)
        cost = sum(h.shares * h.cost_per_share for h in group) / shares if known else None
        merged.append(Holding(symbol, shares, cost))
    return merged


def _daily(closes: pd.Series) -> pd.Series:
    """One value per calendar date, so sources with different timezones line up."""
    series = closes.copy()
    series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
    return series[~series.index.duplicated(keep="last")]


def compute_risk(
    closes: dict[str, pd.Series],
    weights: dict[str, float],
    benchmark: pd.Series | None,
    risk_free_rate: float,
) -> RiskMetrics | None:
    """Risk metrics for a fixed-weight mix, over the dates all holdings share."""
    if not closes:
        return None
    returns = pd.DataFrame({s: _daily(c) for s, c in closes.items()}).dropna().pct_change().dropna()
    if len(returns) < MIN_RISK_DAYS:
        return None

    total_weight = sum(weights[s] for s in closes)
    portfolio = sum(returns[s] * (weights[s] / total_weight) for s in closes)
    growth = (1 + portfolio).cumprod()
    days = len(portfolio)
    period_return = growth.iloc[-1] - 1
    volatility = portfolio.std() * np.sqrt(TRADING_DAYS)
    drawdown = (growth / growth.cummax().clip(lower=1) - 1).min()

    annualized = sharpe = None
    if days >= MIN_ANNUALIZE_DAYS:
        annualized = (1 + period_return) ** (TRADING_DAYS / days) - 1
        if volatility > 1e-12:
            sharpe = (annualized - risk_free_rate) / volatility

    beta = benchmark_return = None
    if benchmark is not None:
        bench = _daily(benchmark).pct_change().dropna()
        paired = pd.concat([portfolio, bench], axis=1, join="inner").dropna()
        if len(paired) >= MIN_RISK_DAYS:
            variance = paired.iloc[:, 1].var()
            if variance > 1e-12:
                beta = paired.iloc[:, 0].cov(paired.iloc[:, 1]) / variance
            benchmark_return = (1 + paired.iloc[:, 1]).prod() - 1

    return RiskMetrics(
        days=days,
        start=returns.index[0].date().isoformat(),
        end=returns.index[-1].date().isoformat(),
        period_return_pct=float(period_return * 100),
        annualized_return_pct=None if annualized is None else float(annualized * 100),
        volatility_pct=float(volatility * 100),
        max_drawdown_pct=float(drawdown * 100),
        sharpe_ratio=None if sharpe is None else float(sharpe),
        beta=None if beta is None else float(beta),
        benchmark_return_pct=None if benchmark_return is None else float(benchmark_return * 100),
    )


def _observations(positions, risk, unavailable, excluded) -> list[str]:
    """Factual notes for the agent to explain. They describe; they never say what to do."""
    notes = []
    top = positions[0]
    if top.weight_pct >= SINGLE_HOLDING_LIMIT:
        notes.append(
            f"{top.symbol} is {top.weight_pct:.1f}% of the portfolio. That is concentrated for a single "
            "stock, while a broad fund spreads it across many companies.")
    top_three = sum(p.weight_pct for p in positions[:3])
    if len(positions) > 3 and top_three >= TOP_THREE_LIMIT:
        notes.append(f"The three largest holdings make up {top_three:.1f}% of the portfolio.")
    if len(positions) < FEW_HOLDINGS:
        notes.append(f"The portfolio holds {len(positions)} symbol(s). Few holdings can still be diversified if they are broad funds.")
    if risk and risk.beta is not None and risk.beta >= HIGH_BETA:
        notes.append(f"Beta of {risk.beta:.2f} means it has moved more sharply than the benchmark.")
    if risk and risk.days < MIN_ANNUALIZE_DAYS:
        notes.append(f"Risk metrics cover only {risk.days} trading days, so treat them with caution.")
    stale = [p.symbol for p in positions if p.stale]
    if stale:
        notes.append(f"Prices may be out of date for: {', '.join(stale)}.")
    if unavailable:
        notes.append(f"Left out of the totals because no price was found: {', '.join(unavailable)}.")
    if excluded:
        notes.append(f"Left out of the risk metrics because no price history was found: {', '.join(excluded)}.")
    return notes


def analyze_portfolio(holdings: list[Holding], market: MarketDataService) -> PortfolioAnalysis:
    if not holdings:
        raise ValueError("Add at least one holding to analyze")
    cfg = get_config()["portfolio"]

    quotes, unavailable = {}, []
    merged = merge_holdings(holdings)
    for holding in merged:
        try:
            quotes[holding.symbol] = market.get_quote(normalize_symbol(holding.symbol))
        except MarketDataError:
            unavailable.append(holding.symbol)
    if not quotes:
        raise MarketDataError("No prices available for any holding")

    total = sum(h.shares * quotes[h.symbol].price for h in merged if h.symbol in quotes)
    positions = []
    for holding in merged:
        if holding.symbol not in quotes:
            continue
        quote = quotes[holding.symbol]
        value = holding.shares * quote.price
        cost = None if holding.cost_per_share is None else holding.shares * holding.cost_per_share
        positions.append(Position(
            symbol=holding.symbol,
            shares=holding.shares,
            price=quote.price,
            value=value,
            weight_pct=value / total * 100,
            day_change_pct=quote.change_pct,
            cost_basis=cost,
            gain=None if cost is None else value - cost,
            gain_pct=(value - cost) / cost * 100 if cost else None,
            as_of=quote.as_of,
            stale=quote.stale,
        ))
    positions.sort(key=lambda p: p.value, reverse=True)

    closes, excluded = {}, []
    for position in positions:
        try:
            closes[position.symbol] = market.get_history(position.symbol, cfg["risk_period"]).closes
        except MarketDataError:
            excluded.append(position.symbol)
    try:
        benchmark = market.get_history(cfg["benchmark"], cfg["risk_period"]).closes
    except MarketDataError:
        benchmark = None
    weights = {p.symbol: p.weight_pct for p in positions}
    risk = compute_risk(closes, weights, benchmark, cfg["risk_free_rate"])

    with_cost = [p for p in positions if p.cost_basis is not None]
    cost_total = sum(p.cost_basis for p in with_cost)
    gain_total = sum(p.gain for p in with_cost) if with_cost else None
    effective = 1 / sum((p.weight_pct / 100) ** 2 for p in positions)

    return PortfolioAnalysis(
        positions=positions,
        total_value=total,
        total_gain=gain_total,
        total_gain_pct=gain_total / cost_total * 100 if cost_total else None,
        effective_holdings=effective,
        risk=risk,
        unavailable=unavailable,
        excluded_from_risk=excluded,
        observations=_observations(positions, risk, unavailable, excluded),
    )
