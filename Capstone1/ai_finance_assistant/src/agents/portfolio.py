"""Portfolio Analysis: metrics come from code, and the LLM only explains them."""
from pydantic import BaseModel

from src.agents.common import explain, extract, money, output
from src.data.market_data import MarketDataError
from src.utils.metrics import annotate, traced
from src.utils.portfolio_metrics import Holding, PortfolioAnalysis, analyze_portfolio

ASK = ("To analyze your portfolio, I need each ticker and number of shares, plus what you paid per "
       "share if you know it. For example: \"10 shares of AAPL at $150 and 5 shares of VOO\".")


class HoldingDetail(BaseModel):
    symbol: str
    shares: float
    cost_per_share: float | None = None


class HoldingsRequest(BaseModel):
    holdings: list[HoldingDetail]


INSTRUCTION = ("List the investments the user says they currently own. Include only holdings with a "
               "stated number of shares. Convert company names to US tickers. Use the latest amounts "
               "if they changed. Never invent holdings; return an empty list if there are none.")


def format_analysis(a: PortfolioAnalysis) -> str:
    lines = [f"Total value: {money(a.total_value)}"]
    if a.total_gain is not None:
        lines.append(f"Gain on holdings with a known cost: {money(a.total_gain)}"
                     + (f" ({a.total_gain_pct:.1f}%)" if a.total_gain_pct is not None else ""))
    lines.append(f"Effective number of holdings, counting a fund as one: {a.effective_holdings:.1f}")
    for p in a.positions:
        lines.append(f"- {p.symbol}: {p.shares:g} shares at {money(p.price)} = {money(p.value)}, "
                     f"{p.weight_pct:.1f}% of portfolio"
                     + (f", gain {money(p.gain)}" if p.gain is not None else ""))
    if a.risk:
        r = a.risk
        lines.append(f"Past behavior of today's mix, {r.start} to {r.end} ({r.days} trading days; "
                     f"not the user's actual results): return {r.period_return_pct:.1f}%, annualized "
                     f"volatility {r.volatility_pct:.1f}%, largest drop {r.max_drawdown_pct:.1f}%")
        if r.sharpe_ratio is not None:
            lines.append(f"Sharpe ratio: {r.sharpe_ratio:.2f}")
        if r.beta is not None:
            lines.append(f"Beta against the S&P 500: {r.beta:.2f}")
        if r.benchmark_return_pct is not None:
            lines.append(f"S&P 500 price return, same period, excluding dividends: {r.benchmark_return_pct:.1f}%")
    else:
        lines.append("Risk metrics are unavailable because there is too little price history.")
    lines += [f"Note: {note}" for note in a.observations]
    return "\n".join(lines)


def make_portfolio_agent(llm, get_market):
    def run(state):
        details = state.get("holdings")
        if not details:
            request = extract(llm, HoldingsRequest, INSTRUCTION, state)
            details = [h.model_dump() for h in request.holdings] if request else []
        if not details:
            return output("portfolio", ASK)
        try:
            holdings = [Holding(d["symbol"], d["shares"], d.get("cost_per_share")) for d in details]
            with traced("portfolio_metrics", holdings=len(holdings)):
                analysis = analyze_portfolio(holdings, get_market())
                annotate(unavailable=len(analysis.unavailable))
        except ValueError as exc:
            return output("portfolio", f"I can't analyze that yet: {exc}.")
        except MarketDataError:
            return output("portfolio", "I couldn't get prices for those holdings right now.", error=True)
        facts = format_analysis(analysis)
        task = ("Explain what these results mean and what the notes show about diversification and "
                "risk. Mention any missing or out-of-date data. Don't suggest specific trades.")
        return output("portfolio", explain(llm, task, facts, state),
                      ["Market data: Yahoo Finance or Alpha Vantage"], facts=facts)

    return run
