"""Market Analysis: live quotes and recent trends, described but never predicted."""
from pydantic import BaseModel, Field

from src.agents.common import explain, extract, money, output
from src.data.market_data import MarketDataError

OVERVIEW = {"^GSPC": "S&P 500", "^DJI": "Dow Jones Industrial Average", "^IXIC": "Nasdaq Composite"}
MAX_SYMBOLS = 5


class MarketRequest(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    asks_prediction: bool = False


INSTRUCTION = ("Read the user's latest message and list the tickers it asks about. Convert company "
               "names to US tickers, and use ^GSPC for the S&P 500, ^DJI for the Dow, and ^IXIC for "
               "the Nasdaq. Use earlier messages only to resolve words like \"it\". Return an empty "
               "list for a general question about the market. Set asks_prediction to true if the "
               "latest message asks what a price or market will do in the future.")
NO_PREDICTION = "I can't predict what prices or markets will do. Here is what has already happened:"


def describe(symbol: str, market) -> tuple[str, str | None]:
    """One line of facts for a symbol, plus which provider supplied it."""
    label = OVERVIEW.get(symbol, symbol)
    try:
        quote = market.get_quote(symbol)
    except MarketDataError:
        return f"- {label}: no data available", None
    # Index levels are points, not dollars.
    price = f"{quote.price:,.2f}" if symbol in OVERVIEW else money(quote.price)
    line = f"- {label}: {price} as of {quote.as_of.date().isoformat()}"
    if quote.change_pct is not None:
        line += f", {quote.change_pct:+.2f}% on the day"
    try:
        closes = market.get_history(symbol, "1mo").closes
        if len(closes) >= 2 and closes.iloc[0]:
            line += f", {(closes.iloc[-1] / closes.iloc[0] - 1) * 100:+.1f}% over about one month"
    except MarketDataError:
        pass
    if quote.stale:
        line += " (may be out of date)"
    return line, quote.source


def make_market_agent(llm, get_market):
    def run(state):
        request = extract(llm, MarketRequest, INSTRUCTION, state)
        symbols = (request.symbols if request else [])[:MAX_SYMBOLS] or list(OVERVIEW)
        market = get_market()
        described = [describe(symbol, market) for symbol in symbols]
        if not any(source for _, source in described):
            return output("market", "Market data is unavailable right now.", error=True)
        facts = "\n".join(line for line, _ in described)
        task = ("Describe what these figures show, including any dates. Don't explain why prices moved or "
                "predict them, and don't call past performance a good or positive sign.")
        sources = [f"Market data: {s}" for s in dict.fromkeys(s for _, s in described if s)]
        answer = explain(llm, task, facts, state)
        # Prompt rules didn't reliably produce this, so code adds it.
        if request and request.asks_prediction:
            answer = f"{NO_PREDICTION}\n\n{answer}"
        return output("market", answer, sources, facts=facts)

    return run
