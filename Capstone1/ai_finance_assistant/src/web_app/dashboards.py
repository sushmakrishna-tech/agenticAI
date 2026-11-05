"""The Portfolio, Market, and Goals tabs. Figures come from code, as in chat."""
import pandas as pd
import streamlit as st

from src.agents.market import OVERVIEW
from src.data.accounts import AccountStore
from src.data.market_data import PERIOD_DAYS, MarketDataError, get_market_data
from src.utils.goal_projections import GoalInput, compare_risk_profiles
from src.utils.portfolio_metrics import Holding, analyze_portfolio
from src.web_app.charts import allocation_chart, goal_fan_chart, growth_chart, price_chart

EDUCATION_NOTE = "For education only. Past results don't predict future ones."
HOLDING_COLUMNS = ["symbol", "shares", "cost_per_share"]


def money(value: float | None, decimals: int = 2) -> str:
    return "—" if value is None else f"${value:,.{decimals}f}"


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}%"


def parse_holdings(frame: pd.DataFrame) -> list[dict]:
    """Rows from the editor, checked by the same rules as the chat. Blank rows are skipped."""
    rows = []
    for record in frame.to_dict("records"):
        symbol = str(record.get("symbol") or "").strip().upper()
        if not symbol or symbol == "NAN":
            continue
        shares = record.get("shares")
        cost = record.get("cost_per_share")
        cost = None if cost is None or pd.isna(cost) else float(cost)
        if shares is None or pd.isna(shares):
            raise ValueError(f"{symbol}: enter the number of shares")
        holding = Holding(symbol, float(shares), cost)  # raises ValueError for bad values
        rows.append({"symbol": holding.symbol, "shares": holding.shares,
                     "cost_per_share": holding.cost_per_share})
    return rows


# --- Portfolio ---
def render_portfolio(store: AccountStore, user: dict) -> None:
    st.subheader("Your holdings")
    st.caption("Saved to your account. The chat uses these when you ask about your portfolio, "
               "unless you list other holdings in your question.")
    saved = store.get_holdings(user["id"])
    with st.form("holdings_form"):
        edited = st.data_editor(
            pd.DataFrame(saved, columns=HOLDING_COLUMNS),
            num_rows="dynamic", width="stretch", key="holdings_editor",
            column_config={
                "symbol": st.column_config.TextColumn("Ticker", required=True, max_chars=15),
                "shares": st.column_config.NumberColumn("Shares", min_value=0.0, format="%.4g"),
                "cost_per_share": st.column_config.NumberColumn(
                    "Cost per share (optional)", min_value=0.0, format="$%.2f"),
            },
        )
        submitted = st.form_submit_button("Save and analyze")
    if submitted:
        try:
            holdings = parse_holdings(edited)
        except ValueError as error:
            st.error(f"Please fix your holdings: {error}.")
        else:
            store.save_holdings(user["id"], holdings)
            st.session_state.pop("portfolio_analysis", None)
            if holdings:
                with st.spinner("Getting prices..."):
                    try:
                        st.session_state.portfolio_analysis = analyze_portfolio(
                            [Holding(**h) for h in holdings], get_market_data())
                    except MarketDataError:
                        st.error("Couldn't get prices for these holdings right now. Please try again.")

    analysis = st.session_state.get("portfolio_analysis")
    if analysis is None:
        if saved and not submitted:
            st.info("Click **Save and analyze** to see your dashboard.")
        return
    show_portfolio(analysis)


def show_portfolio(a) -> None:
    columns = st.columns(3)
    columns[0].metric("Total value", money(a.total_value))
    columns[1].metric("Gain on known costs", money(a.total_gain),
                      pct(a.total_gain_pct) if a.total_gain_pct is not None else None)
    columns[2].metric("Effective holdings", f"{a.effective_holdings:.1f}",
                      help="How many equal-sized holdings your mix behaves like. A fund counts as one.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Allocation**")
        st.plotly_chart(allocation_chart(a), width="stretch", key="allocation_chart")
    with right:
        st.markdown("**$100 in today's mix vs the S&P 500**")
        if a.growth is not None:
            st.plotly_chart(growth_chart(a.growth), width="stretch", key="growth_chart")
            st.caption("Applies today's weights to past prices, so it isn't what you actually earned. "
                       "Excludes dividends.")
        else:
            st.caption("Not enough price history to draw this chart.")

    if a.risk:
        r = a.risk
        st.markdown(f"**Past behavior of today's mix**, {r.start} to {r.end} ({r.days} trading days)")
        cols = st.columns(5)
        cols[0].metric("Return", pct(r.period_return_pct))
        cols[1].metric("S&P 500 return", pct(r.benchmark_return_pct))
        cols[2].metric("Volatility", pct(r.volatility_pct), help="How much daily returns swung, annualized.")
        cols[3].metric("Largest drop", pct(r.max_drawdown_pct), help="Biggest fall from a previous high.")
        cols[4].metric("Beta", "—" if r.beta is None else f"{r.beta:.2f}",
                       help="1.0 moved like the S&P 500. Higher moved more sharply.")

    st.markdown("**Positions**")
    st.dataframe(pd.DataFrame([{
        "Ticker": p.symbol, "Shares": p.shares, "Price": p.price, "Value": p.value,
        "Weight": p.weight_pct, "Day change": p.day_change_pct,
        "Cost per share": None if p.cost_basis is None else p.cost_basis / p.shares,
        "Gain": p.gain, "Gain %": p.gain_pct, "Price may be out of date": p.stale,
    } for p in a.positions]), hide_index=True, width="stretch", column_config={
        "Price": st.column_config.NumberColumn(format="$%.2f"),
        "Value": st.column_config.NumberColumn(format="$%.2f"),
        "Cost per share": st.column_config.NumberColumn(format="$%.2f"),
        "Gain": st.column_config.NumberColumn(format="$%.2f"),
        "Weight": st.column_config.NumberColumn(format="%.1f%%"),
        "Day change": st.column_config.NumberColumn(format="%.2f%%"),
        "Gain %": st.column_config.NumberColumn(format="%.1f%%"),
    })
    for note in a.observations:
        st.caption(note.replace("$", r"\$"))
    st.caption(EDUCATION_NOTE)


# --- Market ---
def render_market() -> None:
    market = get_market_data()
    st.subheader("Major indexes")
    columns = st.columns(len(OVERVIEW))
    for column, (symbol, name) in zip(columns, OVERVIEW.items()):
        try:
            quote = market.get_quote(symbol)
        except MarketDataError:
            column.metric(name, "Unavailable")
            continue
        column.metric(name, f"{quote.price:,.2f}",
                      None if quote.change_pct is None else f"{quote.change_pct:+.2f}% today")
        column.caption(f"As of {quote.as_of:%b %d, %Y}" + (" · may be out of date" if quote.stale else ""))

    st.subheader("Price history")
    left, right = st.columns([2, 3])
    symbol = left.text_input("Ticker", value="SPY", max_chars=15, key="market_symbol").strip().upper()
    period = right.segmented_control("Period", list(PERIOD_DAYS), default="6mo", key="market_period") or "6mo"
    if not symbol:
        return
    try:
        quote = market.get_quote(symbol)
        history = market.get_history(symbol, period)
    except MarketDataError:
        st.warning(f"No price data found for {symbol}. Check the ticker and try again.")
        return
    start, end = history.closes.iloc[0], history.closes.iloc[-1]
    cols = st.columns(2)
    cols[0].metric(f"{symbol} latest price", money(quote.price),
                   None if quote.change_pct is None else f"{quote.change_pct:+.2f}% today")
    cols[1].metric(f"Change over the period shown", pct((end - start) / start * 100))
    st.plotly_chart(price_chart(history.closes, symbol), width="stretch", key="price_chart")
    first, last = history.closes.index[0], history.closes.index[-1]
    st.caption(f"Daily closes, {first:%b %d, %Y} to {last:%b %d, %Y}, from {history.source}. "
               + ("May be out of date. " if history.stale or quote.stale else "") + EDUCATION_NOTE)


# --- Goals ---
def render_goals(profile: dict) -> None:
    st.subheader("Plan a savings goal")
    risk_levels = ["conservative", "moderate", "aggressive"]
    with st.form("goal_form"):
        cols = st.columns(2)
        target = cols[0].number_input("Goal amount, in today's dollars", min_value=1.0, value=50_000.0, step=1_000.0)
        years = cols[1].number_input("Years until you need it", min_value=0.1, max_value=60.0, value=10.0, step=1.0)
        current = cols[0].number_input("Saved so far", min_value=0.0, value=5_000.0, step=500.0)
        monthly = cols[1].number_input("Monthly contribution", min_value=0.0, value=300.0, step=50.0)
        risk = st.selectbox("Risk level", risk_levels,
                            index=risk_levels.index(profile.get("risk_tolerance", "moderate")), key="goal_risk")
        submitted = st.form_submit_button("Project my goal")
    if submitted:
        try:
            goal = GoalInput(target, years, current, monthly, risk)
            st.session_state.goal_results = (risk, compare_risk_profiles(goal))
        except ValueError as error:
            st.error(f"Please check your inputs: {error}.")
            return
    if "goal_results" not in st.session_state:
        return
    chosen, results = st.session_state.goal_results
    show_goal(results[chosen], results)


def show_goal(p, results) -> None:
    cols = st.columns(4)
    cols[0].metric("Chance of reaching the goal", f"{p.success_probability_pct:.0f}%",
                   help="Share of 5,000 simulated market paths that reached the goal.")
    cols[1].metric("Middle outcome", money(p.final_p50, 0))
    cols[2].metric("Goal after inflation", money(p.inflated_target, 0))
    cols[3].metric("Monthly for an 8 in 10 chance", money(p.monthly_for_80pct_success, 0))
    st.plotly_chart(goal_fan_chart(p), width="stretch", key="goal_chart")

    st.markdown("**The same goal at each risk level**")
    st.dataframe(pd.DataFrame([{
        "Risk level": name, "Assumed yearly return": r.expected_return_pct,
        "Assumed volatility": r.volatility_pct, "Chance of reaching the goal": r.success_probability_pct,
        "Middle outcome": r.final_p50, "Monthly for an 8 in 10 chance": r.monthly_for_80pct_success,
    } for name, r in results.items()]), hide_index=True, width="stretch", column_config={
        "Assumed yearly return": st.column_config.NumberColumn(format="%.1f%%"),
        "Assumed volatility": st.column_config.NumberColumn(format="%.1f%%"),
        "Chance of reaching the goal": st.column_config.NumberColumn(format="%.0f%%"),
        "Middle outcome": st.column_config.NumberColumn(format="$%.0f"),
        "Monthly for an 8 in 10 chance": st.column_config.NumberColumn(format="$%.0f"),
    })
    st.caption("Returns and volatility are illustrative assumptions, not forecasts. Assumes 2.5% "
               "inflation and steady contributions, and ignores taxes and fees. " + EDUCATION_NOTE)
