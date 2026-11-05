import pandas as pd
import pytest

from src.data.accounts import AccountStore
from src.utils.goal_projections import GoalInput, project_goal
from src.utils.portfolio_metrics import Holding, analyze_portfolio, growth_index
from src.web_app.charts import BLUE, ORANGE, allocation_chart, goal_fan_chart, growth_chart, price_chart
from src.web_app.dashboards import parse_holdings
from tests.fakes import FakeMarket, history


def analysis():
    return analyze_portfolio([Holding("AAPL", 10, 150), Holding("VOO", 6)], FakeMarket())


def test_allocation_bars_put_the_largest_on_top_with_labels():
    fig = allocation_chart(analysis())
    bar = fig.data[0]
    assert list(bar.y) == ["AAPL", "VOO"]  # drawn bottom-up, so VOO (largest) is on top
    assert bar.text[-1] == "60.0%" and bar.marker.color == BLUE


def test_growth_index_starts_both_lines_at_100():
    closes = {"AAPL": history("AAPL"), "VOO": history("VOO")}
    growth = growth_index(closes, {"AAPL": 50, "VOO": 50}, history("^GSPC"))
    assert list(growth.columns) == ["Your mix", "S&P 500"]
    assert growth.iloc[0].tolist() == [100, 100]


def test_growth_index_needs_enough_shared_history():
    short = history("AAPL")[:10]
    assert growth_index({"AAPL": short}, {"AAPL": 100}, None) is None
    assert growth_index({}, {}, None) is None


def test_analysis_includes_growth_for_the_chart():
    growth = analysis().growth
    fig = growth_chart(growth)
    assert [t.name for t in fig.data] == ["Your mix", "S&P 500"]
    assert [t.line.color for t in fig.data] == [BLUE, ORANGE]
    assert {a.text for a in fig.layout.annotations} == {"Your mix", "S&P 500"}  # direct labels


def test_price_chart_is_one_series():
    fig = price_chart(history("SPY"), "SPY")
    assert len(fig.data) == 1 and not fig.layout.showlegend


def test_fan_chart_shows_range_middle_and_goal():
    projection = project_goal(GoalInput(50_000, 10, 5_000, 300, "moderate"))
    fig = goal_fan_chart(projection)
    assert list(fig.data[2].y) == projection.path_p50
    assert fig.data[1].fill == "tonexty"
    assert fig.layout.shapes[0].y0 == projection.inflated_target


def test_parse_holdings_skips_blank_rows_and_normalizes():
    frame = pd.DataFrame([
        {"symbol": " aapl ", "shares": 2, "cost_per_share": float("nan")},
        {"symbol": None, "shares": None, "cost_per_share": None},
        {"symbol": "VOO", "shares": 1.5, "cost_per_share": 400},
    ])
    assert parse_holdings(frame) == [
        {"symbol": "AAPL", "shares": 2.0, "cost_per_share": None},
        {"symbol": "VOO", "shares": 1.5, "cost_per_share": 400.0},
    ]


@pytest.mark.parametrize("row", [
    {"symbol": "AAPL", "shares": None, "cost_per_share": None},
    {"symbol": "AAPL", "shares": 0, "cost_per_share": None},
    {"symbol": "AAPL", "shares": 1, "cost_per_share": -5},
])
def test_parse_holdings_rejects_bad_rows(row):
    with pytest.raises(ValueError):
        parse_holdings(pd.DataFrame([row]))


def test_holdings_are_saved_per_user(tmp_path):
    store = AccountStore(tmp_path / "h.db")
    a, b = store.create_user("a@example.com", "password1"), store.create_user("b@example.com", "password1")
    store.save_holdings(a["id"], [{"symbol": "AAPL", "shares": 1, "cost_per_share": None}])
    store.save_holdings(a["id"], [{"symbol": "VOO", "shares": 2, "cost_per_share": None}])
    assert store.get_holdings(a["id"]) == [{"symbol": "VOO", "shares": 2, "cost_per_share": None}]
    assert store.get_holdings(b["id"]) == []
