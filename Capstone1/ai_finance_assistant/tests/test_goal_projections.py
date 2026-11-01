import numpy as np
import pytest

from src.utils import goal_projections as gp
from src.utils.goal_projections import (
    GoalInput, compare_risk_profiles, future_value, project_goal, required_monthly,
)


def test_future_value_zero_return():
    assert future_value(1000, 100, 0, 12) == 2200


def test_future_value_lump_sum_by_hand():
    assert future_value(1000, 0, 0.06, 120) == pytest.approx(1000 * 1.06**10)


def test_future_value_monthly_contributions_by_hand():
    rate = 1.06 ** (1 / 12) - 1
    expected = 100 * ((1 + rate) ** 12 - 1) / rate
    assert future_value(0, 100, 0.06, 12) == pytest.approx(expected)


def test_required_monthly_reverses_future_value():
    monthly = required_monthly(100_000, 5_000, 0.06, 120)
    assert future_value(5_000, monthly, 0.06, 120) == pytest.approx(100_000)


def test_required_monthly_zero_when_already_on_track():
    assert required_monthly(1_000, 5_000, 0.06, 120) == 0


def test_simulated_mean_matches_the_math():
    rng = np.random.default_rng(1)
    factors = gp._monthly_returns(rng, 200_000, 120, 0.06, 0.15)
    assert factors.prod(axis=1).mean() == pytest.approx(1.06**10, rel=0.01)


def test_simulated_volatility_matches_assumption():
    rng = np.random.default_rng(1)
    factors = gp._monthly_returns(rng, 200_000, 12, 0.06, 0.15)
    assert factors.prod(axis=1).std() == pytest.approx(0.15, rel=0.02)


def test_same_inputs_same_answer():
    goal = GoalInput(50_000, 10, 1_000, 200)
    assert project_goal(goal) == project_goal(goal)


def test_target_grows_with_inflation():
    assert project_goal(GoalInput(100_000, 10)).inflated_target == pytest.approx(100_000 * 1.025**10)


def test_contributing_more_never_lowers_success():
    low = project_goal(GoalInput(100_000, 10, 0, 300)).success_probability_pct
    high = project_goal(GoalInput(100_000, 10, 0, 900)).success_probability_pct
    assert high > low


def test_monthly_for_80pct_really_gives_80pct():
    first = project_goal(GoalInput(100_000, 15, 5_000, 0))
    check = project_goal(GoalInput(100_000, 15, 5_000, first.monthly_for_80pct_success))
    assert check.success_probability_pct == pytest.approx(80, abs=0.1)


def test_80pct_contribution_is_above_straight_line_amount():
    result = project_goal(GoalInput(100_000, 15, 5_000, 0))
    assert result.monthly_for_80pct_success > result.required_monthly_straight_line


def test_goal_already_funded():
    result = project_goal(GoalInput(1_000, 10, 100_000, 0, "conservative"))
    assert result.success_probability_pct == 100
    assert result.monthly_for_80pct_success == 0
    assert result.required_monthly_straight_line == 0


def test_no_money_in_means_no_chance_and_no_crash():
    result = project_goal(GoalInput(10_000, 5))
    assert result.success_probability_pct == 0
    assert result.final_p50 == 0
    assert result.monthly_for_80pct_success > 0


def test_percentiles_are_ordered():
    r = project_goal(GoalInput(100_000, 20, 0, 300))
    assert r.final_p10 < r.final_p50 < r.final_p90
    assert all(a <= b <= c for a, b, c in zip(r.path_p10, r.path_p50, r.path_p90))


def test_paths_run_from_today_to_the_final_year():
    r = project_goal(GoalInput(100_000, 3, 500, 100))
    assert r.path_years == [0, 1, 2, 3]
    assert r.path_p50[0] == 500
    assert r.path_p50[-1] == pytest.approx(r.final_p50)


def test_partial_year_horizon_ends_exactly_on_time():
    r = project_goal(GoalInput(10_000, 1.5, 0, 100))
    assert r.path_years == [0, 1, 1.5]


def test_one_month_horizon_works():
    assert project_goal(GoalInput(10_000, 1 / 12, 0, 100)).path_years == [0, pytest.approx(1 / 12)]


def test_aggressive_has_the_widest_outcomes():
    results = compare_risk_profiles(GoalInput(100_000, 20, 0, 300))
    assert set(results) == {"conservative", "moderate", "aggressive"}
    spread = {k: r.final_p90 / r.final_p10 for k, r in results.items()}
    assert spread["conservative"] < spread["moderate"] < spread["aggressive"]


def test_short_goal_shows_aggressive_can_end_lower_in_bad_cases():
    results = compare_risk_profiles(GoalInput(20_000, 2, 15_000, 200))
    assert results["aggressive"].final_p10 < results["conservative"].final_p10


def test_unknown_risk_profile_rejected():
    with pytest.raises(ValueError):
        project_goal(GoalInput(1000, 5, risk_profile="yolo"))


@pytest.mark.parametrize("kwargs", [
    {"target": 0, "years": 5},
    {"target": 100, "years": 0},
    {"target": 100, "years": 61},
    {"target": 100, "years": 5, "current_savings": -1},
    {"target": 100, "years": 5, "monthly_contribution": -1},
    {"target": float("inf"), "years": 5},
])
def test_bad_goals_rejected(kwargs):
    with pytest.raises(ValueError):
        GoalInput(**kwargs)
