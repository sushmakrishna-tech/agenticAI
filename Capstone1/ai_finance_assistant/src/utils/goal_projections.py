"""Goal projections: a straight-line estimate plus a Monte Carlo simulation.

Monte Carlo runs thousands of possible market histories, because a single
"8% a year" hides the risk of bad years. It answers "how often does this plan
reach the goal?" Results depend on the assumptions in config.yaml, which are
illustrative, and they ignore taxes and fees.
"""
from dataclasses import dataclass

import numpy as np

from src.core.config import get_config

MAX_YEARS = 60
TARGET_SUCCESS = 0.80


@dataclass(frozen=True)
class GoalInput:
    target: float  # in today's dollars
    years: float
    current_savings: float = 0.0
    monthly_contribution: float = 0.0
    risk_profile: str = "moderate"

    def __post_init__(self):
        values = (self.target, self.years, self.current_savings, self.monthly_contribution)
        if not all(np.isfinite(v) for v in values):
            raise ValueError("Goal amounts must be real numbers")
        if self.target <= 0:
            raise ValueError("The target must be greater than 0")
        if not 1 / 12 <= self.years <= MAX_YEARS:
            raise ValueError(f"Years must be between one month and {MAX_YEARS} years")
        if self.current_savings < 0 or self.monthly_contribution < 0:
            raise ValueError("Savings and contributions cannot be negative")


@dataclass(frozen=True)
class GoalProjection:
    risk_profile: str
    expected_return_pct: float
    volatility_pct: float
    inflated_target: float  # the target in future dollars
    straight_line_value: float  # the result if every year earned the expected return
    required_monthly_straight_line: float
    success_probability_pct: float
    monthly_for_80pct_success: float
    final_p10: float
    final_p50: float
    final_p90: float
    # One value per year including year 0, for fan charts.
    path_years: list[float]
    path_p10: list[float]
    path_p50: list[float]
    path_p90: list[float]


def future_value(current: float, monthly: float, annual_return: float, months: int) -> float:
    """Contributions arrive at the end of each month."""
    rate = (1 + annual_return) ** (1 / 12) - 1
    if abs(rate) < 1e-12:
        return current + monthly * months
    growth = (1 + rate) ** months
    return current * growth + monthly * (growth - 1) / rate


def required_monthly(target: float, current: float, annual_return: float, months: int) -> float:
    """The monthly amount that reaches the target at exactly the expected return."""
    shortfall = target - future_value(current, 0, annual_return, months)
    if shortfall <= 0:
        return 0.0
    return shortfall / future_value(0, 1, annual_return, months)


def _monthly_returns(rng, runs: int, months: int, expected: float, volatility: float):
    """Random monthly growth factors whose average compounds to `expected` per year.

    Uses a lognormal distribution, so returns can never fall below -100%.
    """
    sigma2 = np.log(1 + volatility**2 / (1 + expected) ** 2)
    mu = np.log(1 + expected) - sigma2 / 2
    return np.exp(rng.normal(mu / 12, np.sqrt(sigma2 / 12), size=(runs, months)))


def project_goal(goal: GoalInput, risk_profile: str | None = None) -> GoalProjection:
    cfg = get_config()["goals"]
    profile = risk_profile or goal.risk_profile
    if profile not in cfg["risk_profiles"]:
        raise ValueError(f"Risk profile must be one of {sorted(cfg['risk_profiles'])}")
    expected = cfg["risk_profiles"][profile]["expected_return"]
    volatility = cfg["risk_profiles"][profile]["volatility"]

    months = max(1, round(goal.years * 12))
    target = goal.target * (1 + cfg["inflation"]) ** (months / 12)
    runs = cfg["simulations"]
    rng = np.random.default_rng(cfg["random_seed"])
    factors = _monthly_returns(rng, runs, months, expected, volatility)

    # Track savings and "value of $1/month" apart: final = savings + contribution * unit.
    # That lets us solve for the contribution without re-running the simulation.
    savings = np.cumprod(factors, axis=1) * goal.current_savings
    unit = np.zeros((runs, months))
    balance = np.zeros(runs)
    for month in range(months):
        balance = balance * factors[:, month] + 1
        unit[:, month] = balance
    paths = savings + unit * goal.monthly_contribution
    final = paths[:, -1]

    # In run i, reaching the target needs (target - savings_i) / unit_i per month.
    # The 80th percentile of that amount succeeds in 80% of runs.
    needed = (target - savings[:, -1]) / unit[:, -1]
    monthly_80 = max(0.0, float(np.quantile(needed, TARGET_SUCCESS)))

    checkpoints = sorted({*range(12, months + 1, 12), months})
    columns = [c - 1 for c in checkpoints]
    p10, p50, p90 = np.percentile(paths[:, columns], [10, 50, 90], axis=0)
    start = goal.current_savings

    return GoalProjection(
        risk_profile=profile,
        expected_return_pct=expected * 100,
        volatility_pct=volatility * 100,
        inflated_target=target,
        straight_line_value=future_value(goal.current_savings, goal.monthly_contribution, expected, months),
        required_monthly_straight_line=required_monthly(target, goal.current_savings, expected, months),
        success_probability_pct=float((final >= target).mean() * 100),
        monthly_for_80pct_success=monthly_80,
        final_p10=float(np.percentile(final, 10)),
        final_p50=float(np.percentile(final, 50)),
        final_p90=float(np.percentile(final, 90)),
        path_years=[0.0, *(c / 12 for c in checkpoints)],
        path_p10=[start, *map(float, p10)],
        path_p50=[start, *map(float, p50)],
        path_p90=[start, *map(float, p90)],
    )


def compare_risk_profiles(goal: GoalInput) -> dict[str, GoalProjection]:
    """The same goal under each risk level, so the tradeoff is visible."""
    return {name: project_goal(goal, name) for name in get_config()["goals"]["risk_profiles"]}
