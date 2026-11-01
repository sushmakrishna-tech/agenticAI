"""Goal Planning: projections come from simulation, shaped by the user's risk level."""
from pydantic import BaseModel

from src.agents.common import explain, extract, money, output
from src.utils.metrics import traced
from src.utils.goal_projections import GoalInput, compare_risk_profiles


class GoalRequest(BaseModel):
    target: float | None = None
    years: float | None = None
    current_savings: float | None = None
    monthly_contribution: float | None = None
    risk_profile: str | None = None


INSTRUCTION = ("Read the user's financial goal. Fill in only values they stated: the target amount in "
               "dollars, years until they need it, current savings, monthly contribution, and "
               "risk_profile (conservative, moderate, or aggressive). Use the latest values if they "
               "changed. Leave anything unstated empty. Never guess.")


def format_projection(goal: GoalInput, results: dict, assumed: list[str]) -> str:
    chosen = results[goal.risk_profile]
    lines = [
        f"Goal: {money(goal.target)} in today's dollars, in {goal.years:g} years, starting with "
        f"{money(goal.current_savings)} and adding {money(goal.monthly_contribution)} per month.",
        f"With assumed inflation, the target becomes {money(chosen.inflated_target)}.",
        f"Selected risk level: {goal.risk_profile}.",
    ]
    for name, r in results.items():
        lines.append(
            f"- {name} (assumes {r.expected_return_pct:.0f}% average return, {r.volatility_pct:.0f}% "
            f"volatility): {r.success_probability_pct:.0f}% of simulations reach the target; typical "
            f"ending balance {money(r.final_p50)}; weak-case {money(r.final_p10)}; strong-case "
            f"{money(r.final_p90)}; about {money(r.monthly_for_80pct_success)} per month would reach "
            f"the target in 80% of simulations."
        )
    lines.append("Limits: return assumptions are illustrative, not forecasts, and taxes and fees are ignored.")
    lines += [f"Assumed: {item}." for item in assumed]
    return "\n".join(lines)


def make_goals_agent(llm):
    def run(state):
        request = extract(llm, GoalRequest, INSTRUCTION, state) or GoalRequest()
        missing = [label for label, value in (("your target amount", request.target),
                                              ("how many years you have", request.years)) if not value]
        if missing:
            return output("goals", f"To project this goal, please tell me {' and '.join(missing)}. "
                                   "Your current savings and monthly contribution help too.")
        assumed = []
        if request.current_savings is None:
            assumed.append("current savings of $0 because none was given")
        if request.monthly_contribution is None:
            assumed.append("monthly contribution of $0 because none was given")
        profile = (state.get("profile") or {}).get("risk_tolerance")
        risk = request.risk_profile if request.risk_profile in {"conservative", "moderate", "aggressive"} else profile
        if not risk:
            risk = "moderate"
            assumed.append("moderate risk because no risk level was given")
        try:
            goal = GoalInput(request.target, request.years, request.current_savings or 0,
                             request.monthly_contribution or 0, risk)
        except ValueError as exc:
            return output("goals", f"I can't project that yet: {exc}.")
        with traced("goal_simulation", risk_profile=risk, years=goal.years):
            facts = format_projection(goal, compare_risk_profiles(goal), assumed)
        task = ("Explain the chance of reaching this goal at the selected risk level and how the other "
                "levels compare, including that more risk widens the outcomes. State the limits and any "
                "assumed values. Don't tell the user which level to choose.")
        return output("goals", explain(llm, task, facts, state), facts=facts)

    return run
