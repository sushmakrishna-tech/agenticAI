"""Plotly figures for the dashboards. Plain functions, so tests need no Streamlit.

Colors come from a colorblind-checked palette: slot 1 blue for the main series,
slot 2 orange for the comparison. Streamlit's theme supplies the background and
text colors, so the charts follow light and dark mode.
"""
import pandas as pd
import plotly.graph_objects as go

from src.utils.goal_projections import GoalProjection
from src.utils.portfolio_metrics import PortfolioAnalysis

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
BAND = "rgba(42, 120, 214, 0.28)"
LINE_WIDTH = 2


def _layout(fig: go.Figure, height: int, **extra) -> go.Figure:
    extra.setdefault("margin", dict(l=8, r=8, t=8, b=8))
    fig.update_layout(height=height, **extra)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=1)
    return fig


def allocation_chart(analysis: PortfolioAnalysis) -> go.Figure:
    """Share of the portfolio by holding, largest at the top."""
    positions = list(reversed(analysis.positions))  # Plotly draws the first bar at the bottom
    fig = go.Figure(go.Bar(
        x=[p.weight_pct for p in positions],
        y=[p.symbol for p in positions],
        orientation="h",
        marker=dict(color=BLUE, cornerradius=4),
        text=[f"{p.weight_pct:.1f}%" for p in positions],
        textposition="outside",
        customdata=[f"${p.value:,.2f}" for p in positions],
        hovertemplate="%{y}: %{x:.1f}% (%{customdata})<extra></extra>",
    ))
    fig.update_xaxes(ticksuffix="%", range=[0, max(p.weight_pct for p in positions) * 1.2])
    return _layout(fig, height=80 + 36 * len(positions), bargap=0.35)


def growth_chart(growth: pd.DataFrame) -> go.Figure:
    """$100 in today's mix beside $100 in the benchmark, on one shared axis."""
    fig = go.Figure()
    for column, color in zip(growth.columns, (BLUE, ORANGE)):
        fig.add_trace(go.Scatter(
            x=growth.index, y=growth[column], name=column, mode="lines",
            line=dict(color=color, width=LINE_WIDTH),
            hovertemplate=f"{column}: $%{{y:,.2f}}<extra></extra>",
        ))
        # Direct label at the line's end, so identity doesn't rely on color alone.
        fig.add_annotation(x=growth.index[-1], y=growth[column].iloc[-1], text=column,
                           showarrow=False, xanchor="left", xshift=6)
    fig.update_yaxes(tickprefix="$")
    return _layout(fig, height=340, hovermode="x unified", margin=dict(l=8, r=90, t=8, b=8),
                   legend=dict(orientation="h", y=1.08, x=0))


def price_chart(closes: pd.Series, symbol: str) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=closes.index, y=closes.values, mode="lines", name=symbol,
        line=dict(color=BLUE, width=LINE_WIDTH),
        hovertemplate="%{x|%b %d, %Y}: %{y:,.2f}<extra></extra>",
    ))
    return _layout(fig, height=320, hovermode="x", showlegend=False)


def goal_fan_chart(projection: GoalProjection) -> go.Figure:
    """The middle outcome and the range 8 in 10 simulated paths fell within."""
    years = projection.path_years
    fig = go.Figure([
        go.Scatter(x=years, y=projection.path_p90, mode="lines", line=dict(width=0),
                   hoverinfo="skip", showlegend=False),
        go.Scatter(x=years, y=projection.path_p10, mode="lines", line=dict(width=0),
                   fill="tonexty", fillcolor=BAND, name="8 in 10 outcomes fell in this range",
                   hoverinfo="skip"),
        go.Scatter(x=years, y=projection.path_p50, mode="lines", name="Middle outcome",
                   line=dict(color=BLUE, width=LINE_WIDTH),
                   customdata=list(zip(projection.path_p10, projection.path_p90)),
                   hovertemplate=("Year %{x:.1f}<br>Middle: $%{y:,.0f}<br>"
                                  "Range: $%{customdata[0]:,.0f} to $%{customdata[1]:,.0f}<extra></extra>")),
    ])
    fig.add_hline(y=projection.inflated_target, line=dict(color=GRAY, width=1.5, dash="dash"),
                  annotation_text=f"Goal: ${projection.inflated_target:,.0f}",
                  annotation_position="top left")
    fig.update_xaxes(title="Years from now")
    fig.update_yaxes(tickprefix="$")
    return _layout(fig, height=360, legend=dict(orientation="h", y=1.1, x=0))
