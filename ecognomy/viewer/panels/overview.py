"""Participation and trade volume -- is anything happening at all?

The first question about any run. A dead world should be unmistakable here
before you look anywhere else.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dcc, html

from ecognomy.viewer.panel import Panel
from ecognomy.viewer.theme import P, figure, line, series_color


def _stat(label: str, value: str, tone: str | None = None) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"fontSize": "11px", "color": P["text_muted"],
                                   "textTransform": "uppercase", "letterSpacing": "0.04em"}),
            html.Div(value, style={"fontSize": "26px", "fontWeight": 600,
                                   "color": tone or P["text_primary"], "lineHeight": "1.15"}),
        ],
        style={"padding": "12px 18px 14px", "background": P["panel"],
               "border": f"1px solid {P['grid']}", "borderRadius": "6px", "minWidth": "132px"},
    )


def build(data):
    t = data.ticks
    trades = data["n_trades"]
    total = float(trades.sum())
    dead = total == 0

    # Participation is a fraction and volume is a unit count. They do not share
    # a scale, and forcing them onto one axis flattens participation into a
    # straight line. Two charts, one measure each -- so neither needs a legend.
    fig = go.Figure()
    line(fig, t, data["fraction_producing"], "producing", series_color(0), label=False)
    figure(fig, ylabel="fraction producing", height=190, margin_right=24)
    fig.update_yaxes(rangemode="tozero")

    fig2 = go.Figure()
    line(fig2, t, data["trade_volume"], "trade volume", series_color(1), label=False)
    figure(fig2, ylabel="trade volume (units)", height=190, margin_right=24)
    fig2.update_yaxes(rangemode="tozero")

    return html.Div([
        html.Div([
            _stat("total trades", f"{total:,.0f}",
                  P["critical"] if dead else None),
            _stat("trade volume", f"{data['trade_volume'].sum():,.1f}"),
            _stat("mean producing", f"{data['fraction_producing'].mean():.0%}"),
            _stat("mean reward", f"{data['reward_mean'].mean():.4f}"),
            _stat("illegal actions", f"{data['n_illegal'].sum():,.0f}"),
        ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "14px"}),
        html.Div(
            "No trades executed in this run — the world is dead. That is a valid outcome, "
            "not necessarily a bug.",
            style={"color": P["critical"], "fontSize": "12px", "marginBottom": "10px"},
        ) if dead else html.Div(),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        dcc.Graph(figure=fig2, config={"displayModeBar": False}),
    ])


PANEL = Panel(
    id="overview",
    title="Participation and trade",
    blurb="Is anything happening at all? A flat line here means a dead world.",
    build=build,
    order=10,
)
