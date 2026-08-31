"""One agent, over time.

The world panel is a snapshot: it shows everything about a single tick. This is
the complement -- one agent's trajectory across the whole run.

The series that matters most is **posted price**. There is no published price in
this world; each agent posts its own valuation and trades are struck between
them, so price formation *is* the convergence of these lines. That claim is only
inspectable per agent, and this is where you inspect it.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from ecognomy.viewer.panel import Panel
from ecognomy.viewer.theme import P, figure, line, series_color

_ID = "agent"
_MONO = "ui-monospace, SFMono-Regular, Menlo, monospace"


def build(data):
    return html.Div([
        html.Div([
            html.Label("agent", style={"fontSize": "11px", "color": P["text_muted"],
                                       "marginRight": "8px"}),
            dcc.Dropdown(
                id=f"{_ID}-pick",
                options=[{"label": f"agent {i}", "value": i} for i in range(data.n_agents)],
                value=0, clearable=False, style={"width": "160px", "fontSize": "12px"},
            ),
            html.Div(id=f"{_ID}-summary", style={"fontSize": "12px", "fontFamily": _MONO,
                                                 "color": P["text_secondary"],
                                                 "marginLeft": "16px"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "14px"}),

        html.Div("Posted price — its own valuation of each good, relative to its average. "
                 "Under a linear reward these are the agent's fixed preference weights, so "
                 "the lines are flat by construction; movement here would mean a policy that "
                 "shades its prices rather than posting them honestly.",
                 style={"fontSize": "11px", "color": P["text_muted"], "margin": "0 0 4px"}),
        dcc.Graph(id=f"{_ID}-price", config={"displayModeBar": False}),

        html.Div("Inventory held", style={"fontSize": "11px", "color": P["text_muted"],
                                          "margin": "12px 0 4px"}),
        dcc.Graph(id=f"{_ID}-inventory", config={"displayModeBar": False}),

        html.Div("Cumulative welfare", style={"fontSize": "11px", "color": P["text_muted"],
                                              "margin": "12px 0 4px"}),
        dcc.Graph(id=f"{_ID}-reward", config={"displayModeBar": False}),
    ])


def _relative(price_series: np.ndarray) -> np.ndarray:
    """(T, G) price divided by its own geometric mean, per tick.

    A price vector is meaningful only up to scale, so raw levels are not
    comparable across agents or across ticks. Dividing by the geometric mean
    leaves the relative prices, which are the part that carries information, and
    keeps every good on the same footing rather than privileging a numeraire.
    """
    p = np.maximum(price_series.astype(np.float64), 1e-12)
    gm = np.exp(np.log(p).mean(axis=1, keepdims=True))
    return p / gm


def register(app, data):
    @app.callback(
        Output(f"{_ID}-summary", "children"),
        Output(f"{_ID}-price", "figure"),
        Output(f"{_ID}-inventory", "figure"),
        Output(f"{_ID}-reward", "figure"),
        Input(f"{_ID}-pick", "value"),
    )
    def _update(i):
        i = int(i)
        t = data["snapshot_ticks"]

        wants = data["theta"][i]
        makes = data["efficiency"][i]
        summary = (f"wants {', '.join(f'{data.goods[g]} {wants[g]:.2f}' for g in np.argsort(-wants)[:3])}"
                   f"   ·   makes {', '.join(f'{data.goods[g]} {makes[g]:.2f}' for g in np.flatnonzero(makes)) or 'nothing'}"
                   f"   ·   sight {int(data['sight'][i])}"
                   f"   ·   mobility {data['mobility'][i]:.2f}")

        rel = _relative(data["snap_price"][:, i, :])  # constant under linear reward
        fig = go.Figure()
        for g, name in enumerate(data.goods):
            line(fig, t, rel[:, g], name, series_color(g))
        fig.add_hline(y=1.0, line=dict(color=P["axis"], width=1, dash="dot"))
        figure(fig, ylabel="price ÷ own mean", legend=True, height=250, margin_right=95)

        inv = data["snap_inventory"][:, i, :]
        fig2 = go.Figure()
        for g, name in enumerate(data.goods):
            line(fig2, t, inv[:, g], name, series_color(g))
        figure(fig2, ylabel="units held", legend=True, height=230, margin_right=95)

        cum = np.cumsum(data["snap_reward"][:, i].astype(np.float64))
        fig3 = go.Figure()
        line(fig3, t, cum, "welfare", series_color(0), label=False)
        figure(fig3, ylabel="cumulative", height=190, margin_right=24)

        return summary, fig, fig2, fig3


PANEL = Panel(
    id=_ID,
    title="Agent over time",
    blurb="One agent's trajectory: the prices it posts, what it accumulates, and what it "
          "earns. Price formation is the convergence of these lines across agents.",
    build=build,
    register=register,
    order=60,
    requires=("snap_inventory", "snap_price"),
)
