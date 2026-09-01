"""One agent, over time.

The world panel is a snapshot: it shows everything about a single tick. This is
the complement -- one agent's trajectory across the whole run.

The series that matters most is the **implied value** of each good. There is no
published price in this world; each agent posts its own rate matrix and trades
are struck between them, so price formation *is* the convergence of these lines.
That claim is only inspectable per agent, and this is where you inspect it.

Below it sits the part a value vector cannot hold: the **round trip**, which is
what the agent's two postings on a pair say about converting a unit out and
straight back. Above 1 is a spread, the margin it demands in both directions;
below 1 is a posting that can be cycled against it.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from ecognomy.metrics import implied_value, round_trip
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

        html.Div("Implied value — the single valuation its rate matrix is nearest to, "
                 "relative to its average good. Under a linear reward and honest posting "
                 "these are the agent's fixed preference weights, so the lines are flat by "
                 "construction; movement here would mean a policy that shades its rates "
                 "rather than posting them honestly.",
                 style={"fontSize": "11px", "color": P["text_muted"], "margin": "0 0 4px"}),
        dcc.Graph(id=f"{_ID}-price", config={"displayModeBar": False}),

        html.Div("Round trip — convert a unit of one good into another and straight back at "
                 "this agent's own posted rates, and this is what it keeps the reciprocal of. "
                 "Above 1 it is quoting a spread; below 1 its own bid crosses its own ask and "
                 "a counterparty can cycle goods through it. Flat at 1 is honest posting.",
                 style={"fontSize": "11px", "color": P["text_muted"], "margin": "12px 0 4px"}),
        dcc.Graph(id=f"{_ID}-spread", config={"displayModeBar": False}),

        html.Div("Inventory held", style={"fontSize": "11px", "color": P["text_muted"],
                                          "margin": "12px 0 4px"}),
        dcc.Graph(id=f"{_ID}-inventory", config={"displayModeBar": False}),

        html.Div("Cumulative welfare", style={"fontSize": "11px", "color": P["text_muted"],
                                              "margin": "12px 0 4px"}),
        dcc.Graph(id=f"{_ID}-reward", config={"displayModeBar": False}),
    ])


def _relative(ask_series: np.ndarray) -> np.ndarray:
    """(T, G) the value level the agent's matrix implies, per tick.

    A rate matrix carries only ratios, so this is normalised to a geometric mean
    of 1 -- which keeps every good on the same footing rather than privileging a
    numeraire, and puts the series on the same scale the region panel uses.
    """
    return implied_value(ask_series.astype(np.float64))


def _round_trips(ask_series: np.ndarray):
    """(T,) median and worst round trip across the pairs the agent will cycle.

    Pairs it refuses in either direction cannot be cycled at all and are left
    out; a tick where it refuses everything has no round trip to report.
    """
    trips = round_trip(ask_series.astype(np.float64))
    t = trips.shape[0]
    med, worst = np.full(t, np.nan), np.full(t, np.nan)
    for k in range(t):
        finite = trips[k][np.isfinite(trips[k])]
        if finite.size:
            med[k], worst[k] = np.median(finite), finite.min()
    return med, worst


def register(app, data):
    @app.callback(
        Output(f"{_ID}-summary", "children"),
        Output(f"{_ID}-price", "figure"),
        Output(f"{_ID}-spread", "figure"),
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

        rel = _relative(data["snap_ask"][:, i, :, :])  # constant under honest posting
        fig = go.Figure()
        for g, name in enumerate(data.goods):
            line(fig, t, rel[:, g], name, series_color(g))
        fig.add_hline(y=1.0, line=dict(color=P["axis"], width=1, dash="dot"))
        figure(fig, ylabel="value ÷ own mean", legend=True, height=250, margin_right=95)

        med, worst = _round_trips(data["snap_ask"][:, i, :, :])
        figs = go.Figure()
        line(figs, t, med, "median pair", series_color(0))
        line(figs, t, worst, "worst pair", series_color(1))
        figs.add_hline(y=1.0, line=dict(color=P["axis"], width=1, dash="dot"),
                       annotation_text="no spread", annotation_position="top left",
                       annotation_font=dict(size=10, color=P["text_muted"]))
        figure(figs, ylabel="kept per round trip⁻¹", legend=True, height=210, margin_right=95)

        inv = data["snap_inventory"][:, i, :]
        fig2 = go.Figure()
        for g, name in enumerate(data.goods):
            line(fig2, t, inv[:, g], name, series_color(g))
        figure(fig2, ylabel="units held", legend=True, height=230, margin_right=95)

        cum = np.cumsum(data["snap_reward"][:, i].astype(np.float64))
        fig3 = go.Figure()
        line(fig3, t, cum, "welfare", series_color(0), label=False)
        figure(fig3, ylabel="cumulative", height=190, margin_right=24)

        return summary, fig, figs, fig2, fig3


PANEL = Panel(
    id=_ID,
    title="Agent over time",
    blurb="One agent's trajectory: the prices it posts, what it accumulates, and what it "
          "earns. Price formation is the convergence of these lines across agents.",
    build=build,
    register=register,
    order=60,
    requires=("snap_inventory", "snap_ask"),
)
