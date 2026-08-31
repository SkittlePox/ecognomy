"""The annealed-commodity-money experiment.

Three lines that answer the headline question together: the token's intrinsic
consumption weight (what the anneal is removing), the share of trades it appears
in (whether it is being accepted), and holdings of goods their owner does not
value (whether anything is being held for exchange value at all).

If the trade share holds up while the weight goes to zero, acceptance is being
sustained by expectation alone.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html

from ecognomy.viewer.panel import Panel
from ecognomy.viewer.theme import P, figure, line, series_color


def build(data):
    t = data.ticks
    weight = data["token_weight"]
    share = data["token_trade_share"]
    annealing = float(weight.max() - weight.min()) > 1e-6

    # Weight and share share a 0-1 scale honestly (both are fractions), so they
    # belong on one axis. Exchange-value holdings are units, so they get their
    # own chart rather than a second y-axis.
    fig = go.Figure()
    line(fig, t, weight, "intrinsic weight", series_color(0))
    line(fig, t, share, "share of trades", series_color(1))
    figure(fig, ylabel="fraction", legend=True, height=260)

    fig2 = go.Figure()
    line(fig2, t, data["exchange_value_holdings"], "held for exchange", series_color(2))
    figure(fig2, ylabel="units", height=200)

    note = ("No anneal configured in this run — the token is an ordinary good throughout. "
            "Run with --anneal-ticks to test whether acceptance survives losing intrinsic value.")
    return html.Div([
        html.Div(note, style={"fontSize": "12px", "color": P["text_muted"], "marginBottom": "10px"})
        if not annealing else html.Div(),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        html.Div("Goods held by agents whose preference weight for them is ~0. The shared "
                 "precursor to arbitrage and to money acceptance — they are the same behaviour.",
                 style={"fontSize": "11px", "color": P["text_muted"], "margin": "6px 0 2px"}),
        dcc.Graph(figure=fig2, config={"displayModeBar": False}),
    ])


PANEL = Panel(
    id="token",
    title=f"Token: intrinsic value vs acceptance",
    blurb="The headline experiment. Acceptance surviving the anneal means the token is held "
          "up by mutual expectation and nothing else.",
    build=build,
    order=30,
    requires=("token_weight",),
)
