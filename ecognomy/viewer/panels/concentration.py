"""Concentration of holdings -- is anyone cornering a market?

Herfindahl index over each good's holdings: 1/N when evenly spread, 1.0 when one
agent holds everything. Low-spoilage goods are the ones this can happen to, since
durability is what makes a corner sustainable.
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dcc, html

from ecognomy.viewer.panel import Panel
from ecognomy.viewer.theme import P, figure, line, series_color


def build(data):
    t = data.ticks
    h = data["herfindahl"]  # (T, G)
    even = 1.0 / data.n_agents

    fig = go.Figure()
    for g, name in enumerate(data.goods):
        line(fig, t, h[:, g], name, series_color(g))
    fig.add_hline(y=even, line=dict(color=P["axis"], width=1, dash="dot"),
                  annotation_text="evenly held", annotation_position="top left",
                  annotation_font=dict(size=10, color=P["text_muted"]))
    figure(fig, ylabel="Herfindahl", legend=True, height=280, margin_right=90)

    return html.Div([
        html.Div(f"1/N = {even:.3f} means evenly spread; 1.0 means one agent holds the lot.",
                 style={"fontSize": "11px", "color": P["text_muted"], "marginBottom": "6px"}),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
    ])


PANEL = Panel(
    id="concentration",
    title="Concentration of holdings",
    blurb="Rising lines mean a good is being cornered — which low spoilage permits.",
    build=build,
    order=50,
    requires=("herfindahl",),
)
