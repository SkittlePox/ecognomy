"""Which good pairs carry the volume.

Money emerging shows up here as structure, not just as a rate: if the token's
row and column stay thick while ordinary pairs go dead, the token has become the
routing hub for indirect exchange. That is money measured as centrality in the
trade graph.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html

from ecognomy.viewer.panel import Panel
from ecognomy.viewer.theme import P, figure, sequential_scale


def build(data):
    volume = data["volume_by_pair"].sum(axis=0)  # (G, G) over the whole run
    goods = data.goods
    token = data.token_good

    # Magnitude -> one-hue sequential ramp, light to dark. Never a rainbow.
    fig = go.Figure(go.Heatmap(
        z=volume, x=goods, y=goods,
        colorscale=sequential_scale(), zmin=0,
        xgap=2, ygap=2,  # surface gap between cells
        colorbar=dict(title=dict(text="units", side="right", font=dict(size=11)),
                      thickness=12, outlinewidth=0, tickfont=dict(size=10)),
        hovertemplate="%{y} given for %{x}<br>%{z:.2f} units<extra></extra>",
    ))
    fig.update_layout(
        height=340,
        margin=dict(l=80, r=20, t=14, b=50),
        paper_bgcolor=P["panel"], plot_bgcolor=P["panel"],
        font=dict(size=12, color=P["text_secondary"]),
    )
    fig.update_xaxes(title_text="received", side="bottom",
                     title_font=dict(size=11, color=P["text_muted"]))
    fig.update_yaxes(title_text="given", autorange="reversed",
                     title_font=dict(size=11, color=P["text_muted"]))

    total = volume.sum()
    tok_share = (volume[token].sum() + volume[:, token].sum() - volume[token, token]) / total if total else 0.0

    return html.Div([
        html.Div(
            f"Token ({goods[token]}) is on one side of {tok_share:.0%} of all volume."
            if total else "No volume recorded — nothing to route.",
            style={"fontSize": "12px", "color": P["text_secondary"], "marginBottom": "8px"},
        ),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
    ])


PANEL = Panel(
    id="trade-matrix",
    title="Volume by good pair",
    blurb="A thick token row and column beside dead ordinary pairs is indirect exchange.",
    build=build,
    order=40,
    requires=("volume_by_pair",),
)
