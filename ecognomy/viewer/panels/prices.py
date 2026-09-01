"""Prices by region.

Built on **posted** rates rather than realised trade ratios. Every agent posts a
valuation for every good every tick, so the posted board is dense, while executed
trades are sparse and give a ragged signal with long gaps.

Agents post a G x G matrix of rates, which has no levels in it at all -- only
ratios. What is plotted is the **implied value** of each good: the single
valuation vector the matrix is nearest to, normalised to a geometric mean of 1,
so every good is treated alike and none is privileged as a numeraire.

Two things that summary deliberately throws away, both visible in the agent
inspector instead: the spread an agent quotes around its own level, and any
incoherence between its rates, which is precisely the part no value vector can
represent.

There is deliberately no good-pair selector. A pair list carries each rate twice
-- apple/banana and banana/apple are the same fact inverted -- and G(G-1) entries
for what G implied values already say.
"""

from __future__ import annotations

import numpy as np

from ecognomy.metrics import implied_value
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from ecognomy.viewer.panel import Panel
from ecognomy.viewer.theme import P, figure, line, series_color

_ID = "prices"


def _relative(ask: np.ndarray) -> np.ndarray:
    """(T, N, G) the value level each agent's rate matrix implies, per good.

    A matrix has no levels, only ratios, so this is the rank-1 fit: normalised
    to a geometric mean of 1, which is the same scale the price-vector version
    of this panel used. What it cannot show is the spread an agent quotes around
    that level -- for that, see the round trip in the agent inspector.
    """
    return implied_value(ask.astype(np.float64))


def _by_region(rel: np.ndarray, region: np.ndarray, n_regions: int):
    """Median implied value per region per good, and the within-region spread.

    Median rather than mean because a single agent that refuses most of its pairs
    implies an extreme level, and a mean would track that one agent instead of
    the region.
    """
    t, n, g = rel.shape
    med = np.full((t, n_regions, g), np.nan)
    spread = np.full((t, n_regions, g), np.nan)
    for r in range(n_regions):
        here = region == r
        for i in range(t):
            rows = rel[i][here[i]]
            if rows.shape[0] >= 2:
                med[i, r] = np.median(rows, axis=0)
                spread[i, r] = np.subtract(*np.percentile(rows, [75, 25], axis=0))
    return med, spread


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    y = y.astype(float).copy()
    idx = np.arange(len(y))
    valid = np.isfinite(y)
    if not valid.any():
        return y
    y = np.interp(idx, idx[valid], y[valid])
    y[: int(idx[valid][0])] = np.nan
    if window > 1:
        kernel = np.ones(window)
        # Normalised by the kernel's own overlap, or the half-window at each end
        # averages against implicit zeros and fakes a plunge.
        y = np.convolve(y, kernel, mode="same") / np.convolve(np.ones_like(y), kernel, mode="same")
    return y


def build(data):
    return html.Div([
        html.Div([
            html.Label("good", style={"fontSize": "11px", "color": P["text_muted"],
                                      "marginRight": "8px"}),
            dcc.Dropdown(id=f"{_ID}-good",
                         options=[{"label": g, "value": i} for i, g in enumerate(data.goods)],
                         value=0, clearable=False,
                         style={"width": "160px", "fontSize": "12px"}),
            html.Label("smoothing", style={"fontSize": "11px", "color": P["text_muted"],
                                           "margin": "0 8px 0 18px"}),
            dcc.Slider(id=f"{_ID}-smooth", min=1, max=61, step=10, value=11,
                       marks=None, tooltip={"placement": "bottom"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "6px",
                  "marginBottom": "10px", "maxWidth": "700px"}),
        html.Div("Median implied value in each region, as a multiple of what an agent asks "
                 "for its average good. A gap between regions is the chokepoint readout.",
                 style={"fontSize": "11px", "color": P["text_muted"], "margin": "0 0 4px"}),
        dcc.Graph(id=f"{_ID}-level", config={"displayModeBar": False}),
        html.Div("Disagreement within each region (interquartile spread). Falling means "
                 "agents are converging on a price; a market that never converges has none.",
                 style={"fontSize": "11px", "color": P["text_muted"], "margin": "12px 0 4px"}),
        dcc.Graph(id=f"{_ID}-spread", config={"displayModeBar": False}),
    ])


def register(app, data):
    rel = _relative(data["snap_ask"])
    med, spread = _by_region(rel, data["snap_region"], data.n_regions)

    @app.callback(
        Output(f"{_ID}-level", "figure"),
        Output(f"{_ID}-spread", "figure"),
        Input(f"{_ID}-good", "value"),
        Input(f"{_ID}-smooth", "value"),
    )
    def _update(good, window):
        good, window = int(good), int(window)
        t = data["snapshot_ticks"]

        fig = go.Figure()
        for r in range(data.n_regions):
            line(fig, t, _smooth(med[:, r, good], window), f"region {r}", series_color(r))
        fig.add_hline(y=1.0, line=dict(color=P["axis"], width=1, dash="dot"),
                      annotation_text="average good", annotation_position="top left",
                      annotation_font=dict(size=10, color=P["text_muted"]))
        figure(fig, ylabel=f"{data.goods[good]} value ÷ own mean", legend=True,
               height=260, margin_right=90)

        fig2 = go.Figure()
        for r in range(data.n_regions):
            line(fig2, t, _smooth(spread[:, r, good], window), f"region {r}", series_color(r))
        figure(fig2, ylabel="IQR of implied value", legend=True, height=210, margin_right=90)
        return fig, fig2


PANEL = Panel(
    id=_ID,
    title="Prices by region",
    blurb="The value level each agent's posted rates imply, set every tick — denser and "
          "steadier than the sparse record of what actually executed.",
    build=build,
    register=register,
    order=30,
    requires=("snap_ask", "snap_region"),
)
