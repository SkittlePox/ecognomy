"""Welfare — the designer's score.

Total realised reward across the population: consumption utility net of the
effort and travel spent getting it. A measure of how much pleasurable
consumption the world actually delivered.

The number alone means little, because a world can post a high total simply by
making production cheap. What matters is how much of it the *market* is
responsible for, so it is shown against the same world with the market switched
off. Tuning parameters to raise the gap is the game.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html

from ecognomy.viewer.panel import Panel
from ecognomy.viewer.theme import P, figure, line, series_color

# Diverging pair: two poles that read as opposite, with sign carrying the meaning.
_UP, _DOWN = series_color(0), P["critical"]


def _stat(label: str, value: str, tone: str | None = None, note: str = "") -> html.Div:
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "color": P["text_muted"],
                               "textTransform": "uppercase", "letterSpacing": "0.04em"}),
        html.Div(value, style={"fontSize": "27px", "fontWeight": 600,
                               "color": tone or P["text_primary"], "lineHeight": "1.15",
                               "fontVariantNumeric": "tabular-nums"}),
        html.Div(note, style={"fontSize": "11px", "color": P["text_muted"]}),
    ], style={"padding": "12px 18px 14px", "background": P["panel"],
              "border": f"1px solid {P['grid']}", "borderRadius": "6px", "minWidth": "140px"})


def build(data):
    t = data.ticks
    cumulative = data["cumulative_reward"]
    welfare = float(cumulative[-1]) if len(cumulative) else 0.0
    has_base = data.has("baseline_welfare")

    if not has_base:
        tiles = [_stat("total welfare", f"{welfare:,.1f}",
                       note="no autarky baseline in this run")]
        gain = None
    else:
        base = float(data["baseline_welfare"][0])
        gain = welfare - base
        ratio = (welfare / base) if base > 1e-12 else float("nan")
        ratio_note = "vs autarky" if ratio != ratio else f"x{ratio:.2f} vs autarky"
        per = data["final_reward_per_agent"]
        base_per = data["baseline_reward_per_agent"]
        helped = float((per > base_per).mean())
        tiles = [
            _stat("total welfare", f"{welfare:,.1f}", note="with the market"),
            _stat("autarky baseline", f"{base:,.1f}", note="market switched off"),
            _stat("gain from trade", f"{gain:+,.1f}",
                  tone=_UP if gain > 0 else (_DOWN if gain < 0 else None), note=ratio_note),
            _stat("agents better off", f"{helped:.0%}",
                  note="than their autarky self"),
        ]

    children = [
        html.Div(tiles, style={"display": "flex", "gap": "10px", "flexWrap": "wrap",
                               "marginBottom": "16px"}),
    ]

    if has_base and gain is not None:
        verdict = ("Trade is creating value." if gain > 1e-6 else
                   "Trade is destroying value — agents would do better with no market."
                   if gain < -1e-6 else "Trade is doing nothing here.")
        children.append(html.Div(verdict, style={
            "fontSize": "12px", "marginBottom": "12px",
            "color": _UP if gain > 1e-6 else (_DOWN if gain < -1e-6 else P["text_muted"])}))

    # Cumulative welfare against the counterfactual. Same units, one axis.
    fig = go.Figure()
    line(fig, t, cumulative, "with market", series_color(0))
    if data.has("baseline_cumulative_reward"):
        base_cum = data["baseline_cumulative_reward"]
        n = min(len(t), len(base_cum))
        line(fig, t[:n], base_cum[:n], "autarky", P["text_muted"], dash="dot")
    figure(fig, ylabel="cumulative welfare", legend=True, height=260, margin_right=100)
    children.append(dcc.Graph(figure=fig, config={"displayModeBar": False}))

    # Who gained and who lost. A total can rise while most agents fall.
    if has_base:
        per = data["final_reward_per_agent"]
        base_per = data["baseline_reward_per_agent"]
        delta = per - base_per
        order = np.argsort(delta)
        bar = go.Figure(go.Bar(
            x=[f"a{i}" for i in order], y=delta[order],
            marker=dict(color=[_UP if d > 0 else _DOWN for d in delta[order]],
                        cornerradius=4),
            hovertemplate="%{x}: %{y:+.2f} vs autarky<extra></extra>",
        ))
        figure(bar, ylabel="welfare vs autarky", xlabel="", height=220, margin_right=20)
        bar.update_xaxes(showgrid=False)
        bar.add_hline(y=0, line=dict(color=P["axis"], width=1))
        children += [
            html.Div("Per agent, against its own autarky counterpart. A market can lift the "
                     "total while leaving most agents worse off — this is where that shows.",
                     style={"fontSize": "11px", "color": P["text_muted"], "margin": "12px 0 4px"}),
            dcc.Graph(figure=bar, config={"displayModeBar": False}),
        ]

    return html.Div(children)


PANEL = Panel(
    id="welfare",
    title="Welfare vs autarky",
    blurb="Total realised consumption utility, net of effort and travel, measured against "
          "the same world with no market. Raising the gap is the design goal.",
    build=build,
    order=5,
    requires=("cumulative_reward",),
)
