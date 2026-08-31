"""Shared chart theme.

One palette, one Plotly template, applied by every panel so the dashboard reads
as a single instrument rather than a pile of plots. Both modes are validated
(adjacent-pair CVD dE 9.1 light / 8.4 dark; normal-vision 19.6 / 19.3).

The light surface puts three of the five series below 3:1 contrast, so series
identity is never left to color alone: every multi-series chart carries a legend
and lines are direct-labelled at their right end.
"""

from __future__ import annotations

import numpy as np

MODE = "light"  # "light" or "dark" -- selected, never auto-flipped

_PALETTES = {
    "light": {
        "surface": "#fcfcfb",
        "panel": "#ffffff",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "text_muted": "#8a8983",
        "grid": "#e9e8e4",
        "axis": "#c9c8c2",
        "series": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                   "#008300", "#4a3aa7", "#e34948"),
        "sequential": ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
                       "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"),
        "good": "#0ca30c",
        "critical": "#d03b3b",
    },
    "dark": {
        "surface": "#1a1a19",
        "panel": "#232322",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "text_muted": "#8a8983",
        "grid": "#383835",
        "axis": "#4a4a46",
        "series": ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
                   "#008300", "#9085e9", "#e66767"),
        "sequential": ("#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf",
                       "#2a78d6", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"),
        "good": "#0ca30c",
        "critical": "#d03b3b",
    },
}

P = _PALETTES[MODE]


# Sequential ramps for shading table cells by magnitude. One hue each, light to
# dark, never a rainbow. Every step was checked to keep primary ink at or above
# 4.5:1 in its own mode, so a shaded cell stays readable at any value:
#   light  green 4.84  blue 5.69  orange 5.18
#   dark   green 5.30  blue 5.31  orange 4.89
# Hue carries meaning, not just decoration:
#   green   what an agent wants and what it can make
#   blue    what an agent is capable of -- sight and speed
#   orange  quantities of goods, whether held or moving in a trade
_RAMPS = {
    "light": {
        "green": ("#eef7ee", "#cee6d3", "#aed4b9", "#8ec39e", "#6fb283", "#4fa069", "#2f8f4e"),
        "blue": ("#eaf2fd", "#cfe1f8", "#b5d0f2", "#9abfec", "#7faee7", "#659de2", "#4a8cdc"),
        "orange": ("#fdf0e8", "#f6d8c9", "#efc1a9", "#e8a98a", "#e0916b", "#d97a4b", "#d2622c"),
    },
    "dark": {
        "green": ("#1c2a1f", "#1e3725", "#20452c", "#235232", "#255f38", "#276d3e", "#2a7a44"),
        "blue": ("#18222f", "#1b2e47", "#1e3a5f", "#214677", "#24538f", "#275fa7", "#2a6bbf"),
        "orange": ("#2b1e17", "#422718", "#5a301a", "#72381b", "#89411c", "#a04a1e", "#b8531f"),
    },
}
RAMP = _RAMPS[MODE]


def shade(value: float, vmax: float, hue: str = "green", curve: float = 0.6) -> str:
    """Background tint encoding `value` against a ceiling, light to dark.

    `vmax` should be a run-wide ceiling rather than a per-tick one: shading that
    renormalises every tick makes a cell change colour when nothing about it
    changed, which defeats tracking a value across time.

    `curve` below 1 spreads the low end, where most values sit -- a linear map
    leaves everything except the outliers in the palest two steps.
    """
    if not np.isfinite(value) or value <= 0 or vmax <= 0:
        return "transparent"
    steps = RAMP[hue]
    t = min(max(value / vmax, 0.0), 1.0) ** curve
    return steps[min(int(t * len(steps)), len(steps) - 1)]


def series_color(i: int) -> str:
    """Categorical hue for slot `i`, in fixed order.

    Never cycles: a 9th series folds into 'other' or gets its own facet rather
    than reusing slot 1, because color must follow the entity, not its rank.
    """
    if i >= len(P["series"]):
        return P["text_muted"]
    return P["series"][i]


def sequential_scale() -> list[list]:
    """One-hue light-to-dark ramp for magnitude (heatmaps). Never a rainbow."""
    n = len(P["sequential"])
    return [[i / (n - 1), c] for i, c in enumerate(P["sequential"])]


def figure(fig, *, height: int = 300, ylabel: str = "", xlabel: str = "tick",
           legend: bool = False, margin_right: int = 70):
    """Apply the house style. `margin_right` leaves room for direct labels."""
    fig.update_layout(
        height=height,
        margin=dict(l=56, r=margin_right, t=14, b=36),
        paper_bgcolor=P["panel"],
        plot_bgcolor=P["panel"],
        font=dict(family="ui-sans-serif, -apple-system, Segoe UI, sans-serif",
                  size=12, color=P["text_secondary"]),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=P["panel"], font_size=11,
                        bordercolor=P["axis"], font_color=P["text_primary"]),
    )
    axis = dict(showgrid=True, gridcolor=P["grid"], gridwidth=1, zeroline=False,
                linecolor=P["axis"], ticks="outside", tickcolor=P["axis"],
                ticklen=4, tickfont=dict(size=11, color=P["text_muted"]))
    fig.update_xaxes(title_text=xlabel, title_font=dict(size=11, color=P["text_muted"]), **axis)
    fig.update_yaxes(title_text=ylabel, title_font=dict(size=11, color=P["text_muted"]), **axis)
    return fig


def line(fig, x, y, name: str, color: str, *, label: bool = True, width: int = 2, dash=None):
    """A 2px line with an optional direct label at its right end.

    Direct labels are the relief for the light surface's sub-3:1 series, and
    they mean the reader never has to trace a hue back to a legend swatch.
    """
    import plotly.graph_objects as go

    fig.add_trace(go.Scatter(
        x=x, y=y, name=name, mode="lines",
        line=dict(color=color, width=width, dash=dash),
        hovertemplate=f"{name}: %{{y:.3f}}<extra></extra>",
    ))
    if label and len(x):
        import numpy as np
        yi = np.asarray(y, dtype=float)
        finite = np.flatnonzero(np.isfinite(yi))
        if finite.size:
            j = int(finite[-1])
            fig.add_annotation(x=x[j], y=yi[j], text=f" {name}", showarrow=False,
                               xanchor="left", yanchor="middle",
                               font=dict(size=11, color=P["text_secondary"]))
    return fig
