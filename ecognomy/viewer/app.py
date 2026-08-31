"""Dashboard assembly.

    python -m ecognomy.viewer.app                 # newest run under runs/
    python -m ecognomy.viewer.app runs/my-run     # a specific run

Panels are discovered from `panels/`; this module knows none of them by name.
"""

from __future__ import annotations

import argparse

from dash import Dash, html

from ecognomy.recorder import RunData
from ecognomy.viewer.registry import discover_panels
from ecognomy.viewer.theme import P


def _section(panel, data) -> html.Div:
    return html.Div([
        html.H2(panel.title, style={"fontSize": "15px", "fontWeight": 600, "margin": "0 0 2px",
                                    "color": P["text_primary"]}),
        html.Div(panel.blurb, style={"fontSize": "12px", "color": P["text_muted"],
                                     "margin": "0 0 12px", "maxWidth": "70ch"}),
        panel.build(data),
    ], id=f"panel-{panel.id}",
        style={"background": P["panel"], "border": f"1px solid {P['grid']}",
               "borderRadius": "8px", "padding": "18px 20px 20px", "marginBottom": "16px"})


def build_app(data: RunData) -> Dash:
    app = Dash(__name__, title="Ecognomy")
    panels = [p for p in discover_panels() if p.available(data)]

    cfg = data.config
    header = html.Div([
        html.H1("Ecognomy", style={"fontSize": "19px", "fontWeight": 600, "margin": "0 0 3px",
                                   "color": P["text_primary"]}),
        html.Div(
            f"{data.path.name} · {data.n_ticks} ticks · {data.n_agents} agents · "
            f"{data.n_regions} regions · sight={cfg['visibility']['sight_mean']}"
            f"±{cfg['visibility']['sight_spread']}",
            style={"fontSize": "12px", "color": P["text_muted"],
                   "fontFamily": "ui-monospace, monospace"},
        ),
    ], style={"marginBottom": "18px"})

    app.layout = html.Div(
        [header] + [_section(p, data) for p in panels],
        style={"maxWidth": "1000px", "margin": "0 auto", "padding": "26px 22px 60px",
               "background": P["surface"], "minHeight": "100vh",
               "fontFamily": "ui-sans-serif, -apple-system, Segoe UI, sans-serif"},
    )
    for panel in panels:
        if panel.register is not None:
            panel.register(app, data)
    return app


def main() -> None:
    p = argparse.ArgumentParser(description="View a recorded run.")
    p.add_argument("run", nargs="?", default=None, help="run directory; default: newest under runs/")
    p.add_argument("--port", type=int, default=8050)
    args = p.parse_args()

    data = RunData(args.run) if args.run else RunData.latest()
    print(f"serving {data.path} on http://127.0.0.1:{args.port}")
    build_app(data).run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
