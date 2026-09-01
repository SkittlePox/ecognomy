"""The world, one tick at a time.

Summary metrics tell you whether an economy formed. They cannot tell you why an
agent did what it did. This panel replays the recorded run step by step:

  * where every agent is, and everything true about it -- preferences,
    production efficiency, mobility, market access
  * each region's **rate board**, the postings every trade is made against
  * which agents could actually see which postings this tick
  * every trade that executed, and who was in it

Nothing here is re-simulated; it is all read from the run.
"""

from __future__ import annotations

import numpy as np
from dash import Input, Output, State, ctx, dcc, html

from ecognomy.metrics import implied_value, round_trip
from ecognomy.viewer.panel import Panel
from ecognomy.viewer.theme import P, series_color, shade

_ID = "world"

_IDLE, _PRODUCE, _CONSUME, _BOTH = 0, 1, 2, 3
_ACTIVITY = {
    _IDLE: (P["text_muted"], "idle"),
    _PRODUCE: (series_color(0), "producing"),
    _CONSUME: (series_color(2), "consuming"),
    _BOTH: (series_color(1), "producing + consuming"),
}
_SEEN = series_color(3)     # rows the selected agent can see
_SELF = series_color(0)     # the selected agent
_TRADED = series_color(1)   # agents that traded this tick

_MONO = "ui-monospace, SFMono-Regular, Menlo, monospace"
_BTN = {"fontSize": "12px", "padding": "3px 12px", "cursor": "pointer",
        "background": P["surface"], "color": P["text_primary"],
        "border": f"1px solid {P['grid']}", "borderRadius": "5px"}
_CARD = {"flex": "1 1 300px", "minWidth": "280px", "background": P["panel"],
         "border": f"1px solid {P['grid']}", "borderRadius": "8px",
         "padding": "12px 14px 14px"}


def _num(v, width=6, places=2, zero="·"):
    return zero.rjust(width) if abs(v) <= 1e-6 else f"{v:{width}.{places}f}"


def _vec(row, places=2):
    return " ".join(_num(v, 6, places) for v in row)


def _activity(effort_row, consume_row) -> int:
    return _PRODUCE * (float(effort_row.sum()) > 1e-6) + _CONSUME * (float(consume_row.sum()) > 1e-6)


def _describe(data, effort_row, consume_row, move_val) -> str:
    bits = []
    if float(effort_row.sum()) > 1e-6:
        g = int(np.argmax(effort_row))
        bits.append(f"produce {data.goods[g]} ×{effort_row[g]:.2f}")
    if float(consume_row.sum()) > 1e-6:
        bits.append(f"eat {consume_row.sum():.2f}")
    if int(move_val) >= 0:
        dst = int(data["edge_dst"][int(move_val)]) if int(move_val) < len(data["edge_dst"]) else -1
        bits.append(f"move → region {dst}")
    return ", ".join(bits) if bits else "idle"


def _th(text, align="right"):
    return html.Th(text, style={"fontSize": "10px", "fontWeight": 600, "textAlign": align,
                                "color": P["text_muted"], "padding": "2px 6px",
                                "borderBottom": f"1px solid {P['grid']}",
                                "whiteSpace": "nowrap"})


def _td(text, align="right", color=None, weight=None, bg=None):
    return html.Td(text, style={"fontSize": "11px", "fontFamily": _MONO, "textAlign": align,
                                "padding": "2px 6px", "color": color or P["text_primary"],
                                "fontWeight": weight, "whiteSpace": "nowrap",
                                "background": bg})


def build(data):
    n_snaps = len(data["snapshot_ticks"])
    agent_opts = [{"label": "none", "value": -1}] + [
        {"label": f"agent {i}", "value": i} for i in range(data.n_agents)]
    return html.Div([
        html.Div([
            html.Button("◀", id=f"{_ID}-prev", n_clicks=0, style=_BTN),
            html.Button("▶", id=f"{_ID}-next", n_clicks=0, style=_BTN),
            html.Button("play", id=f"{_ID}-play", n_clicks=0, style=_BTN),
            html.Div(id=f"{_ID}-tick", style={"fontFamily": _MONO, "fontSize": "12px",
                                              "color": P["text_secondary"],
                                              "minWidth": "150px", "marginLeft": "10px"}),
            html.Div(style={"flex": "1"}),
            html.Label("highlight what agent sees", style={"fontSize": "11px",
                                                           "color": P["text_muted"],
                                                           "marginRight": "8px"}),
            dcc.Dropdown(id=f"{_ID}-focus", options=agent_opts, value=-1, clearable=False,
                         style={"width": "150px", "fontSize": "12px"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
        dcc.Slider(id=f"{_ID}-step", min=0, max=max(n_snaps - 1, 0), step=1, value=0,
                   marks=None, tooltip={"placement": "bottom"}),
        dcc.Interval(id=f"{_ID}-timer", interval=700, disabled=True),

        html.Div(id=f"{_ID}-legend", style={"margin": "10px 0 2px"}),
        html.Div(id=f"{_ID}-boards", style={"margin": "8px 0"}),
        html.Div(id=f"{_ID}-trades", style={"margin": "14px 0 0"}),
        html.Div(id=f"{_ID}-agents", style={"margin": "14px 0 0"}),
    ])


def _legend(focus: int) -> html.Div:
    def swatch(color, label):
        return html.Span([
            html.Span(style={"display": "inline-block", "width": "8px", "height": "8px",
                             "borderRadius": "2px", "background": color, "marginRight": "5px"}),
            label,
        ], style={"fontSize": "11px", "color": P["text_secondary"], "marginRight": "16px"})
    items = [swatch(series_color(0), "producing"), swatch(series_color(2), "consuming"),
             swatch(series_color(1), "both / traded"), swatch(P["text_muted"], "idle")]
    if focus >= 0:
        items += [swatch(_SELF, f"agent {focus}"), swatch(_SEEN, f"visible to agent {focus}")]
    return html.Div(items)


def register(app, data):
    n_snaps = len(data["snapshot_ticks"])
    has_vis = data.has("snap_visibility")
    has_sight = data.has("sight")

    # Ceilings for the shading, computed once over the whole run. Renormalising
    # per tick would recolour a cell whose value never changed, which is exactly
    # what makes a value impossible to track over time. High percentiles rather
    # than maxima so one outlier does not flatten every ordinary value into the
    # palest step.
    def ceiling(arr, pct=95.0):
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        return float(np.percentile(arr, pct)) if arr.size else 1.0

    wants_ceiling = ceiling(data["theta"], 100.0)
    makes_ceiling = ceiling(data["efficiency"], 100.0)
    sight_ceiling = ceiling(data["sight"], 100.0) if has_sight else 1.0
    speed_ceiling = ceiling(data["mobility"], 100.0)
    inv_ceiling = ceiling(data["snap_inventory"])
    _log = data.trade_log()
    vol_ceiling = ceiling(_log[:, 5] + _log[:, 6]) if _log.size else 1.0
    offer_ceiling = ceiling(data["snap_max_trade"]) if data.has("snap_max_trade") else 1.0

    @app.callback(
        Output(f"{_ID}-step", "value"),
        Input(f"{_ID}-prev", "n_clicks"), Input(f"{_ID}-next", "n_clicks"),
        Input(f"{_ID}-timer", "n_intervals"),
        State(f"{_ID}-step", "value"), prevent_initial_call=True,
    )
    def _move(prev, nxt, _t, current):
        delta = -1 if ctx.triggered_id == f"{_ID}-prev" else 1
        return int(np.clip((current or 0) + delta, 0, max(n_snaps - 1, 0)))

    @app.callback(
        Output(f"{_ID}-timer", "disabled"), Output(f"{_ID}-play", "children"),
        Input(f"{_ID}-play", "n_clicks"), prevent_initial_call=True,
    )
    def _play(n):
        playing = bool(n % 2)
        return (not playing), ("pause" if playing else "play")

    @app.callback(
        Output(f"{_ID}-tick", "children"), Output(f"{_ID}-legend", "children"),
        Output(f"{_ID}-boards", "children"), Output(f"{_ID}-trades", "children"),
        Output(f"{_ID}-agents", "children"),
        Input(f"{_ID}-step", "value"), Input(f"{_ID}-focus", "value"),
    )
    def _render(step, focus):
        s, focus = int(step or 0), int(focus if focus is not None else -1)
        tick = int(data["snapshot_ticks"][s])
        region = data["snap_region"][s]
        edge = data["snap_edge"][s] if data.has("snap_edge") else np.full(len(region), -1)
        ask, consume = data["snap_ask"][s], data["snap_consume"][s]
        # The board is one row per agent, so it shows the value the matrix
        # implies plus the margin it quotes. The matrix itself is G x G and only
        # legible one agent at a time -- it renders under the board for whoever
        # is focused.
        price = implied_value(ask)
        trips = round_trip(ask)
        effort, move = data["snap_effort"][s], data["snap_move"][s]
        inv, reward = data["snap_inventory"][s], data["snap_reward"][s]
        seen = data["snap_visibility"][s] if has_vis else None

        log = data.trade_log()
        rows = log[log[:, 0] == tick] if log.size else np.zeros((0, 9))
        traded = set(int(r[1]) for r in rows) | set(int(r[2]) for r in rows)

        max_trade = data["snap_max_trade"][s] if data.has("snap_max_trade") else np.zeros_like(inv)
        boards = [_board(data, r, region, price, trips, effort, consume, inv, seen, focus,
                         traded, max_trade)
                  for r in range(data.n_regions)]
        boards.append(_transit_card(region, edge))

        header = f"tick {tick}  ·  step {s + 1} / {n_snaps}"
        return (header, _legend(focus),
                html.Div(boards, style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
                _trades(data, rows),
                html.Div([
                    _agents(data, region, edge, price, consume, effort, move, inv,
                            reward, focus, has_sight),
                    html.Div(style={"height": "18px"}),
                    _population(data, focus, has_sight),
                ]))

    def _board(data, r, region, price, trips, effort, consume, inv, seen, focus, traded,
               max_trade):
        """One region: who is here, and the full posting of everyone present.

        A posting is three things now, and all belong here: the **value** its
        rate matrix implies for each good, the **margin** it quotes around that
        value, and **how much** of each good it will actually part with. A rate
        with no quantity behind it cannot be traded against.

        `margin` is the median round trip across the pairs the agent will cycle:
        1.00 is honest reciprocal posting, above 1 is a spread it demands both
        ways, and below 1 means its own postings can be cycled against it.
        """
        here = np.flatnonzero(region == r)
        group = html.Tr([_th("", "left")] +
                        [_th("implied value" if g == 0 else "") for g in range(data.n_goods)] +
                        [_th("")] +
                        [_th("will part with" if g == 0 else "") for g in range(data.n_goods)])
        head = html.Tr([_th("agent", "left")] + [_th(g[:6]) for g in data.goods] +
                       [_th("margin")] + [_th(g[:6]) for g in data.goods])
        body = []
        for i in here:
            i = int(i)
            colour, _ = _ACTIVITY[_activity(effort[i], consume[i])]
            if i in traded:
                colour = _TRADED
            is_self = (i == focus)
            can_see = seen is not None and focus >= 0 and bool(seen[focus, i])
            bg = (P["surface"] if is_self else
                  ("rgba(237,161,0,0.10)" if can_see else "transparent"))
            marker = "●" if i in traded else " "
            label = html.Span([
                html.Span(style={"display": "inline-block", "width": "7px", "height": "7px",
                                 "borderRadius": "50%", "background": colour,
                                 "marginRight": "5px"}),
                f"a{i}{marker}",
            ], style={"display": "inline-flex", "alignItems": "center"})
            cells = [html.Td(label, style={"padding": "2px 6px", "fontSize": "11px",
                                           "fontFamily": _MONO,
                                           "fontWeight": 600 if is_self else None,
                                           "color": _SELF if is_self else P["text_primary"]})]
            cells += [_td(_num(price[i, g], 6, 3),
                          color=_SEEN if can_see else None) for g in range(data.n_goods)]
            finite = trips[i][np.isfinite(trips[i])]
            cells += [_td(_num(float(np.median(finite)), 6, 2) if finite.size else "—",
                          color=P["text_muted"] if not finite.size else None)]
            cells += [_td(_num(max_trade[i, g], 6, 2),
                          bg=shade(max_trade[i, g], offer_ceiling, "orange"))
                      for g in range(data.n_goods)]
            body.append(html.Tr(cells, style={"background": bg}))

        table = html.Table([html.Thead([group, head]), html.Tbody(body)],
                           style={"width": "100%", "borderCollapse": "collapse"}) if len(here) \
            else html.Div("empty", style={"fontSize": "11px", "color": P["text_muted"]})

        subtitle = "value levels are per-agent and only ratios matter"
        return html.Div([
            html.Div(f"region {r}  ·  {len(here)} agent{'s' if len(here) != 1 else ''}",
                     style={"fontSize": "12px", "fontWeight": 600,
                            "color": P["text_primary"], "marginBottom": "2px"}),
            html.Div(subtitle, style={"fontSize": "10px", "color": P["text_muted"],
                                      "fontFamily": _MONO, "marginBottom": "7px"}),
            table,
        ], style=_CARD)

    def _transit_card(region, edge):
        moving = np.flatnonzero(region < 0)
        return html.Div([
            html.Div(f"in transit  ·  {len(moving)}",
                     style={"fontSize": "12px", "fontWeight": 600,
                            "color": P["text_primary"], "marginBottom": "9px"}),
            html.Div(", ".join(f"a{i} (edge {int(edge[i])})" for i in moving) or "nobody",
                     style={"fontSize": "11px", "fontFamily": _MONO,
                            "color": P["text_secondary"]}),
        ], style={**_CARD, "flex": "0 1 200px", "minWidth": "170px",
                  "background": P["surface"]})

    def _trades(data, rows):
        """Executed trades, split by region.

        Trades only ever happen between agents standing in the same region, so
        grouping keeps each market's activity legible instead of interleaving
        several markets in one list.
        """
        cards = []
        for r in range(data.n_regions):
            mine = rows[rows[:, 8] == r]
            head = html.Tr([_th(c, "left") for c in
                            ("", "gave", "", "gave", "rate", "vol")])
            body = []
            for t in mine:
                volume = float(t[5]) + float(t[6])
                bg = shade(volume, vol_ceiling, "orange")
                body.append(html.Tr([
                    # Plain ink, not the accent colour: these rows are shaded
                    # orange by volume, and orange-on-orange is unreadable.
                    _td(f"a{int(t[1])}", "left", weight=600),
                    _td(f"{t[5]:.3f} {data.goods[int(t[3])]}", "left"),
                    _td(f"a{int(t[2])}", "left", weight=600),
                    _td(f"{t[6]:.3f} {data.goods[int(t[4])]}", "left"),
                    _td(f"{t[7]:.3f}", "left"),
                    _td(f"{volume:.2f}", "left", weight=600),
                ], style={"background": bg}))
            # Regions with nothing to show still get a card, so the row of cards
            # keeps the same shape tick to tick and a market going quiet reads as
            # a change rather than as a layout shift.
            table = (html.Table([html.Thead(head), html.Tbody(body)],
                                style={"borderCollapse": "collapse"}) if len(mine)
                     else html.Div("quiet", style={"fontSize": "11px",
                                                   "color": P["text_muted"]}))
            cards.append(html.Div([
                html.Div(f"region {r}  ·  {len(mine)} trade{'s' if len(mine) != 1 else ''}",
                         style={"fontSize": "11px", "fontWeight": 600,
                                "color": P["text_secondary"] if len(mine) else P["text_muted"],
                                "marginBottom": "5px"}),
                table,
            ], style=_CARD))

        return html.Div([
            html.Div(f"{len(rows)} trade{'s' if len(rows) != 1 else ''} executed",
                     style={"fontSize": "12px", "fontWeight": 600,
                            "color": P["text_primary"], "marginBottom": "2px"}),
            html.Div("Shaded by total units moved, against a run-wide ceiling — so a "
                     "big trade looks the same shade whenever it happens.",
                     style={"fontSize": "11px", "color": P["text_muted"],
                            "marginBottom": "7px"}),
            html.Div(cards, style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ])

    def _agents(data, region, edge, price, consume, effort, move, inv, reward, focus, has_sight):
        """What each agent is doing and holding right now.

        Fixed attributes live in the population table below instead: they never
        change from tick to tick, so repeating them here only crowds out the part
        that does.
        """
        head = html.Tr(
            [_th(c, "left") for c in ("agent", "where", "doing")] +
            [_th("earned")] + [_th(g[:6]) for g in data.goods]
        )
        sub = html.Tr([_th("", "left")] * 3 + [_th("")] +
                      [_th("holds" if g == 0 else "") for g in range(data.n_goods)])
        order = np.argsort(np.where(region < 0, data.n_regions, region), kind="stable")
        body = []
        for i in order:
            i = int(i)
            is_self = (i == focus)
            body.append(html.Tr([
                _td(f"a{i}", "left", color=_SELF if is_self else None,
                    weight=600 if is_self else None),
                _td(f"region {int(region[i])}" if region[i] >= 0 else f"edge {int(edge[i])}", "left"),
                _td(_describe(data, effort[i], consume[i], move[i]), "left"),
                _td(f"{reward[i]:+.4f}"),
            ] + [_td(_num(inv[i, g], 6, 2), bg=shade(inv[i, g], inv_ceiling, "orange"))
                 for g in range(data.n_goods)],
                style={"background": P["surface"] if is_self else "transparent"}))
        return html.Div([
            html.Div("This tick", style={"fontSize": "12px", "fontWeight": 600,
                                         "color": P["text_primary"], "marginBottom": "5px"}),
            html.Div(html.Table([html.Thead([sub, head]), html.Tbody(body)],
                                style={"borderCollapse": "collapse", "width": "100%"}),
                     style={"overflowX": "auto"}),
        ])


    def _population(data, focus, has_sight):
        """Fixed attributes, drawn once at spawn and never changing.

        Every good gets its own column rather than a padded string in one cell,
        so the numbers line up column by column no matter what font the browser
        resolves -- which is what makes agents comparable at a glance.
        """
        goods = data.goods
        group = html.Tr(
            [_th("", "left"), _th(""), _th("")] +
            [_th("wants — how much it values each good" if g == 0 else "")
             for g in range(len(goods))] +
            [_th("can make — units per full tick of effort" if g == 0 else "")
             for g in range(len(goods))]
        )
        head = html.Tr(
            [_th("agent", "left"), _th("sight"), _th("speed")] +
            [_th(g[:6]) for g in goods] + [_th(g[:6]) for g in goods]
        )
        body = []
        for i in range(data.n_agents):
            is_self = (i == focus)
            theta, eff = data["theta"][i], data["efficiency"][i]
            sight_v = int(data["sight"][i]) if has_sight else 0
            cells = [
                _td(f"a{i}", "left", color=_SELF if is_self else None,
                    weight=600 if is_self else None),
                _td(str(sight_v) if has_sight else "—",
                    bg=shade(sight_v, sight_ceiling, "blue")),
                _td(f"{data['mobility'][i]:.2f}",
                    bg=shade(float(data["mobility"][i]), speed_ceiling, "blue")),
            ]
            cells += [_td(_num(theta[g], 6, 2),
                          color=None if theta[g] > 1e-6 else P["text_muted"],
                          bg=shade(float(theta[g]), wants_ceiling, "green"))
                      for g in range(len(goods))]
            cells += [_td(_num(eff[g], 6, 2),
                          color=None if eff[g] > 1e-6 else P["text_muted"],
                          weight=600 if eff[g] > 1e-6 else None,
                          bg=shade(float(eff[g]), makes_ceiling, "green"))
                      for g in range(len(goods))]
            body.append(html.Tr(cells, style={
                "background": P["surface"] if is_self else "transparent"}))
        return html.Div([
            html.Div("Population — ground truth", style={"fontSize": "12px", "fontWeight": 600,
                                                         "color": P["text_primary"]}),
            html.Div("None of this is observable by any other agent. `wants` drives what an "
                     "agent will pay for; `can make` is what it can produce at all — a dot "
                     "means it cannot make that good. `sight` is how many posted rates it "
                     "can see in its region; `speed` is how fast it crosses an edge. "
                     "Darker means larger throughout: green for what an agent wants and can "
                     "make, blue for what it is capable of.",
                     style={"fontSize": "11px", "color": P["text_muted"],
                            "margin": "1px 0 7px", "maxWidth": "88ch"}),
            html.Div(html.Table([html.Thead([group, head]), html.Tbody(body)],
                                style={"borderCollapse": "collapse", "width": "100%"}),
                     style={"overflowX": "auto"}),
        ])


PANEL = Panel(
    id=_ID,
    title="World, step by step",
    blurb="Who is where, what they are truly like, the rates each region is posting, "
          "who can see them, and what executed.",
    build=build,
    register=register,
    order=10,
    requires=("snap_region", "snap_ask"),
)
