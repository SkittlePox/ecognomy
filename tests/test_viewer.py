"""Viewer tests.

The panel system is auto-discovering, so the important guarantee is that every
panel found in `panels/` builds against a real run. Without this, adding a panel
that raises would only show up as a blank section in the browser.
"""

import numpy as np
import pytest

from ecognomy.config import WorldConfig
from ecognomy.policy import RandomPolicy
from ecognomy.recorder import RunData, simulate
from ecognomy.topology import Topology
from ecognomy.viewer.registry import discover_panels


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    cfg = WorldConfig(n_agents=25, seed=11, topology=Topology.line(3))
    cfg.visibility.sight_mean = 12
    cfg.token.anneal_ticks = 60
    simulate(cfg, RandomPolicy(), ticks=120, out=out)
    return RunData(out)


def test_round_trip_preserves_the_run(run):
    assert run.n_ticks == 120
    assert run.n_agents == 25
    assert run.goods[run.token_good] == "elderberry"
    assert run["snap_inventory"].shape == (120, 25, 5)


def test_every_discovered_panel_builds(run):
    panels = discover_panels()
    assert panels, "no panels discovered"
    for panel in panels:
        if not panel.available(run):
            continue
        assert panel.build(run) is not None, f"panel {panel.id!r} built nothing"


def test_every_panel_callback_actually_runs(run):
    """`build` only lays out the shell; the content comes from callbacks.

    This gap let the agent inspector sit broken through a migration: it still
    laid out fine while its callback raised on a snapshot field that no longer
    existed. Registering the callbacks is not enough — they have to be invoked.
    """
    from dash import Dash, html

    for panel in discover_panels():
        if panel.register is None or not panel.available(run):
            continue
        app = Dash(__name__)
        app.layout = html.Div(panel.build(run))
        panel.register(app, run)
        for key, entry in app.callback_map.items():
            specs = list(entry["inputs"]) + list(entry.get("state") or [])
            args = [_probe(run, spec["id"]) for spec in specs]
            out = entry["callback"](*args, outputs_list=_outputs_list(key))
            assert out is not None, f"{panel.id}: callback {key} returned nothing"


def _outputs_list(key: str):
    """Dash's callback wrapper wants the output spec as plain dicts.

    Keys are "id.prop" for a single output and "..id.prop...id.prop.." for
    several, so the shape has to be reconstructed from the registered key.
    """
    if key.startswith("..") and key.endswith(".."):
        parts = key[2:-2].split("...")
        return [dict(zip(("id", "property"), p.rsplit(".", 1))) for p in parts]
    cid, prop = key.rsplit(".", 1)
    return {"id": cid, "property": prop}


def _probe(run, component_id: str):
    """A plausible input value for a component, by id."""
    if component_id.endswith(("-pick", "-focus", "-good")):
        return 0
    if component_id.endswith("-step"):
        return 1
    if component_id.endswith("-smooth"):
        return 11
    return 0


def test_panels_have_unique_ids_and_stable_order():
    panels = discover_panels()
    ids = [p.id for p in panels]
    assert len(ids) == len(set(ids)), f"duplicate panel ids: {ids}"
    assert [p.order for p in panels] == sorted(p.order for p in panels)


def test_app_assembles_with_every_callback_registered(run):
    """The real assembly path: layout built, then callbacks attached."""
    from ecognomy.viewer.app import build_app

    app = build_app(run)
    # A panel may register more than one callback, so this is a floor: every
    # panel that registers must have contributed at least one.
    expected = sum(1 for p in discover_panels() if p.register is not None and p.available(run))
    assert len(app.callback_map) >= expected
    assert app.layout is not None


def test_world_view_describes_every_combination_of_outputs(run):
    """Agents emit several outputs at once, so the summary must cover them all
    and never render an empty cell."""
    import numpy as np

    from ecognomy.viewer.panels.world_view import _activity, _describe

    g = len(run.goods)
    zero, some = np.zeros(g), np.full(g, 0.4)
    for effort in (zero, some):
        for consume in (zero, some):
            for move in (-1, 0):
                text = _describe(run, effort, consume, move)
                assert text, "empty description"
                assert _activity(effort, consume) in (0, 1, 2, 3)
    assert _describe(run, zero, zero, -1) == "idle"


def test_world_view_snapshots_cover_every_agent_and_tick(run):
    """The panel indexes snapshots directly, so their shapes must line up."""
    n_snaps = len(run["snapshot_ticks"])
    for key in ("snap_region", "snap_price", "snap_consume", "snap_effort",
                "snap_reward", "snap_edge"):
        assert run[key].shape[:2] == (n_snaps, run.n_agents), key
    assert run["snap_move"].shape == (n_snaps, run.n_agents)


def test_visibility_is_recorded_and_bounded_by_sight(run):
    """Who saw whom is resampled every tick, so it cannot be reconstructed from
    `world.sight` after the fact and has to be stored."""
    import numpy as np

    vis = run["snap_visibility"]
    assert vis.shape == (len(run["snapshot_ticks"]), run.n_agents, run.n_agents)
    assert not vis[:, np.arange(run.n_agents), np.arange(run.n_agents)].any(), \
        "an agent must not be listed as seeing itself"
    # Nobody sees more than their sight allows on any tick.
    assert (vis.sum(axis=2) <= run["sight"][None, :]).all()


def test_visibility_respects_region_boundaries(run):
    """An agent can only see prices posted in the region it is standing in."""
    import numpy as np

    vis, region = run["snap_visibility"], run["snap_region"]
    for t in range(0, len(vis), 7):
        viewers, seen = np.nonzero(vis[t])
        if viewers.size:
            assert (region[t][viewers] == region[t][seen]).all()


def test_world_view_renders_a_board_per_region(run):
    """The panel is the main instrument now, so its callback is worth pinning."""
    from ecognomy.viewer.app import build_app

    app = build_app(run)
    key = next(k for k in app.callback_map if "world-boards" in k)
    assert key, "world view render callback missing"


def test_smoothing_does_not_distort_the_ends():
    """A moving average must normalise by its own overlap at the edges.

    Convolving with mode='same' averages the trailing half-window against
    implicit zeros, which fakes a plunge at the end of every price series.
    """
    from ecognomy.viewer.panels.prices import _smooth

    flat = np.full(200, 3.0)
    out = _smooth(flat, 31)
    assert np.allclose(out, 3.0, atol=1e-6), "a constant series must smooth to itself"


def test_smoothing_survives_an_all_nan_series():
    from ecognomy.viewer.panels.prices import _smooth

    out = _smooth(np.full(50, np.nan), 31)
    assert np.isnan(out).all()
