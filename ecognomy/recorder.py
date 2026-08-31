"""Run recording.

The simulator is headless and writes an append-only record to disk; the viewer
reads it. Rendering can therefore never throttle the simulation, and replay is
not a separate feature -- live viewing is tailing a run in progress, replay is
reading a finished one, and both are the same code path.

A run is one directory: `config.json` plus `run.npz`.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from ecognomy.config import WorldConfig
from ecognomy.metrics import Metrics


class RunRecorder:
    """Accumulates a run in memory, then writes it out.

    Aggregates come from `Metrics`; per-agent snapshots and the trade log are
    kept here because only the viewer needs them.
    """

    def __init__(self, world, metrics: Metrics | None = None) -> None:
        self.world = world
        self.config = world.config
        self.metrics = metrics or Metrics()
        self.snapshot_ticks: list[int] = []
        self.inventory: list[np.ndarray] = []
        self.region: list[np.ndarray] = []
        self.edge: list[np.ndarray] = []
        self.reward: list[np.ndarray] = []
        self.price: list[np.ndarray] = []
        self.consume: list[np.ndarray] = []
        self.effort: list[np.ndarray] = []
        self.max_trade: list[np.ndarray] = []
        self.move: list[np.ndarray] = []
        self.visibility: list[np.ndarray] = []
        self.trades: list[tuple] = []
        # Set by `simulate` when an autarky counterfactual is run alongside.
        self.comparison = None
        self.baseline_cumulative = None

    def record(self) -> None:
        w = self.world
        self.metrics.record(w)
        if self.config.recording.record_trades:
            for tr in w.last_trades:
                self.trades.append(
                    (w.t, tr.agent_a, tr.agent_b, tr.good_a, tr.good_b,
                     tr.qty_a, tr.qty_b, tr.price, tr.region)
                )
        if (w.t - 1) % max(1, self.config.recording.snapshot_interval) != 0:
            return
        self.snapshot_ticks.append(w.t)
        self.inventory.append(w.inventory.copy())
        self.region.append(w.region.copy())
        self.edge.append(w.edge.copy())
        self.reward.append(w.last_reward.copy())
        a = w.last_actions
        self.price.append(a.price.copy())
        self.consume.append(a.consume.copy())
        self.effort.append(a.effort.copy())
        self.max_trade.append(a.max_trade.copy())
        self.move.append(a.move.copy())
        if w.n_agents <= self.config.recording.visibility_max_agents:
            self.visibility.append(w.last_visibility.copy())

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        recs = self.metrics.records
        w = self.world

        arrays = {
            # aggregates, one row per tick
            "t": np.array([r.t for r in recs], dtype=np.int32),
            "reward_mean": self.metrics.series("reward_mean"),
            "reward_total": self.metrics.series("reward_total"),
            "consumption_utility": self.metrics.series("consumption_utility"),
            "cumulative_reward": self.metrics.series("cumulative_reward"),
            "final_reward_per_agent": w.cumulative_reward.astype(np.float64),
            "final_utility_per_agent": w.cumulative_utility.astype(np.float64),
            "n_trades": self.metrics.series("n_trades"),
            "trade_volume": self.metrics.series("trade_volume"),
            "fraction_producing": self.metrics.series("fraction_producing"),
            "goods_stock": self.metrics.series("goods_stock"),
            "utility_burned": self.metrics.series("utility_burned"),
            "token_weight": self.metrics.series("token_weight"),
            "token_trade_share": self.metrics.series("token_trade_share"),
            "exchange_value_holdings": self.metrics.series("exchange_value_holdings"),
            "n_in_transit": self.metrics.series("n_in_transit"),
            "n_illegal": self.metrics.series("n_illegal"),
            "price_by_region": np.stack([r.price_by_region for r in recs]) if recs else np.zeros((0,)),
            "volume_by_pair": np.stack([r.volume_by_pair for r in recs]) if recs else np.zeros((0,)),
            "herfindahl": np.stack([r.herfindahl for r in recs]) if recs else np.zeros((0,)),
            # per-agent snapshots
            "snapshot_ticks": np.array(self.snapshot_ticks, dtype=np.int32),
            "snap_inventory": np.stack(self.inventory) if self.inventory else np.zeros((0,)),
            "snap_region": np.stack(self.region) if self.region else np.zeros((0,)),
            "snap_edge": np.stack(self.edge) if self.edge else np.zeros((0,)),
            "snap_reward": np.stack(self.reward) if self.reward else np.zeros((0,)),
            "snap_price": np.stack(self.price) if self.price else np.zeros((0,)),
            "snap_consume": np.stack(self.consume) if self.consume else np.zeros((0,)),
            "snap_effort": np.stack(self.effort) if self.effort else np.zeros((0,)),
            "snap_max_trade": np.stack(self.max_trade) if self.max_trade else np.zeros((0,)),
            "snap_move": np.stack(self.move) if self.move else np.zeros((0,)),
            "snap_visibility": np.stack(self.visibility) if self.visibility else np.zeros((0,)),
            # agent statics
            "theta": w.theta, "theta_base": w.theta_base,
            "efficiency": w.efficiency, "mobility": w.mobility,
            "sight": w.sight,
            # topology
            "edge_src": w.topology.src, "edge_dst": w.topology.dst,
            "edge_weight": w.topology.weight, "edge_capacity": w.topology.capacity,
            "trades": np.array(self.trades, dtype=np.float64) if self.trades else np.zeros((0, 9)),
        }
        if self.comparison is not None:
            c = self.comparison
            arrays["baseline_welfare"] = np.array([c.baseline_welfare])
            arrays["baseline_reward_per_agent"] = c.baseline_per_agent
            arrays["baseline_consumption_utility"] = np.array([c.baseline_consumption_utility])
            if self.baseline_cumulative is not None:
                arrays["baseline_cumulative_reward"] = self.baseline_cumulative
        np.savez_compressed(path / "run.npz", **arrays)
        (path / "config.json").write_text(json.dumps(_config_to_dict(self.config), indent=2))
        return path


def _config_to_dict(cfg: WorldConfig) -> dict:
    """Config as plain JSON. Topology becomes edge lists, which the npz also holds."""
    def convert(v):
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return {f.name: convert(getattr(v, f.name)) for f in dataclasses.fields(v)}
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        if isinstance(v, (list, tuple)):
            return [convert(x) for x in v]
        return v
    return convert(cfg)


class RunData:
    """A saved run, loaded for viewing. Read-only."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with np.load(self.path / "run.npz", allow_pickle=False) as z:
            self.arrays = {k: z[k] for k in z.files}
        self.config = json.loads((self.path / "config.json").read_text())
        self.goods: list[str] = list(self.config["goods"])
        self.n_agents: int = int(self.config["n_agents"])
        self.n_regions: int = int(self.config["topology"]["n_regions"])
        self.token_good: int = int(self.config["token"]["token_good"])

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    def has(self, key: str) -> bool:
        """Panels use this to degrade gracefully when a run predates a field."""
        return key in self.arrays and self.arrays[key].size > 0

    @property
    def n_goods(self) -> int:
        return len(self.goods)

    @property
    def ticks(self) -> np.ndarray:
        return self.arrays["t"]

    @property
    def n_ticks(self) -> int:
        return len(self.arrays["t"])

    def trade_log(self) -> np.ndarray:
        """(n_trades, 9): tick, agent_a, agent_b, good_a, good_b, qty_a, qty_b, price, region."""
        return self.arrays["trades"]

    @classmethod
    def latest(cls, root: str | Path = "runs") -> "RunData":
        root = Path(root)
        runs = sorted(p for p in root.glob("*") if (p / "run.npz").exists())
        if not runs:
            raise FileNotFoundError(f"no runs found under {root!r} -- run `python -m ecognomy.simulate` first")
        return cls(runs[-1])


def simulate(config: WorldConfig, policy, ticks: int | None = None,
             out: str | Path | None = None, scenario=None, baseline: bool = True):
    """Run headless and save. Returns the recorder.

    With `baseline`, the same world is also run with the market switched off, so
    the saved run carries the autarky counterfactual its welfare is measured
    against.
    """
    from ecognomy.world import World

    ticks = ticks if ticks is not None else config.recording.default_ticks
    world = World(config, scenario=scenario)
    rec = RunRecorder(world)
    for _ in range(ticks):
        world.step(policy.act(world, world.rng))
        rec.record()

    if baseline:
        from ecognomy.baseline import Comparison, run_baseline

        base_world, base_metrics = run_baseline(config, policy, ticks, scenario=scenario)
        rec.comparison = Comparison(
            welfare=world.welfare,
            baseline_welfare=base_world.welfare,
            consumption_utility=float(world.cumulative_utility.sum()),
            baseline_consumption_utility=float(base_world.cumulative_utility.sum()),
            per_agent=world.cumulative_reward.copy(),
            baseline_per_agent=base_world.cumulative_reward.copy(),
        )
        rec.baseline_cumulative = base_metrics.series("cumulative_reward")

    if out is not None:
        rec.save(out)
    return rec
