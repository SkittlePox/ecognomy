"""Observables.

The dashboard is the product, so these are first-class rather than logging. Each
tick appends one record; arrays come out at the end for plotting or replay.

Two accounts are tracked separately: goods (created by production, destroyed by
consumption and spoilage) and utility (destroyed by effort and travel). A world
can be goods-stable and utility-bankrupt, and only tracking both shows it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# An agent is treated as not valuing a good when its weight is below this. Goods
# held anyway are being held for exchange value -- the shared precursor to both
# arbitrage and money acceptance.
INDIFFERENCE_THRESHOLD = 0.02


@dataclass
class TickRecord:
    t: int
    reward_mean: float
    reward_total: float
    consumption_utility: float
    cumulative_reward: float
    n_trades: int
    trade_volume: float
    n_producing: int
    fraction_producing: float
    goods_stock: float
    utility_burned: float
    token_weight: float
    token_trade_share: float
    exchange_value_holdings: float
    price_by_region: np.ndarray  # (R, G, G) nan where no trade
    volume_by_pair: np.ndarray  # (G, G)
    herfindahl: np.ndarray  # (G,)
    n_in_transit: int
    n_illegal: int


@dataclass
class Metrics:
    """Recorder. Call `record(world)` after each `world.step`."""

    records: list[TickRecord] = field(default_factory=list)

    def record(self, world) -> TickRecord:
        g, r = world.n_goods, world.topology.n_regions
        trades = world.last_trades
        token = world.config.token.token_good

        price = np.full((r, g, g), np.nan, dtype=np.float32)
        volume = np.zeros((g, g), dtype=np.float32)
        counts = np.zeros((r, g, g), dtype=np.int32)
        sums = np.zeros((r, g, g), dtype=np.float64)
        token_trades = 0
        for tr in trades:
            volume[tr.good_a, tr.good_b] += tr.qty_a
            volume[tr.good_b, tr.good_a] += tr.qty_b
            sums[tr.region, tr.good_a, tr.good_b] += tr.price
            counts[tr.region, tr.good_a, tr.good_b] += 1
            sums[tr.region, tr.good_b, tr.good_a] += 1.0 / tr.price
            counts[tr.region, tr.good_b, tr.good_a] += 1
            if token in (tr.good_a, tr.good_b):
                token_trades += 1
        nz = counts > 0
        price[nz] = (sums[nz] / counts[nz]).astype(np.float32)

        # Concentration of holdings per good. Rises when a good is being
        # cornered, which low spoilage permits.
        totals = world.inventory.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            shares = np.where(totals > 0, world.inventory / np.maximum(totals, 1e-12), 0.0)
        herfindahl = (shares**2).sum(axis=0).astype(np.float32)

        indifferent = world.theta < INDIFFERENCE_THRESHOLD
        exchange_value = float(world.inventory[indifferent].sum())

        producing = int((world.last_production.sum(axis=1) > 0).sum())
        utility_burned = float(
            world.config.production.effort_cost * producing
            + world.config.mobility.travel_cost_per_tick * int((~world.settled).sum())
        )

        rec = TickRecord(
            t=world.t,
            reward_mean=float(world.last_reward.mean()),
            reward_total=float(world.last_reward.sum()),
            consumption_utility=float(world.last_utility.sum()),
            cumulative_reward=float(world.cumulative_reward.sum()),
            n_trades=len(trades),
            trade_volume=float(volume.sum()),
            n_producing=producing,
            fraction_producing=producing / world.n_agents,
            goods_stock=float(world.inventory.sum()),
            utility_burned=utility_burned,
            token_weight=world.token_weight(),
            token_trade_share=(token_trades / len(trades)) if trades else 0.0,
            exchange_value_holdings=exchange_value,
            price_by_region=price,
            volume_by_pair=volume,
            herfindahl=herfindahl,
            n_in_transit=int((~world.settled).sum()),
            n_illegal=int(world.last_illegal.sum()),
        )
        self.records.append(rec)
        return rec

    # ------------------------------------------------------------- readouts

    def series(self, name: str) -> np.ndarray:
        return np.array([getattr(r, name) for r in self.records])

    def price_dispersion(self, good_a: int, good_b: int) -> np.ndarray:
        """(T,) spread of the a/b price across regions, nan where under-traded.

        This is the chokepoint readout: throttling an edge should open a
        visible and persistent wedge here.
        """
        out = []
        for rec in self.records:
            p = rec.price_by_region[:, good_a, good_b]
            p = p[~np.isnan(p)]
            out.append(float(p.max() - p.min()) if p.size >= 2 else np.nan)
        return np.array(out)

    def summary(self) -> dict[str, float]:
        """Whole-run headline numbers. A dead world should be obvious here."""
        if not self.records:
            return {}
        return {
            "ticks": len(self.records),
            "welfare": float(self.series("reward_total").sum()),
            "consumption_utility": float(self.series("consumption_utility").sum()),
            "total_trades": float(self.series("n_trades").sum()),
            "trade_volume": float(self.series("trade_volume").sum()),
            "mean_fraction_producing": float(self.series("fraction_producing").mean()),
            "mean_reward": float(self.series("reward_mean").mean()),
            "final_token_weight": float(self.series("token_weight")[-1]),
            "mean_token_trade_share": float(self.series("token_trade_share").mean()),
            "final_exchange_value_holdings": float(self.series("exchange_value_holdings")[-1]),
            "max_herfindahl": float(np.max([r.herfindahl.max() for r in self.records])),
        }
