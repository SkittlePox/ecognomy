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

# Guards log(0) for a posting that gives a good away. The resulting round trip
# is enormous but finite, which is honest: a giveaway really is a deep pump if
# there is a way back.
RATE_FLOOR = 1e-12

# Postings are stored as float32, so an exactly-reciprocal matrix round trips to
# 1 give or take ~1e-7. Counting that as a money pump would report every honest
# agent in the population as exploitable. This is a storage tolerance, not an
# economic threshold -- a pump worth naming is orders of magnitude deeper.
PUMP_TOLERANCE = 1e-5


def round_trip(ask: np.ndarray) -> np.ndarray:
    """(N, G, G) what a unit survives on a round trip through each pair.

    `ask[a, b] * ask[b, a]` is what an agent's own two postings imply about
    converting a unit of `a` into `b` and straight back at its own reservation
    rates: it keeps `1 / product` of what it started with.

      * `> 1` -- a spread. The agent demands a margin in both directions, which
        is the two-sided quote the rate matrix exists to make expressible.
      * `== 1` -- an honest reciprocal posting, the whole of what the old price
        vector could say.
      * `< 1` -- its own bid crosses its own ask. A counterparty can cycle a unit
        through it and hand back less than it took. Legal, and measured here
        rather than prevented.

    A refusal in either leg makes the round trip impassable rather than free, so
    it comes back as `inf`: the pair cannot be cycled at all. That is what keeps
    a good an agent does not consume -- refused one way, given away the other --
    from reading as an infinite money pump.
    """
    a = np.asarray(ask, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        product = a * np.swapaxes(a, -1, -2)
    blocked = ~np.isfinite(a) | ~np.isfinite(np.swapaxes(a, -1, -2))
    product[blocked] = np.inf
    idx = np.arange(a.shape[-1])
    product[..., idx, idx] = np.inf
    return product


def implied_value(ask: np.ndarray) -> np.ndarray:
    """(..., G) the one-number-per-good valuation a rate matrix is nearest to.

    For a coherent posting `ask[a, b] = p_a / p_b`, so the mean of `log ask[a, :]`
    recovers `log p_a` up to an additive constant: the row means *are* the value
    vector, normalised to a geometric mean of 1. For an incoherent posting there
    is no such vector and this returns the least-squares rank-1 fit -- the best
    single-number summary of a matrix that does not admit one. Refusals are
    dropped rather than treated as enormous rates, and a good an agent refuses
    to give up at all comes back as nan.

    **This is a readout, not a price.** It exists so a dashboard can draw one
    line per good instead of G lines per agent. Nothing in the tick computes it,
    and no agent observes it -- a global valuation visible to agents is exactly
    the published price the design refuses.
    """
    a = np.asarray(ask, dtype=np.float64)
    with np.errstate(divide="ignore"):
        logs = np.log(np.maximum(a, RATE_FLOOR))
    usable = np.isfinite(a)
    # The diagonal counts, as a self-rate of exactly 1. Dropping it would leave
    # row `a` averaging over every good *except* `a`, so the constant being
    # subtracted would differ per row and the result would not be a rescaling of
    # anything -- close enough to look right and wrong everywhere.
    idx = np.arange(a.shape[-1])
    logs[..., idx, idx] = 0.0
    usable[..., idx, idx] = True
    counts = usable.sum(axis=-1)
    with np.errstate(invalid="ignore"):
        level = np.where(usable, logs, 0.0).sum(axis=-1) / counts
    return np.exp(np.where(counts > 0, level, np.nan))


def arbitrage_depth(ask: np.ndarray) -> np.ndarray:
    """(N,) the deepest money pump available against each agent, per leg.

    The rate matrix is a currency exchange table, so being pumpable is the
    classic negative-cycle problem: take logs and look for a cycle summing below
    zero. Reported as a factor per leg of the cycle -- `1.0` means no cycle takes
    anything off the agent, `0.9` means the best pump skims 10% at every hop.

    Per *leg* rather than per cycle, because cycles of different lengths are
    otherwise incomparable and the total is recoverable as `factor ** length`.
    This is Karp's minimum mean cycle, which is exact and O(G^3); the obvious
    Floyd-Warshall version is not, because with a negative cycle present its
    relaxation happily walks the same cycle twice and reports a pump deeper than
    any cycle in the graph.

    Refusals are absent edges rather than expensive ones, so a good an agent
    will not accept at any rate cannot appear in a pump against it.
    """
    a = np.asarray(ask, dtype=np.float64)
    single = a.ndim == 2
    if single:
        a = a[None]
    n, g, _ = a.shape
    with np.errstate(divide="ignore"):
        w = np.log(np.maximum(a, RATE_FLOOR))
    w[~np.isfinite(a)] = np.inf
    idx = np.arange(g)
    w[:, idx, idx] = np.inf

    # d[k, i, v]: least-weight walk of exactly k edges to v, from a virtual
    # source joined to every node at zero cost -- which is what makes the
    # minimum well defined when refusals leave the graph disconnected.
    d = np.full((g + 1, n, g), np.inf)
    d[0] = 0.0
    for k in range(1, g + 1):
        d[k] = np.min(d[k - 1][:, :, None] + w, axis=1)

    with np.errstate(invalid="ignore"):
        means = (d[g][None] - d[:g]) / (g - np.arange(g))[:, None, None]
    means = np.where(np.isnan(means), -np.inf, means)   # inf - inf: no such walk
    per_node = means.max(axis=0)
    per_node[~np.isfinite(d[g])] = np.inf               # v not on any g-edge walk
    return np.exp(np.minimum(per_node.min(axis=1), 0.0))[0 if single else slice(None)]


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
    quoted_spread: float         # median round trip a posting demands, >= 1 is a margin
    arbitrage_depth: float       # mean per-leg factor of the best pump per agent
    n_pumpable: int              # agents whose own postings can be cycled against them
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

        # What the postings themselves say, independent of what traded. Under
        # random play trade is thin enough that realised rates barely estimate
        # anything, while every agent posts a full matrix every tick.
        ask = world.last_actions.ask if world.last_actions is not None else None
        if ask is None:
            spread, depth, pumpable = float("nan"), float("nan"), 0
        else:
            trips = round_trip(ask)
            finite = trips[np.isfinite(trips)]
            spread = float(np.median(finite)) if finite.size else float("nan")
            per_agent = arbitrage_depth(ask)
            depth = float(per_agent.mean())
            pumpable = int((per_agent < 1.0 - PUMP_TOLERANCE).sum())

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
            quoted_spread=spread,
            arbitrage_depth=depth,
            n_pumpable=pumpable,
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
