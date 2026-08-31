"""Exchange mechanism: the meeting rule and the execution rule.

Bilateral, discriminatory (pay-as-bid), rationed by rate priority. Each region
carries a board of posted prices; an agent sees `sight_i` of them per tick, so
matching stays local and incomplete.

**There is no solver.** A routine that takes the board and maximises total
surplus is a Walrasian auctioneer: it dissolves the search friction that a medium
of exchange exists to solve, and makes the allocation stipulated rather than
emergent. Trades are found pairwise and filled greedily in surplus order, so the
best rate fills first and the next best takes the remainder -- which is what
pressures agents to shade their prices against rivals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecognomy.actions import Actions

# Prices are floored only where a division needs them. Surplus is always scored
# with the true posted price, so a good an agent genuinely values at zero
# contributes exactly zero however large the exchange rate is. Flooring before
# scoring instead lets `eps * rate` masquerade as gain: with a rate of ~2.6e4 it
# produced spurious surplus large enough to make dead scenarios look alive.
PRICE_FLOOR = 1e-12


def _normalised(price: np.ndarray) -> np.ndarray:
    """A posting rescaled so its largest entry is 1. Ratios are untouched."""
    p = price.astype(np.float64)
    top = p.max()
    return p / top if top > 0 else p


@dataclass(frozen=True)
class Trade:
    """One executed exchange. `price` is units of good_b per unit of good_a."""

    agent_a: int
    agent_b: int
    good_a: int
    good_b: int
    qty_a: float
    qty_b: float
    price: float
    region: int


@dataclass
class BilateralMechanism:
    """Sampled bilateral matching over posted price vectors.

    For a meeting pair, the best trade is the argmax over the G x G matrix of
    joint surplus. Agent i gives `a` and receives `b` at rate `r` = b per a:

        r      = sqrt( (p_i[a]/p_i[b]) * (p_j[a]/p_j[b]) )     geometric mean
        du_i   = p_i[b] * r - p_i[a]     per unit of a given
        du_j   = p_j[a] - p_j[b] * r     per unit of a received

    Both must be strictly positive, so neither side is ever handed a trade it is
    merely indifferent to. The geometric mean is used because it is invariant
    under relabelling which good is "a"; an arithmetic midpoint is not, and would
    quietly favour whichever good the implementation indexed first.
    """

    min_surplus: float = 0.0

    def run(self, world, actions: Actions, rng: np.random.Generator) -> list[Trade]:
        # Who saw whom this tick, for the viewer. Sight is resampled every tick,
        # so this cannot be reconstructed after the fact from `world.sight`.
        world.last_visibility = np.zeros((world.n_agents, world.n_agents), dtype=bool)
        settled = np.flatnonzero((world.region >= 0) & (actions.max_trade.sum(axis=1) > 0))
        if settled.size < 2:
            return []

        candidates = []
        for region in range(world.topology.n_regions):
            here = settled[world.region[settled] == region]
            if here.size < 2:
                continue
            candidates += self._region_candidates(world, actions, here, region, rng)

        # Best trades clear first, so inventory is not consumed by marginal ones.
        # Ties are broken by a seeded shuffle before the sort, never by agent id,
        # which would hand a permanent structural advantage to low-numbered agents.
        rng.shuffle(candidates)
        candidates.sort(key=lambda c: -c[0])
        return self._execute(world, actions, candidates)

    def _region_candidates(self, world, actions, here, region, rng):
        """Every gainful pairing among agents who can see each other."""
        out = []
        seen: set[tuple[int, int]] = set()
        for i in here:
            others = here[here != i]
            if others.size == 0:
                continue
            k = min(int(world.sight[i]), others.size)
            for j in rng.choice(others, size=k, replace=False):
                world.last_visibility[int(i), int(j)] = True
                key = (min(int(i), int(j)), max(int(i), int(j)))
                if key in seen:
                    continue
                seen.add(key)
                found = self._best_trade(world, actions, key[0], key[1])
                if found is not None:
                    surplus, a, b, rate = found
                    out.append((surplus, key[0], key[1], a, b, rate, region))
        return out

    def _best_trade(self, world, actions, i: int, j: int):
        # Normalise each posting to a common scale before scoring.
        #
        # A price vector means the same thing multiplied by any positive
        # constant -- (1, 6, 2) and (1000, 6000, 2000) imply identical rates.
        # But `du` is linear in the posting, so without this an agent could
        # multiply its whole vector by 1000, change nothing about the trade it
        # is willing to make, and inflate its joint surplus 500-fold, buying the
        # front of the greedy queue for free. Normalising by the max (not the
        # geometric mean, which the price floor stops scaling linearly) makes
        # the ranking depend on how much the two agents *disagree*, which is the
        # real signal, rather than on who posts the largest numbers.
        true_i = _normalised(actions.price[i])
        true_j = _normalised(actions.price[j])
        f_i = np.maximum(true_i, PRICE_FLOOR)
        f_j = np.maximum(true_j, PRICE_FLOOR)

        rate = np.sqrt((f_i[:, None] / f_i[None, :]) * (f_j[:, None] / f_j[None, :]))
        du_i = true_i[None, :] * rate - true_i[:, None]   # (a, b), per unit of a
        du_j = true_j[:, None] - true_j[None, :] * rate

        avail_i = np.minimum(actions.max_trade[i], world.inventory[i]).astype(np.float64)
        avail_j = np.minimum(actions.max_trade[j], world.inventory[j]).astype(np.float64)
        qty = np.minimum(avail_i[:, None], avail_j[None, :] / np.maximum(rate, PRICE_FLOOR))

        ok = (du_i > self.min_surplus) & (du_j > self.min_surplus) & (qty > 1e-9)
        np.fill_diagonal(ok, False)
        if not ok.any():
            return None
        joint = np.where(ok, (du_i + du_j) * qty, -np.inf)
        a, b = np.unravel_index(np.argmax(joint), joint.shape)
        return float(joint[a, b]), int(a), int(b), float(rate[a, b])

    def _execute(self, world, actions, candidates) -> list[Trade]:
        """Fill greedily, decrementing both sides' budgets so nothing double-spends."""
        remaining = np.minimum(actions.max_trade, world.inventory).astype(np.float64)
        trades: list[Trade] = []
        for _, i, j, a, b, rate, region in candidates:
            qty_a = min(remaining[i, a], remaining[j, b] / max(rate, PRICE_FLOOR),
                        float(world.inventory[i, a]))
            qty_b = qty_a * rate
            if qty_a <= 1e-9 or qty_b <= 1e-9 or qty_b > world.inventory[j, b]:
                continue

            world.inventory[i, a] -= qty_a
            world.inventory[j, a] += qty_a
            world.inventory[j, b] -= qty_b
            world.inventory[i, b] += qty_b
            remaining[i, a] -= qty_a
            remaining[j, b] -= qty_b

            trades.append(Trade(agent_a=i, agent_b=j, good_a=a, good_b=b,
                                qty_a=float(qty_a), qty_b=float(qty_b),
                                price=float(rate), region=int(region)))
        return trades
