"""Exchange mechanism: the meeting rule and the execution rule.

Bilateral, rationed by rate priority. Each region carries a board of posted rate
matrices; an agent sees `sight_i` of them per tick, so matching stays local and
incomplete.

**There is no solver.** A routine that takes the board and maximises total
surplus is a Walrasian auctioneer: it dissolves the search friction that a medium
of exchange exists to solve, and makes the allocation stipulated rather than
emergent. Trades are found pairwise and filled greedily in rate order, so the
best rate fills first and the next best takes the remainder -- which is what
pressures agents to shade their rates against rivals.

**There is no order book either, and that is not a matter of taste.** A book
needs a quote currency, and this is a barter economy: G goods, G(G-1)/2 pairs,
no privileged one. Whether a numeraire emerges *is* the headline experiment, so
installing a book would presuppose the answer to the question the sandbox exists
to ask.

## The rule

Agent `i` gives good `a` and receives good `b` at rate `r` (units of b per a).
Both sides posted a reservation matrix, so:

    i accepts iff  r >= ask_i[a, b]
    j accepts iff  1/r >= ask_j[b, a]

which leaves a bargaining interval, non-empty exactly when

    cross:  ask_i[a, b] * ask_j[b, a] < 1

Read with a numeraire this is "the bid crosses the ask", but the product form
needs no numeraire and so is the one the code uses. Execution splits the
interval geometrically, and the gain factor is the same number for both sides:

    rate    r = sqrt( ask_i[a, b] / ask_j[b, a] )
    depth   w = 1 / sqrt( ask_i[a, b] * ask_j[b, a] )      > 1 whenever crossed

`r = ask_i[a, b] * w`: each side receives its own reservation demand multiplied
by the depth, which is why one global sort by `w` simultaneously serves every
agent's private ranking over the competing uses of its goods.

**The geometric mean is forced, not chosen.** The general rule `r = lo^(1-k) *
hi^k` must give the reciprocal rate when the same trade is written in the other
good's units; that requires `(hi/lo)^(1-k) = (hi/lo)^k`, so `k = 1/2`. Every
other split -- pay-as-bid included -- needs a nominated numeraire, which a barter
economy does not have.

**Buying queue priority always costs terms of trade.** Since `r = sqrt(ask_i /
ask_j)`, an agent that lowers its ask to deepen the cross lowers its own received
rate by exactly the same square root. There is no setting where aggression is
free, so escalation is self-limiting rather than merely discouraged.

**Incoherent postings are legal.** An agent whose round trip `ask[a,b] *
ask[b,a]` falls below 1 can be money-pumped, and the world does not stop it --
the mechanism guarantees both sides gain *in posted terms*, never in true
utility, exactly as it has always let `RandomPolicy` trade itself poorer.
Protecting an agent from its own postings would be doing its reasoning for it.
`metrics` measures the exposure instead, over cycles of any length.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecognomy.actions import Actions

# Guards the division that forms an exchange rate, never the crossing test. A
# posting of zero -- "any positive amount will do" -- is a real statement an
# agent may make, and it is scored as written: the depth it implies is genuinely
# unbounded, because the two postings really are unboundedly far apart. What the
# floor does is keep the *rate* finite and positive so the trade can execute at
# all. Nothing here rescores a posting; the ask is the whole of what the agent
# said, so there is no true-versus-floored gap for `eps * rate` to hide in.
RATE_FLOOR = 1e-12


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
    """Sampled bilateral matching over posted rate matrices.

    A meeting contributes **every** crossing swap it finds to the queue, not just
    its deepest. Two agents who can beneficially swap apples-for-bananas *and*
    cherries-for-durians do both, which is how barter between two people who each
    hold several things the other wants actually goes.

    It used to be an argmax, and the reason to change is that "best within this
    pair" is not a meaningful rank. The queue is global: a swap that came second
    in its own meeting still takes its place against candidates from every other
    meeting, and fills if the goods survive that far. Under the argmax a good
    only ever moved if it happened to be the star of some meeting, which is
    arbitrary in a way the global ordering is not.

    `trades_per_meeting` keeps the old behaviour available as an ablation, since
    the restriction was load-bearing for how thin trade is and thinness is what
    the token experiment feeds on.
    """

    min_depth: float = 1.0
    trades_per_meeting: int = 0   # 0 = every crossing

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

        # Best rates clear first, so inventory is not consumed by marginal ones.
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
                for depth, a, b, rate in self._crossings(world, actions, *key):
                    out.append((depth, key[0], key[1], a, b, rate, region))
        return out

    def _crossings(self, world, actions, i: int, j: int):
        """Every crossing swap between two agents, deepest first.

        Sweeps all G x G directed swaps at once: cell (a, b) is `i` giving `a`
        and receiving `b`. No normalisation is needed anywhere here, unlike the
        price-vector version -- a rate matrix has no free scale, so the exploit
        where an agent multiplied its whole posting by 1000 to buy queue
        priority for nothing cannot be expressed.
        """
        ask_i = actions.ask[i].astype(np.float64)          # (a, b)
        ask_j = actions.ask[j].astype(np.float64).T        # (a, b) <- ask_j[b, a]

        # inf * 0 is the one product numpy cannot evaluate: a refusal against a
        # giveaway. It is a refusal, so it must not cross.
        with np.errstate(invalid="ignore", divide="ignore"):
            product = ask_i * ask_j
            depth = 1.0 / np.sqrt(product)
            # Both sides are floored, not just the divisor. An ask of exactly
            # zero says "any positive amount of b will do", and the geometric
            # split of the interval [0, hi] is zero -- which would hand the
            # receiver the good for literally nothing and then be dropped by the
            # quantity check, so a good an agent does not want would stop being
            # dumpable at all. Flooring the numerator keeps the rate tiny and
            # positive, which is what the statement means.
            rate = np.sqrt(np.maximum(ask_i, RATE_FLOOR)
                           / np.maximum(ask_j, RATE_FLOOR))
        # Strictness lives in `min_depth` alone, not here as well. Hard-coding
        # `product < 1` here too made every setting below 1.0 a silent no-op --
        # the knob looked like it loosened matching and did nothing. Now
        # min_depth = 1.0 is exactly "both sides must gain strictly", and below
        # that is a real ablation: matching accepts trades that lose one side
        # goods, which is a way to check the guards can still catch a permissive
        # mechanism.
        crossed = np.isfinite(product)

        avail_i = np.minimum(actions.max_trade[i], world.inventory[i]).astype(np.float64)
        avail_j = np.minimum(actions.max_trade[j], world.inventory[j]).astype(np.float64)
        qty = np.minimum(avail_i[:, None], avail_j[None, :] / np.maximum(rate, RATE_FLOOR))

        # A crossing of exactly 1.0 leaves both sides indifferent. Strictness is
        # load-bearing: it is what stops an agent being traded into a swap it
        # gains nothing from, and what keeps `triangular` a working control.
        ok = crossed & (depth > self.min_depth) & (qty > 1e-9) & np.isfinite(rate)
        np.fill_diagonal(ok, False)
        if not ok.any():
            return []

        # A pair can cross in both directions at once -- i gives `a` for `b` and
        # also gives `b` for `a` -- but only when the two round trips multiply to
        # less than 1, which takes at least one agent posting a negative spread.
        # Two coherent agents can never do it, since their round trips are both
        # exactly 1. So this is the money pump being walked rather than a double
        # count, and it is legal by the same ruling that made incoherence legal.
        rows, cols = np.nonzero(ok)
        found = [(float(depth[a, b]), int(a), int(b), float(rate[a, b]))
                 for a, b in zip(rows, cols)]
        found.sort(key=lambda c: -c[0])
        return found if self.trades_per_meeting <= 0 else found[:self.trades_per_meeting]

    def _execute(self, world, actions, candidates) -> list[Trade]:
        """Fill greedily, decrementing both sides' budgets so nothing double-spends."""
        remaining = np.minimum(actions.max_trade, world.inventory).astype(np.float64)
        trades: list[Trade] = []
        for _, i, j, a, b, rate, region in candidates:
            qty_a = min(remaining[i, a], remaining[j, b] / max(rate, RATE_FLOOR),
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
