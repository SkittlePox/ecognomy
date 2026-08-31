"""Rung 1: myopic rationality, with no learned parameters.

The agent knows its own preferences, substitution parameter, production
efficiency and inventory, and nothing whatever about anyone else. Under the
vector action space it emits all of its decisions at once:

    price      its own marginal valuation of each good, posted honestly
    consume    the closed-form optimal fraction of its holdings
    effort     all of it on the good with the highest marginal value
    max_trade  a fixed fraction of each holding

**There is no markup parameter.** The posted price is simultaneously the
valuation and the ask, so a rung that posts honestly has nothing left to tune.
Shading prices to capture more of the surplus means overstating what you hold and
understating what you want, and choosing how far to shade requires knowing what
rivals post -- which is rung 3, not this one.

What this rung structurally cannot do:

  * It never accepts a good it does not consume, because it prices such a good at
    zero and the mechanism requires strictly positive surplus on both sides. That
    rules out indirect exchange, money, and arbitrage -- all the same behaviour.
  * It cannot choose where to move, having no information about other regions.
  * It posts honest prices, so it captures only the geometric-mean split of each
    trade rather than competing for a larger share.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecognomy.actions import NO_MOVE, Actions
from ecognomy.utility import marginal_utility

_EPS = 1e-12


@dataclass
class MyopicPolicy:
    """One-step utility maximisation against the agent's own preferences.

    Args:
        kappa: value of a unit of stock utility against a unit of consumed
            utility. Above 1 the agent is patient and accumulates; below 1 it is
            impatient and eats down its holdings.
        max_trade_fraction: share of each holding the agent will part with per
            tick. Caps trades so the marginal valuation posted in `price` stays a
            reasonable approximation over the quantity actually exchanged.
        explore_move: per-tick probability of a random relocation. Undirected,
            because this rung has no basis for preferring one region.
    """

    kappa: float = 3.0
    max_trade_fraction: float = 0.5
    explore_move: float = 0.02

    def act(self, world, rng: np.random.Generator) -> Actions:
        n, g = world.n_agents, world.n_goods
        inv = world.inventory.astype(np.float64)
        actions = Actions.idle(n, g)

        # --- price: honest marginal valuation, which is the whole strategy here
        price = marginal_utility(inv, world.theta, world.rho, world.alpha)
        actions.price = np.maximum(price, 0.0).astype(np.float32)

        # --- consume: closed-form optimum, no search
        #
        # CES times alpha is homogeneous of degree alpha, so consuming fraction f
        # of a holding worth U scores  U * [ f**a + kappa*((1-f)**a - 1) ].
        # Setting the derivative to zero gives f = z/(1+z) with z = kappa**(1/(a-1)),
        # which is exact and costs nothing to evaluate.
        alpha = np.clip(world.alpha.astype(np.float64), 1e-3, 1.0 - 1e-3)
        z = np.power(max(self.kappa, _EPS), 1.0 / (alpha - 1.0))
        frac = (z / (1.0 + z))[:, None]
        actions.consume = (inv * frac).astype(np.float32)

        # --- effort: the action space permits splitting a tick across goods,
        # but production is *linear* in effort within a tick, so splitting never
        # beats going all-in and the optimum is always a corner. A split becomes
        # rational only with diminishing returns inside the tick (effort**beta,
        # beta < 1), which the environment does not currently impose.
        stock = world.stock[np.clip(world.region, 0, None)]
        yield_per_tick = np.minimum(world.efficiency.astype(np.float64), stock)
        value = actions.price.astype(np.float64) * yield_per_tick
        best = np.argmax(value, axis=1)
        rows = np.arange(n)
        worth_it = value[rows, best] > world.config.production.effort_cost
        effort = np.zeros((n, g), dtype=np.float32)
        effort[rows[worth_it], best[worth_it]] = 1.0
        actions.effort = effort

        # --- max_trade: a bounded slice of each holding
        actions.max_trade = (inv * self.max_trade_fraction).astype(np.float32)

        # --- move: undirected exploration
        movers = (rng.random(n) < self.explore_move) & (world.region >= 0)
        for i in np.flatnonzero(movers):
            out = world.topology.out_edges(world.region[i])
            if out.size:
                actions.move[i] = int(rng.choice(out))

        # An agent crossing an edge can do nothing but arrive. Zeroing here
        # rather than leaving it to `sanitize` keeps the policy's proposals
        # legal by construction, so a mismatch after sanitising is a real bug
        # rather than expected tidying.
        in_transit = world.region < 0
        actions.consume[in_transit] = 0.0
        actions.effort[in_transit] = 0.0
        actions.price[in_transit] = 0.0
        actions.max_trade[in_transit] = 0.0
        return actions
