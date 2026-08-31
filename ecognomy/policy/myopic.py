"""Rung 1: myopic rationality, with no learned parameters.

Under a linear reward the myopic policy is almost trivial, and that is a feature
-- there is nothing left to tune, so anything the population does is a property
of the world rather than of a hand-set constant.

    price      its own preference weights, posted honestly. A good's value never
               changes with how much is held, so this is exact at any quantity
               and the mechanism can never approve a trade that hurts it.
    max_trade  everything it holds. Trade resolves before consumption within a
               tick, and every executed trade must raise both sides' posted
               value, so offering the lot can only improve the basket.
    consume    whatever survives trading. A good is worth the same now as later
               while spoilage taxes holding it, so waiting is never better.
    effort     all of it on the good with the highest preference x efficiency,
               since production is linear in effort and the optimum is a corner.

What this rung structurally cannot do:

  * It never accepts a good it does not consume: it prices such a good at zero
    and the mechanism requires strictly positive surplus on both sides. That
    rules out indirect exchange, money and arbitrage -- the same behaviour under
    three names -- which is exactly what the `triangular` scenario detects.
  * It never holds anything for later, so it cannot corner a market.
  * It cannot choose where to move, having no information about other regions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecognomy.actions import Actions
from ecognomy.utility import marginal_value


@dataclass
class MyopicPolicy:
    """Immediate value maximisation against the agent's own preferences.

    Args:
        explore_move: per-tick probability of a random relocation. Undirected,
            because this rung has no basis for preferring one region.
    """

    explore_move: float = 0.02

    def act(self, world, rng: np.random.Generator) -> Actions:
        n, g = world.n_agents, world.n_goods
        inv = world.inventory.astype(np.float64)
        actions = Actions.idle(n, g)

        actions.price = marginal_value(world.theta).astype(np.float32)
        actions.max_trade = inv.astype(np.float32)
        # Clipped to inventory by the world, after trading has resolved.
        actions.consume = np.full((n, g), np.inf, dtype=np.float32)

        value = world.theta.astype(np.float64) * world.efficiency.astype(np.float64)
        best = np.argmax(value, axis=1)
        rows = np.arange(n)
        worth_it = value[rows, best] > world.config.production.effort_cost
        effort = np.zeros((n, g), dtype=np.float32)
        effort[rows[worth_it], best[worth_it]] = 1.0
        actions.effort = effort

        movers = (rng.random(n) < self.explore_move) & (world.region >= 0)
        for i in np.flatnonzero(movers):
            out = world.topology.out_edges(world.region[i])
            if out.size:
                actions.move[i] = int(rng.choice(out))

        # An agent crossing an edge can do nothing but arrive.
        in_transit = world.region < 0
        actions.consume[in_transit] = 0.0
        actions.effort[in_transit] = 0.0
        actions.price[in_transit] = 0.0
        actions.max_trade[in_transit] = 0.0
        return actions
