"""The control.

Random vectors. Its purpose is to establish a floor: a world that cannot be
traded in by random prices is broken, and a world that produces a healthy economy
under random play has assumed its own answer.

It posts prices unrelated to its own preferences, so it accepts trades that make
it worse off. That is correct for a control and is worth keeping -- a mechanism
that protected an irrational agent from its own offers would be doing the agents'
reasoning for them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecognomy.actions import Actions


@dataclass
class RandomPolicy:
    """Random legal vectors. A control, not a contender."""

    move_prob: float = 0.1
    produce_prob: float = 0.5
    consume_scale: float = 0.4

    def act(self, world, rng: np.random.Generator) -> Actions:
        n, g = world.n_agents, world.n_goods
        inv = world.inventory.astype(np.float64)
        a = Actions.idle(n, g)

        a.price = rng.uniform(0.1, 1.0, size=(n, g)).astype(np.float32)
        a.consume = (inv * rng.uniform(0.0, self.consume_scale, size=(n, g))).astype(np.float32)
        a.max_trade = (inv * rng.uniform(0.0, 1.0, size=(n, g))).astype(np.float32)

        effort = np.zeros((n, g), dtype=np.float32)
        producing = rng.random(n) < self.produce_prob
        picks = rng.integers(0, g, size=n)
        effort[np.flatnonzero(producing), picks[producing]] = 1.0
        a.effort = effort

        movers = (rng.random(n) < self.move_prob) & (world.region >= 0)
        for i in np.flatnonzero(movers):
            out = world.topology.out_edges(world.region[i])
            if out.size:
                a.move[i] = int(rng.choice(out))

        in_transit = world.region < 0
        a.consume[in_transit] = 0.0
        a.effort[in_transit] = 0.0
        a.price[in_transit] = 0.0
        a.max_trade[in_transit] = 0.0
        return a
