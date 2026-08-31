"""Action space: simultaneous continuous vectors.

Every agent emits all of these every tick. There is no single action slot, which
is deliberate -- when consuming, producing and offering competed for one slot,
posting an offer carried an opportunity cost equal to the best alternative, and
the result was a knife edge: a policy either never traded or traded constantly
and starved. That was an artifact of the encoding, not economics.

    consume    (N, G)  quantities to eat, clipped to inventory; +inf means all
    effort     (N, G)  effort allocation across goods, rows sum to <= 1
    price      (N, G)  subjective valuation, meaningful only up to scale
    max_trade  (N, G)  how much of each good the agent will part with
    move       (N,)    edge index to enter, or -1

`price` is a vector, not a G x G matrix of ratios. The rate between `a` and `b`
is implied as `p_a / p_b`. That is G numbers rather than G**2, and it makes
cross-rates automatically consistent, so an agent cannot be arbitraged by its own
incoherence -- arbitrage should come from different agents disagreeing and from
spatial separation, not from one agent contradicting itself.

There is no markup parameter anywhere. The posted price is simultaneously the
agent's valuation and its ask, so shading the price *is* the markup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NO_MOVE = -1


@dataclass
class Actions:
    """One set of vectors per agent."""

    consume: np.ndarray    # (N, G) float32
    effort: np.ndarray     # (N, G) float32
    price: np.ndarray      # (N, G) float32
    max_trade: np.ndarray  # (N, G) float32
    move: np.ndarray       # (N,) int32

    @classmethod
    def idle(cls, n: int, g: int) -> "Actions":
        z = lambda: np.zeros((n, g), dtype=np.float32)
        return cls(consume=z(), effort=z(), price=z(), max_trade=z(),
                   move=np.full(n, NO_MOVE, dtype=np.int32))

    def __len__(self) -> int:
        return len(self.move)

    def copy(self) -> "Actions":
        return Actions(self.consume.copy(), self.effort.copy(), self.price.copy(),
                       self.max_trade.copy(), self.move.copy())


@dataclass(frozen=True)
class ActionSpace:
    """Which output heads are live, and how proposals are made legal.

    Legality is clipping rather than rejection. The previous discrete space
    rejected malformed actions and silently substituted IDLE, which hid a real
    bug for a while: consume actions arrived with a zero quantity, were rejected,
    and agents produced without ever eating. Clipping cannot fail that way -- a
    nonsensical proposal becomes a harmless one, not a different one.

    Disabling a head is how ablations are run: a world without `trade` is the
    autarky control, one without `move` is a single-region world.
    """

    consume: bool = True
    produce: bool = True
    trade: bool = True
    move: bool = True

    def sanitize(self, world, actions: Actions) -> Actions:
        n, g = world.n_agents, world.n_goods
        out = actions.copy()
        settled = world.region >= 0

        # +inf is meaningful for consume and max_trade: it says "all of it",
        # which the clip to inventory below resolves exactly. Flattening it to
        # zero here would silently turn "eat everything" into "eat nothing".
        out.consume = np.nan_to_num(out.consume, nan=0.0, posinf=np.inf, neginf=0.0)
        out.effort = np.nan_to_num(out.effort, nan=0.0, posinf=0.0, neginf=0.0)
        out.price = np.nan_to_num(out.price, nan=0.0, posinf=0.0, neginf=0.0)
        out.max_trade = np.nan_to_num(out.max_trade, nan=0.0, posinf=np.inf, neginf=0.0)

        np.clip(out.consume, 0.0, None, out=out.consume)
        np.clip(out.effort, 0.0, None, out=out.effort)
        np.clip(out.price, 0.0, None, out=out.price)
        np.clip(out.max_trade, 0.0, None, out=out.max_trade)

        if not self.consume:
            out.consume[:] = 0.0
        if not self.produce:
            out.effort[:] = 0.0
        if not self.trade:
            out.max_trade[:] = 0.0

        # An agent cannot eat or sell what it does not hold.
        out.consume = np.minimum(out.consume, world.inventory)
        out.max_trade = np.minimum(out.max_trade, world.inventory)

        # One tick buys one unit of effort; over-allocation is scaled down
        # rather than refused, so the split is kept and only the total is capped.
        total = out.effort.sum(axis=1, keepdims=True)
        over = (total > 1.0).ravel()
        if over.any():
            out.effort[over] /= total[over]

        # In transit: no trading, producing or consuming, and no new move.
        out.consume[~settled] = 0.0
        out.effort[~settled] = 0.0
        out.max_trade[~settled] = 0.0
        out.price[~settled] = 0.0
        out.move[~settled] = NO_MOVE

        if not self.move:
            out.move[:] = NO_MOVE
        else:
            bad = np.zeros(n, dtype=bool)
            for i in np.flatnonzero((out.move >= 0) & settled):
                bad[i] = out.move[i] not in world.topology.out_edges(world.region[i])
            out.move[bad] = NO_MOVE
        return out
