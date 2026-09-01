"""Action space: simultaneous continuous vectors.

Every agent emits all of these every tick. There is no single action slot, which
is deliberate -- when consuming, producing and offering competed for one slot,
posting an offer carried an opportunity cost equal to the best alternative, and
the result was a knife edge: a policy either never traded or traded constantly
and starved. That was an artifact of the encoding, not economics.

    consume    (N, G)     quantities to eat, clipped to inventory; +inf means all
    effort     (N, G)     effort allocation across goods, rows sum to <= 1
    ask        (N, G, G)  reservation rates, ask[i, a, b] per unit of a given up
    max_trade  (N, G)     how much of each good the agent will part with
    move       (N,)       edge index to enter, or -1

`ask` is a matrix, not a vector of valuations. `ask[i, a, b]` is the **minimum
units of `b` that agent `i` demands per unit of `a` it gives up**. The diagonal
is meaningless and never read.

Why a matrix. A price vector pinned an agent's rate one way to the exact
reciprocal of its rate the other way, so it could not demand a margin on both
sides of the same pair -- "I will sell an apple for 2 bananas but only pay 1.5"
was not expressible. Worse, shading was inverted: raising your posted apple price
made you a tougher apple *seller* and simultaneously a more eager apple *buyer*,
because both directions moved off the same number. A two-sided quote is what the
shading and acceptance-model rungs in `policy/` were always waiting on.

`ask[a, b] * ask[b, a]` is the agent's own **round trip**: convert a unit of `a`
into `b` and back at its own posted rates and it keeps `1 / product` of what it
started with. A product above 1 is a spread, which is the point. A product below
1 is an agent whose postings can be money-pumped against it, and that is legal --
see `mechanism` for why the world does not protect an agent from its own
incoherence. Longer cycles work the same way, which makes the matrix a currency
exchange table and incoherence the classic negative-cycle problem.

There is no numeraire, and so no buyer and no seller. Every agent is a barterer
posting rates of exchange; which side of a trade is "buying" depends entirely on
which good you choose to price things in, and the world chooses none.

There is no markup parameter anywhere. The posted rate is simultaneously the
agent's reservation and its ask, so shading the rate *is* the markup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NO_MOVE = -1

# The rate at which an agent is refusing rather than quoting. Any finite ceiling
# would be a rate some counterparty could meet, so this is genuinely infinite;
# `mechanism` tests feasibility as a product and never divides by it.
REFUSE = np.inf


@dataclass
class Actions:
    """One set of vectors per agent."""

    consume: np.ndarray    # (N, G) float32
    effort: np.ndarray     # (N, G) float32
    ask: np.ndarray        # (N, G, G) float32
    max_trade: np.ndarray  # (N, G) float32
    move: np.ndarray       # (N,) int32

    @classmethod
    def idle(cls, n: int, g: int) -> "Actions":
        z = lambda: np.zeros((n, g), dtype=np.float32)
        # An all-zero ask matrix is not a neutral posting -- zero means "I will
        # give this away for nothing". Idle refuses every pair, which crosses with
        # nothing, so an agent that never sets a rate never trades by accident.
        return cls(consume=z(), effort=z(),
                   ask=np.full((n, g, g), REFUSE, dtype=np.float32),
                   max_trade=z(), move=np.full(n, NO_MOVE, dtype=np.int32))

    def __len__(self) -> int:
        return len(self.move)

    def copy(self) -> "Actions":
        return Actions(self.consume.copy(), self.effort.copy(), self.ask.copy(),
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
        # +inf is meaningful for `ask` too, and for the same reason inverted:
        # it is a refusal to trade that pair at any rate. Flattening it to zero
        # would turn "never" into "take it for free", which is the most
        # expensive possible misreading.
        out.ask = np.nan_to_num(out.ask, nan=REFUSE, posinf=REFUSE, neginf=0.0)
        out.max_trade = np.nan_to_num(out.max_trade, nan=0.0, posinf=np.inf, neginf=0.0)

        np.clip(out.consume, 0.0, None, out=out.consume)
        np.clip(out.effort, 0.0, None, out=out.effort)
        np.clip(out.ask, 0.0, None, out=out.ask)
        np.clip(out.max_trade, 0.0, None, out=out.max_trade)

        if not self.consume:
            out.consume[:] = 0.0
        if not self.produce:
            out.effort[:] = 0.0
        if not self.trade:
            out.max_trade[:] = 0.0
            out.ask[:] = REFUSE

        # An agent cannot eat or sell what it does not hold.
        out.consume = np.minimum(out.consume, world.inventory)
        out.max_trade = np.minimum(out.max_trade, world.inventory)

        # One tick buys one unit of effort; over-allocation is scaled down
        # rather than refused, so the split is kept and only the total is capped.
        # The diagonal is a rate for trading a good against itself. It is never
        # read, and pinning it here stops a policy's stray value showing up in
        # a metric that sweeps the matrix.
        idx = np.arange(g)
        out.ask[:, idx, idx] = REFUSE

        total = out.effort.sum(axis=1, keepdims=True)
        over = (total > 1.0).ravel()
        if over.any():
            out.effort[over] /= total[over]

        # In transit: no trading, producing or consuming, and no new move.
        out.consume[~settled] = 0.0
        out.effort[~settled] = 0.0
        out.max_trade[~settled] = 0.0
        out.ask[~settled] = REFUSE
        out.move[~settled] = NO_MOVE

        if not self.move:
            out.move[:] = NO_MOVE
        else:
            bad = np.zeros(n, dtype=bool)
            for i in np.flatnonzero((out.move >= 0) & settled):
                bad[i] = out.move[i] not in world.topology.out_edges(world.region[i])
            out.move[bad] = NO_MOVE
        return out
