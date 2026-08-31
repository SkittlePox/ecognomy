"""World state and the tick transition.

State is arrays over the whole population. `world` never imports `policy`;
policies return actions and the world applies them, in the phase order fixed in
`docs/environment.md`.
"""

from __future__ import annotations

import numpy as np

from ecognomy.actions import NO_MOVE, Actions, ActionSpace
from ecognomy.config import WorldConfig
from ecognomy.mechanism import BilateralMechanism, Trade
from ecognomy.utility import utility

IN_TRANSIT = -1


class World:
    """A population of agents in a region graph.

    Agents are either settled in a region (`region[i] >= 0`) or in transit along
    an edge (`region[i] == IN_TRANSIT`, `edge[i] >= 0`). In transit they hold
    inventory, occupy edge capacity, and can do nothing but arrive.
    """

    def __init__(self, config: WorldConfig, mechanism: BilateralMechanism | None = None,
                 action_space: ActionSpace | None = None, scenario=None) -> None:
        self.config = config
        self.topology = config.topology
        self.n_agents = config.n_agents
        self.n_goods = config.n_goods
        self.rng = np.random.default_rng(config.seed)
        self.mechanism = mechanism or BilateralMechanism(config.market.min_surplus)
        self.action_space = action_space or ActionSpace()

        self.t = 0
        self._spawn()
        if scenario is not None:
            # A diagnostic scenario replaces every sampled attribute with exact
            # values, so the world under test is fully specified.
            scenario.apply(self)

        self.last_trades: list[Trade] = []
        self.last_reward = np.zeros(self.n_agents, dtype=np.float32)
        self.last_consumption = np.zeros((self.n_agents, self.n_goods), dtype=np.float32)
        self.last_production = np.zeros((self.n_agents, self.n_goods), dtype=np.float32)
        self.last_illegal = np.zeros(self.n_agents, dtype=bool)
        self.last_actions = Actions.idle(self.n_agents, self.n_goods)
        self.last_visibility = np.zeros((self.n_agents, self.n_agents), dtype=bool)
        self.goods_destroyed = 0.0
        self.goods_created = 0.0
        # Running totals. `welfare` is net of the effort and travel spent getting
        # it; `consumption_utility` is the gross pleasure before those costs.
        # Net is what a rational agent maximises and what the designer is tuning.
        self.cumulative_reward = np.zeros(self.n_agents, dtype=np.float64)
        self.cumulative_utility = np.zeros(self.n_agents, dtype=np.float64)
        self.last_utility = np.zeros(self.n_agents, dtype=np.float32)

    # ---------------------------------------------------------------- spawn

    def _spawn(self) -> None:
        cfg, rng, n, g = self.config, self.rng, self.n_agents, self.n_goods

        # Tastes.
        self.theta = rng.dirichlet(
            np.full(g, cfg.preference.dirichlet_concentration), size=n
        ).astype(np.float32)
        self.theta_base = self.theta.copy()


        # Production efficiency. Scale makes agents uniformly better or worse and
        # creates no comparative advantage; shape varies the *ranking* of goods
        # within an agent, which does. A world with shape_spread == 0 has no
        # gains from trade on the production side, whatever else is set.
        scale = np.exp(cfg.production.scale_spread * rng.standard_normal((n, 1)))
        shape = np.exp(cfg.production.shape_spread * rng.standard_normal((n, g)))
        shape /= shape.mean(axis=1, keepdims=True)
        efficiency = cfg.production.efficiency_mean * scale * shape
        k = cfg.production.n_producible
        if k is not None and k < g:
            # Keep each agent's top-k goods and zero the rest, so most goods
            # must be obtained from someone else rather than made.
            cutoff = np.partition(efficiency, g - k, axis=1)[:, g - k][:, None]
            efficiency = np.where(efficiency >= cutoff, efficiency, 0.0)
        self.efficiency = efficiency.astype(np.float32)

        self.mobility = np.maximum(
            cfg.mobility.mobility_mean * np.exp(cfg.mobility.mobility_spread * rng.standard_normal(n)),
            1e-3,
        ).astype(np.float32)

        # How many of the region's posted prices each agent can see. Drawn
        # heterogeneously: every agent has a different K, so market access is a
        # capability that differs across the population rather than a constant.
        vis = cfg.visibility
        if vis.sight_mean <= 0:
            self.sight = np.zeros(n, dtype=np.int32)
        else:
            drawn = vis.sight_mean * np.exp(vis.sight_spread * rng.standard_normal(n))
            self.sight = np.maximum(1, np.rint(drawn)).astype(np.int32)

        # Placement and stocks.
        self.region = rng.integers(0, self.topology.n_regions, size=n).astype(np.int32)
        self.edge = np.full(n, -1, dtype=np.int32)
        self.progress = np.zeros(n, dtype=np.float32)
        self.inventory = np.full((n, g), cfg.initial_inventory, dtype=np.float32)
        self.stock = np.full(
            (self.topology.n_regions, g), cfg.resource.stock_capacity, dtype=np.float32
        )

    # ----------------------------------------------------------------- tick

    def step(self, actions: Actions) -> np.ndarray:
        """Advance one tick. Returns (N,) per-agent reward."""
        actions = self.action_space.sanitize(self, actions)
        self.last_actions = actions
        reward = np.zeros(self.n_agents, dtype=np.float32)
        self.last_consumption[:] = 0.0
        self.last_production[:] = 0.0

        self._phase_transit(actions, reward)
        self._phase_produce(actions, reward)
        self.last_trades = self.mechanism.run(self, actions, self.rng)
        self._phase_consume(actions, reward)
        self._phase_spoil()
        self._phase_regen()
        self._phase_anneal()

        self.t += 1
        self.last_reward = reward
        self.cumulative_reward += reward
        self.cumulative_utility += self.last_utility
        return reward

    def _phase_transit(self, actions: Actions, reward: np.ndarray) -> None:
        """Advance in-transit agents, then admit new traversals under capacity."""
        moving = np.flatnonzero(self.region == IN_TRANSIT)
        self.progress[moving] += self.mobility[moving]
        arrived = moving[self.progress[moving] >= self.topology.weight[self.edge[moving]]]
        self.region[arrived] = self.topology.dst[self.edge[arrived]]
        self.edge[arrived] = -1
        self.progress[arrived] = 0.0

        still_moving = self.region == IN_TRANSIT
        reward[still_moving] -= self.config.mobility.travel_cost_per_tick

        occupancy = np.bincount(self.edge[still_moving], minlength=self.topology.n_edges)
        for i in np.flatnonzero(actions.move != NO_MOVE):
            e = int(actions.move[i])
            if occupancy[e] >= self.topology.capacity[e]:
                continue  # chokepoint refuses; the move becomes a no-op
            occupancy[e] += 1
            reward[i] -= self.config.mobility.travel_cost_per_tick
            # Progress accrues on the tick of entry, so crossing a distance-d
            # edge at mobility m takes ceil(d/m) ticks. An agent fast enough to
            # cover the whole edge in one tick arrives without ever being in
            # transit -- which is the payoff for high mobility on short hops.
            self.progress[i] = self.mobility[i]
            if self.progress[i] >= self.topology.weight[e]:
                self.region[i] = self.topology.dst[e]
                self.edge[i] = -1
                self.progress[i] = 0.0
            else:
                self.region[i] = IN_TRANSIT
                self.edge[i] = e

    def _phase_produce(self, actions: Actions, reward: np.ndarray) -> None:
        """Create goods from an effort allocation, rationed against regional stock.

        Effort is a vector summing to at most one, so an agent may split a tick
        across several goods. This is the one genuinely economic tradeoff the old
        single-action encoding got right: time is scarce even though posting a
        price is not.

        **When a region's stock cannot meet demand it is shared pro rata**, every
        claimant scaled by the same factor. The obvious implementation -- loop
        over agents, each drawing from what the last one left -- makes production
        first-come-first-served by agent index, which is a permanent structural
        advantage to low ids rather than anything economic. It showed up as two
        identical agents diverging forever the moment a resource ran short.
        Randomising the order would only make the unfairness fluctuate; sharing
        the shortfall removes it, and is order-independent by construction.

        Effort is charged in proportion to what was actually produced, so an
        agent rationed down to nothing pays nothing.
        """
        active = (self.region >= 0) & (actions.effort.sum(axis=1) > 0)
        if not active.any():
            return

        want = actions.effort.astype(np.float64) * self.efficiency.astype(np.float64)
        want[~active] = 0.0

        demand = np.zeros_like(self.stock, dtype=np.float64)
        np.add.at(demand, self.region[active], want[active])
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(demand > 1e-12,
                             np.minimum(1.0, self.stock / np.maximum(demand, 1e-12)), 0.0)

        granted = want * share[np.clip(self.region, 0, None)]
        granted[~active] = 0.0

        np.subtract.at(self.stock, self.region[active], granted[active])
        np.clip(self.stock, 0.0, None, out=self.stock)

        self.inventory += granted.astype(np.float32)
        self.last_production += granted.astype(np.float32)
        self.goods_created += float(granted.sum())

        with np.errstate(divide="ignore", invalid="ignore"):
            fulfilled = np.where(want.sum(axis=1) > 1e-12,
                                 granted.sum(axis=1) / np.maximum(want.sum(axis=1), 1e-12), 0.0)
        reward -= (self.config.production.effort_cost
                   * actions.effort.sum(axis=1) * fulfilled).astype(np.float32)

    def _phase_consume(self, actions: Actions, reward: np.ndarray) -> None:
        """Voluntary consumption. Nothing forces an agent to eat, which is what
        leaves hoarding and cornering available as strategies."""
        # A full bundle. Reward is linear in each good, so what matters is only
        # how much of each is eaten, never the mix.
        consumed = np.minimum(actions.consume, self.inventory).astype(np.float32)
        self.inventory -= consumed
        self.last_consumption = consumed
        self.goods_destroyed += float(consumed.sum())
        gained = utility(consumed, self.theta)
        self.last_utility = gained
        reward += gained

    def _phase_spoil(self) -> None:
        """Decay held goods. `spoilage` is the hoardability knob: a low-spoilage
        good is durable, corner-able, and a plausible store of value."""
        delta = self.config.spoilage_array()
        lost = self.inventory * delta
        self.inventory -= lost
        self.goods_destroyed += float(lost.sum())

    def _phase_regen(self) -> None:
        cap, rate = self.config.resource.stock_capacity, self.config.resource.regen_rate
        self.stock += rate * (cap - self.stock)
        np.clip(self.stock, 0.0, cap, out=self.stock)

    def _phase_anneal(self) -> None:
        """Drive the token's consumption weight toward `anneal_end`.

        This is the headline experiment: establish acceptance while the token is
        a genuine commodity, then remove its intrinsic value and measure whether
        acceptance survives on expectation alone.
        """
        tok = self.config.token
        if tok.anneal_ticks <= 0:
            return
        frac = (self.t - tok.anneal_start_tick) / tok.anneal_ticks
        frac = float(np.clip(frac, 0.0, 1.0))
        g = tok.token_good
        self.theta[:, g] = (1 - frac) * self.theta_base[:, g] + frac * tok.anneal_end

    # ------------------------------------------------------------- readouts

    @property
    def settled(self) -> np.ndarray:
        return self.region >= 0

    @property
    def welfare(self) -> float:
        """Total realised reward across the population, net of effort and travel.

        The designer's score. Compare it against the same world with the market
        switched off: the difference is what trade contributed, as distinct from
        what production contributed.
        """
        return float(self.cumulative_reward.sum())

    def token_weight(self) -> float:
        return float(self.theta[:, self.config.token.token_good].mean())

    def agents_in(self, region: int) -> np.ndarray:
        return np.flatnonzero(self.region == region)
