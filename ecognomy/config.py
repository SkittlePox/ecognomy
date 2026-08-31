"""Configuration.

One nested dataclass tree. Every quantity in the environment lives here; nothing
is a module constant. The control panel binds to this directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ecognomy.topology import Topology

GOOD_NAMES: tuple[str, ...] = ("apple", "banana", "cherry", "durian", "elderberry")


@dataclass
class PreferenceConfig:
    """Tastes.

    `theta` is the whole of it. Reward is consumption times preference and
    nothing more, so there is no substitutability or curvature parameter: an
    agent's value for a good does not change with how much of it it holds.

    Low concentration gives sharply specialised tastes, which is what creates
    gains from trade -- under a linear reward those come *only* from agents
    valuing goods differently, never from their holding different amounts.
    """

    dirichlet_concentration: float = 0.6  # low => sharply specialised tastes


@dataclass
class ProductionConfig:
    """Who is good at what.

    `shape_spread` is the load-bearing one: it varies the *shape* of each
    agent's efficiency vector, which is what creates comparative advantage.
    `scale_spread` only makes agents uniformly better or worse, which creates
    none. A world with shape_spread == 0 cannot develop trade from production.
    """

    efficiency_mean: float = 1.0
    scale_spread: float = 0.3
    # How differently agents are good at things. This is the brief's "spread this
    # distribution wide", and it carries most of the reason to trade: since every
    # agent can make every good, trade is worth doing only when agents differ
    # sharply in what they are good at. At 1.0 the sampled world is near-autarkic
    # (gain over autarky +29, x1.02, 64% of agents helped); at 2.0 it is +234,
    # x1.11, 86% helped, with higher total welfare as well.
    shape_spread: float = 2.0
    # Utility charged per unit of effort spent producing, so production is not
    # free. It is the threshold below which making something is not worth the
    # bother: a myopic agent produces good g only when `theta_g * e_ig` clears
    # it. Without it an agent always produces something, however worthless.
    # This is a *utility* sink, distinct from efficiency, which is the brief's
    # "who is good at what".
    effort_cost: float = 0.02


@dataclass
class MobilityConfig:
    """Travel speed. Distance is crossed at `mobility` units per tick."""

    mobility_mean: float = 1.0
    mobility_spread: float = 0.4
    travel_cost_per_tick: float = 0.01


@dataclass
class VisibilityConfig:
    """How much of a region's posted board each agent can see.

    `sight_i` is the informational counterpart to `mobility_i`, and it is drawn
    heterogeneously: **each agent has a different K**. Agents who see more of the
    board find better rates, which is a capability from which broker and
    arbitrageur roles can emerge rather than being assigned by hand.

    This is also the search-friction knob. Full visibility for everyone would
    largely dissolve the double coincidence of wants, which is the friction a
    medium of exchange exists to solve. `sight_mean = 0` disables the market
    entirely and is how the autarky counterfactual is run.
    """

    sight_mean: float = 3.0
    sight_spread: float = 0.6  # 0 gives every agent the same K


@dataclass
class MarketConfig:
    """The exchange mechanism."""

    # A trade must be worth strictly more than this to both sides. Zero is safe:
    # surplus is scored with true prices, so a good valued at zero contributes
    # exactly zero and can never be accepted for something valuable.
    min_surplus: float = 0.0


@dataclass
class TokenConfig:
    """The annealed-commodity-money experiment.

    The token is an ordinary good whose consumption weight is driven to
    `anneal_end` over `anneal_ticks`, starting at `anneal_start_tick`. If
    acceptance survives the anneal, it is held up by expectation alone.
    """

    token_good: int = 4  # elderberry
    anneal_start_tick: int = 0
    anneal_ticks: int = 0  # 0 => no anneal, token stays an ordinary good
    anneal_end: float = 0.0


@dataclass
class SinkConfig:
    """What destroys goods.

    `spoilage` doubles as the hoardability knob. A low-spoilage good is durable,
    corner-able, and a plausible store of value; a high-spoilage good cannot be
    monopolised and cannot be money.
    """

    spoilage: tuple[float, ...] = (0.02, 0.02, 0.02, 0.02, 0.005)


@dataclass
class RecordingConfig:
    """What the run recorder keeps.

    Per-agent snapshots are what the agent inspector reads. At 20 agents they
    are free; raise `snapshot_interval` above 1 when the population grows.
    """

    default_ticks: int = 1000
    snapshot_interval: int = 1  # ticks between per-agent snapshots
    record_trades: bool = True
    # Who-saw-whom is an (N, N) boolean per snapshot, so it is only worth
    # keeping for populations small enough to actually look at agent by agent.
    visibility_max_agents: int = 64


@dataclass
class WorldConfig:
    n_agents: int = 20
    goods: tuple[str, ...] = GOOD_NAMES
    seed: int = 0
    gamma: float = 0.99
    initial_inventory: float = 1.0

    topology: Topology = field(default_factory=Topology.line)
    preference: PreferenceConfig = field(default_factory=PreferenceConfig)
    production: ProductionConfig = field(default_factory=ProductionConfig)
    mobility: MobilityConfig = field(default_factory=MobilityConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    visibility: VisibilityConfig = field(default_factory=VisibilityConfig)
    token: TokenConfig = field(default_factory=TokenConfig)
    sink: SinkConfig = field(default_factory=SinkConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)

    @property
    def n_goods(self) -> int:
        return len(self.goods)

    def __post_init__(self) -> None:
        if len(self.sink.spoilage) != self.n_goods:
            raise ValueError(f"spoilage has {len(self.sink.spoilage)} entries, expected {self.n_goods}")
        if not 0 <= self.token.token_good < self.n_goods:
            raise ValueError("token_good is out of range")

    def spoilage_array(self) -> np.ndarray:
        return np.array(self.sink.spoilage, dtype=np.float32)

    def evolve(self, **changes) -> "WorldConfig":
        """A copy with top-level fields replaced."""
        return replace(self, **changes)
