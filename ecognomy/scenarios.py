"""Diagnostic scenarios: small hand-built worlds with known correct answers.

The sampled world tells you whether an economy formed. It cannot tell you *which
capability* a policy is missing, because everything varies at once. A scenario
fixes every preference, efficiency and holding by hand so exactly one capability
is under test, and the answer is known in advance.

Each scenario records whether it is solvable by direct exchange alone. That is the
discriminator between rungs: a policy that cannot hold a good it does not consume
will solve every bilaterally-solvable scenario and none of the others, no matter
how well it is tuned.

Populations are deliberately tiny (2-5 agents) so a failure can be read off the
step-by-step viewer directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecognomy.config import SinkConfig, TokenConfig, WorldConfig
from ecognomy.topology import Topology


@dataclass(frozen=True)
class Scenario:
    """A fully specified world. Nothing is sampled."""

    name: str
    description: str
    goods: tuple[str, ...]
    theta: np.ndarray        # (N, G) preference weights
    efficiency: np.ndarray   # (N, G) units produced per tick of effort
    inventory: np.ndarray    # (N, G) starting holdings
    region: np.ndarray       # (N,)
    topology: Topology
    solvable_bilaterally: bool
    tests: str = ""

    @property
    def n_agents(self) -> int:
        return self.theta.shape[0]

    @property
    def n_goods(self) -> int:
        return self.theta.shape[1]

    def config(self, **overrides) -> WorldConfig:
        # Spoilage and token index are validated at construction, so they have
        # to be sized to this scenario's goods before the config is built.
        cfg = WorldConfig(
            n_agents=self.n_agents,
            goods=self.goods,
            topology=self.topology,
            seed=overrides.pop("seed", 0),
            sink=SinkConfig(spoilage=tuple([0.01] * self.n_goods)),
            token=TokenConfig(token_good=self.n_goods - 1),
        )
        cfg.production.n_producible = None  # the scenario sets efficiency directly
        for key, value in overrides.items():
            head, _, tail = key.partition("__")
            if tail:
                setattr(getattr(cfg, head), tail, value)
            else:
                setattr(cfg, head, value)
        return cfg

    def world(self, **overrides):
        from ecognomy.world import World

        return World(self.config(**overrides), scenario=self)

    def apply(self, world) -> None:
        """Overwrite a freshly spawned world with this scenario's exact values."""
        world.theta = self.theta.astype(np.float32).copy()
        world.theta_base = world.theta.copy()
        world.efficiency = self.efficiency.astype(np.float32).copy()
        world.inventory = self.inventory.astype(np.float32).copy()
        world.region = self.region.astype(np.int32).copy()
        world.edge = np.full(self.n_agents, -1, dtype=np.int32)
        world.progress = np.zeros(self.n_agents, dtype=np.float32)

    # ------------------------------------------------------------ structure

    def supply_graph(self) -> np.ndarray:
        """(N, N) boolean: can i supply something j actually wants?

        Supply counts both production *and* starting inventory. Checking only
        production is how an earlier version of this file claimed `triangular`
        had no bilateral opportunity while every agent was endowed with 0.5 of
        every good -- including ones they could not make and did not want, which
        were freely swappable. The structural claim and the actual initial state
        have to agree or the scenario silently stops discriminating.
        """
        supplies = (self.efficiency > 0) | (self.inventory > 0)
        wants = self.theta > 0
        return (supplies @ wants.T) > 0

    def has_bilateral_double_coincidence(self) -> bool:
        """Is there a pair who each produce something the other wants?

        This is the precondition for direct exchange. Without it, no amount of
        haggling produces a trade, and only a policy able to hold a good it does
        not consume can move anything.
        """
        g = self.supply_graph()
        np.fill_diagonal(g, False)
        return bool((g & g.T).any())

    def has_supply_cycle(self) -> bool:
        """Is there a directed cycle in the supply graph?

        A cycle with no bilateral edge is the signature of a world that can only
        be served by indirect exchange.
        """
        g = self.supply_graph().copy()
        np.fill_diagonal(g, False)
        reach = g.copy()
        for _ in range(self.n_agents):
            reach = reach | (reach @ g)
        return bool(np.diag(reach).any())


def _rows(*rows) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


# --------------------------------------------------------------- scenarios

def mutual_gains() -> Scenario:
    """Two mirrored pairs. Each agent makes what it barely wants.

    The cleanest possible gains-from-trade setup: everyone is *good at producing
    what they do not enjoy*, and the thing they want is made by someone standing
    next to them. Direct exchange solves it, so every rung from the myopic one up
    should trade here. A policy that fails this is broken, not merely limited.
    """
    theta = _rows([0.05, 0.95], [0.95, 0.05], [0.05, 0.95], [0.95, 0.05])
    eff = _rows([1.5, 0.0], [0.0, 1.5], [1.5, 0.0], [0.0, 1.5])
    inv = np.where(eff > 0, 0.5, 0.0)
    return Scenario(
        name="mutual_gains",
        description="Four agents, two goods. Each produces the good it does not want, "
                    "and wants the good its neighbour produces.",
        goods=("apple", "banana"),
        theta=theta, efficiency=eff, inventory=inv,
        region=np.zeros(4, dtype=np.int32),
        topology=Topology.line(1),
        solvable_bilaterally=True,
        tests="direct exchange",
    )


def comparative_advantage() -> Scenario:
    """Both agents can make both goods. Neither is dependent on the other.

    Each is better at producing the good the *other* wants, so the profitable
    move is to specialise away from your own taste and trade back. Unlike
    `mutual_gains`, nobody is locked out of any good -- autarky is survivable
    here, so this tests whether trade happens when it is merely advantageous
    rather than necessary.

    Preferences must differ for this to have any content. Reward is linear, so
    gains from trade come only from valuing goods differently; two agents with
    identical tastes have nothing to gain however lopsided their holdings.
    """
    theta = _rows([0.85, 0.15], [0.15, 0.85])   # a0 wants apple, a1 wants banana
    eff = _rows([0.40, 2.50], [2.50, 0.40])     # but each is better at the other's
    inv = np.full((2, 2), 0.5)
    return Scenario(
        name="comparative_advantage",
        description="Two agents, two goods. Each can make both, but is better at "
                    "producing the one the other wants.",
        goods=("apple", "banana"),
        theta=theta, efficiency=eff, inventory=inv,
        region=np.zeros(2, dtype=np.int32),
        topology=Topology.line(1),
        solvable_bilaterally=True,
        tests="specialisation without dependence",
    )


def triangular() -> Scenario:
    """A three-cycle of wants with no bilateral overlap anywhere.

    Agent 0 makes apple and wants banana; 1 makes banana and wants cherry; 2
    makes cherry and wants apple. Every unwanted good has weight exactly zero, so
    no pair can trade directly -- whoever receives is always handed something
    worthless to them.

    The only route is indirect: 0 gives apple to 2 (who wants it) for cherry
    (which 0 does not consume), then spends that cherry with 1 for banana. That
    requires holding a good purely for its exchange value, which is the same
    capability money needs.

    **This is the discriminator.** A policy that scores goods only by consumption
    cannot solve this at any parameter setting.
    """
    theta = _rows([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    eff = _rows([1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5])
    # Own production only. Endowing everyone with a little of every good would
    # hand each pair something the other wants and destroy the cycle.
    inv = np.where(eff > 0, 0.5, 0.0)
    return Scenario(
        name="triangular",
        description="Three agents, three goods, wants arranged in a cycle. No pair "
                    "has anything to offer each other directly.",
        goods=("apple", "banana", "cherry"),
        theta=theta, efficiency=eff, inventory=inv,
        region=np.zeros(3, dtype=np.int32),
        topology=Topology.line(1),
        solvable_bilaterally=False,
        tests="indirect exchange",
    )


def triangular_with_token() -> Scenario:
    """The three-cycle plus a fourth good nobody consumes.

    Same cycle as `triangular`, but a fourth good exists with zero consumption
    value to everyone, held by all in equal measure. Nothing forces its use and
    it can never be eaten, so any volume it carries is pure medium-of-exchange
    behaviour. Five agents: the cycle plus two agents holding only the token.
    """
    theta = _rows(
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.4, 0.3, 0.3, 0.0],
        [0.3, 0.4, 0.3, 0.0],
    )
    eff = _rows(
        [1.5, 0.0, 0.0, 0.0],
        [0.0, 1.5, 0.0, 0.0],
        [0.0, 0.0, 1.5, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    )
    # Cycle agents hold only what they make, plus the token. The token-rich pair
    # hold nothing but token, so any volume it carries is exchange behaviour.
    inv = np.array([
        [0.5, 0.0, 0.0, 2.0],
        [0.0, 0.5, 0.0, 2.0],
        [0.0, 0.0, 0.5, 2.0],
        [0.0, 0.0, 0.0, 6.0],
        [0.0, 0.0, 0.0, 6.0],
    ])
    return Scenario(
        name="triangular_with_token",
        description="The three-cycle plus two token-rich agents and a good with no "
                    "consumption value to anyone.",
        goods=("apple", "banana", "cherry", "token"),
        theta=theta, efficiency=eff, inventory=inv,
        region=np.zeros(5, dtype=np.int32),
        topology=Topology.line(1),
        solvable_bilaterally=False,
        tests="medium of exchange",
    )


def autarky() -> Scenario:
    """Negative control: everyone already makes exactly what they want.

    No policy should trade here. Trades in this scenario mean the mechanism is
    executing exchanges that make somebody worse off.
    """
    theta = _rows([1.0, 0.0], [0.0, 1.0])
    eff = _rows([1.5, 0.0], [0.0, 1.5])
    # Own production only, or each would hold a useless good the other wants and
    # the "self-sufficient" control would have a genuine trade available.
    inv = np.where(eff > 0, 0.5, 0.0)
    return Scenario(
        name="autarky",
        description="Two agents, each self-sufficient in the only good it wants.",
        goods=("apple", "banana"),
        theta=theta, efficiency=eff, inventory=inv,
        region=np.zeros(2, dtype=np.int32),
        topology=Topology.line(1),
        solvable_bilaterally=False,
        tests="negative control",
    )


ALL = {s().name: s for s in (
    mutual_gains, comparative_advantage, triangular, triangular_with_token, autarky,
)}


def get(name: str) -> Scenario:
    if name not in ALL:
        raise KeyError(f"unknown scenario {name!r}; have {sorted(ALL)}")
    return ALL[name]()
